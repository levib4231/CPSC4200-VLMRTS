from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import os
import re
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv


RCD_BASE_URL = "https://llm.rcd.clemson.edu/v1"
DEFAULT_MODEL = "qwen3-omni-30b-a3b"
DEFAULT_SERVER_URL = "http://127.0.0.1:8000"
DEFAULT_MAP = "vlm-singles.oramap"
BUNDLED_MAP = Path(__file__).resolve().parent / "maps" / DEFAULT_MAP
ALLOWED_ACTIONS = {"ATTACK", "MOVE", "RETREAT", "HOLD"}
OPENRA_BOT_TYPES = {
    "beginner": "normal",
    "easy": "normal",
    "medium": "normal",
    "hard": "normal",
    "normal": "normal",
    "brutal": "rush",
    "rush": "rush",
    "turtle": "turtle",
    "naval": "naval",
    "dummy": "dummy",
}

SYSTEM_PROMPT = """You control one Red Alert player through OpenRA-RL.
Choose exactly one constrained action: ATTACK, MOVE, RETREAT, or HOLD.
Return one JSON object and no markdown:
{"action":"ATTACK|MOVE|RETREAT|HOLD","unit_ids":[101],"target_id":201,"target":{"x":0,"y":0},"reason":"brief observable reason"}
Use only actor ids and map cells in the supplied state. ATTACK requires a visible
enemy target_id. MOVE requires a map-cell target. RETREAT sends units toward the
friendly base; target is optional. HOLD stops selected units. If no enemy is visible,
use MOVE to scout toward the far side of the map when combat units are available.
Never invent ids. Keep the reason under 20 words.
"""


class ActionParseError(ValueError):
    pass


@dataclass(frozen=True)
class Action:
    action: str
    unit_ids: list[int]
    target_id: int | None = None
    target: tuple[float, float] | None = None
    reason: str = ""


def _json_from_text(text: str) -> Mapping[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            raise ActionParseError("response did not contain a JSON object")
        try:
            value, _ = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError as exc:
            raise ActionParseError(f"invalid JSON action: {exc.msg}") from exc
    if not isinstance(value, Mapping):
        raise ActionParseError("action must be a JSON object")
    return value


def parse_action(value: str | Mapping[str, Any]) -> Action:
    obj = dict(value) if isinstance(value, Mapping) else dict(_json_from_text(value))
    action = str(obj.get("action", "")).upper().strip()
    if action not in ALLOWED_ACTIONS:
        raise ActionParseError(f"unsupported action {action!r}")

    raw_ids = obj.get("unit_ids", []) or []
    if not isinstance(raw_ids, list):
        raise ActionParseError("unit_ids must be a list")
    unit_ids: list[int] = []
    for unit_id in raw_ids:
        if isinstance(unit_id, bool) or not isinstance(unit_id, (int, float)) or int(unit_id) != unit_id:
            raise ActionParseError("unit_ids must contain integer ids")
        unit_ids.append(int(unit_id))

    target_id = obj.get("target_id")
    if target_id is not None:
        if isinstance(target_id, bool) or not isinstance(target_id, (int, float)) or int(target_id) != target_id:
            raise ActionParseError("target_id must be an integer")
        target_id = int(target_id)

    target_value = obj.get("target")
    target: tuple[float, float] | None = None
    if target_value is not None:
        if isinstance(target_value, Mapping):
            x, y = target_value.get("x"), target_value.get("y")
        elif isinstance(target_value, (list, tuple)) and len(target_value) == 2:
            x, y = target_value
        else:
            raise ActionParseError("target must contain x and y")
        try:
            target = (float(x), float(y))
        except (TypeError, ValueError) as exc:
            raise ActionParseError("target coordinates must be numbers") from exc
        if not all(math.isfinite(v) for v in target):
            raise ActionParseError("target coordinates must be finite")

    return Action(
        action=action,
        unit_ids=unit_ids,
        target_id=target_id,
        target=target,
        reason=str(obj.get("reason", "")).strip()[:300],
    )


def _units(state: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    values = state.get(key) or []
    if isinstance(values, list):
        return [item for item in values if isinstance(item, Mapping)]
    return []


def _unit_id(item: Mapping[str, Any]) -> int | None:
    value = item.get("id", item.get("actor_id"))
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _friendly_ids(state: Mapping[str, Any]) -> list[int]:
    result = []
    for item in _units(state, "units_summary") or _units(state, "units"):
        value = _unit_id(item)
        if value is not None:
            result.append(value)
    return result


def _enemy_ids(state: Mapping[str, Any]) -> list[int]:
    result = []
    seen: set[int] = set()
    for key in ("enemy_summary", "enemy_buildings_summary", "visible_enemies"):
        for item in _units(state, key):
            value = _unit_id(item)
            if value is not None and value not in seen:
                seen.add(value)
                result.append(value)
    return result


def _combat_ids(state: Mapping[str, Any]) -> list[int]:
    result = []
    for item in _units(state, "units_summary") or _units(state, "units"):
        value = _unit_id(item)
        if value is not None and item.get("can_attack", False):
            result.append(value)
    return result


def _map_size(state: Mapping[str, Any]) -> tuple[int, int]:
    value = state.get("map") or state.get("map_info") or {}
    if not isinstance(value, Mapping):
        return 0, 0
    try:
        return int(value.get("width", value.get("x", 0))), int(value.get("height", value.get("y", 0)))
    except (TypeError, ValueError):
        return 0, 0


def _base_target(state: Mapping[str, Any]) -> tuple[float, float]:
    buildings = _units(state, "buildings_summary") or _units(state, "buildings")
    preferred = next((b for b in buildings if b.get("type") == "fact"), None)
    anchor = preferred or (buildings[0] if buildings else None)
    if anchor is None:
        own = _units(state, "units_summary") or _units(state, "units")
        anchor = own[0] if own else None
    if anchor is not None:
        try:
            return float(anchor.get("cell_x", anchor.get("x", 0))), float(anchor.get("cell_y", anchor.get("y", 0)))
        except (TypeError, ValueError):
            pass
    width, height = _map_size(state)
    return width / 2.0, height / 2.0


def _scout_target(state: Mapping[str, Any]) -> tuple[float, float]:
    width, height = _map_size(state)
    base_x, _ = _base_target(state)
    # Singles-style maps place the two starts in opposite upper quadrants,
    # rather than at diagonally opposite corners.  Aim at the likely enemy
    # spawn so a MOVE fallback actually discovers the opponent.
    if base_x >= width / 2.0:
        x, y = width * 0.10, height * 0.30
    else:
        x, y = width * 0.85, height * 0.20
    return max(0.0, min(float(max(width - 1, 0)), x)), max(
        0.0, min(float(max(height - 1, 0)), y)
    )


def fallback_action(state: Mapping[str, Any]) -> Action:
    friendly = _friendly_ids(state)
    enemy = _enemy_ids(state)
    if friendly and enemy:
        return Action("ATTACK", friendly, target_id=enemy[0], reason="safe local fallback")
    combat = _combat_ids(state)
    if combat:
        return Action("MOVE", combat, target=_scout_target(state), reason="scout for enemy")
    return Action("HOLD", friendly, reason="no combat units")


def sanitize_action(action: Action, state: Mapping[str, Any]) -> Action:
    friendly = _friendly_ids(state)
    enemy = _enemy_ids(state)
    selected = [unit_id for unit_id in action.unit_ids if unit_id in friendly]
    if not selected:
        selected = friendly

    if action.action == "ATTACK":
        target_id = action.target_id if action.target_id in enemy else (enemy[0] if enemy else None)
        if target_id is None:
            combat = _combat_ids(state)
            if combat:
                return Action("MOVE", combat, target=_scout_target(state), reason="scout for enemy")
            return Action("HOLD", selected, reason="no visible enemy")
        return replace(action, unit_ids=selected, target_id=target_id, target=None)

    if action.action == "HOLD" and not enemy:
        combat = _combat_ids(state)
        if combat:
            return Action("MOVE", combat, target=_scout_target(state), reason="scout for enemy")

    if action.action == "RETREAT":
        return replace(action, unit_ids=selected, target=_base_target(state), target_id=None)

    if action.action == "MOVE" and action.target is None:
        action = replace(action, target=_base_target(state))
    width, height = _map_size(state)
    if action.target is not None and width > 0 and height > 0:
        action = replace(
            action,
            target=(
                max(0.0, min(float(width - 1), action.target[0])),
                max(0.0, min(float(height - 1), action.target[1])),
            ),
        )
    return replace(action, unit_ids=selected)


def _message_text(payload: Mapping[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError("LLM response contained no choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        return "".join(str(part.get("text", "")) if isinstance(part, Mapping) else str(part) for part in content)
    if content:
        return str(content)
    # Some Qwen-compatible endpoints put the answer in reasoning when thinking
    # is enabled. This fallback keeps the parser useful without logging secrets.
    return str(message.get("reasoning", ""))


class RCDVLM:
    def __init__(self, api_key: str, model: str, base_url: str = RCD_BASE_URL, timeout: float = 45.0) -> None:
        self.api_key = api_key
        self.model = model
        root = base_url.rstrip("/")
        self.url = root if root.endswith("/chat/completions") else root + "/chat/completions"
        self.timeout = timeout
        self.session = requests.Session()

    def choose(self, state: Mapping[str, Any]) -> tuple[Action, str]:
        prompt = (
            "Structured OpenRA-RL game state (authoritative for actor ids and map cells):\n"
            + json.dumps(state, separators=(",", ":"), sort_keys=True)
            + "\nReturn the JSON action now."
        )
        request = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 1024,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        response = self.session.post(self.url, headers=headers, json=request, timeout=self.timeout)
        response.raise_for_status()
        text = _message_text(response.json())
        return parse_action(text), text


class BootstrapController:
    """Deterministic economy/army setup around the constrained combat policy.

    The VLM remains limited to ATTACK, MOVE, RETREAT, and HOLD. OpenRA-RL
    requires an MCV deployment and production setup before those actions can
    produce a complete game, so this controller handles the mechanical opening.
    """

    def __init__(self) -> None:
        self.training_targets = {"e1": 6, "1tnk": 4}
        self.last_deploy_attempt_tick = -10_000
        self.retry_after_tick = 0

    @staticmethod
    def _production_items(state: Mapping[str, Any]) -> set[str]:
        items = set()
        for value in state.get("production_items", []) or []:
            if isinstance(value, str):
                items.add(value.split("@", 1)[0])
        return items

    @staticmethod
    def _building_types(state: Mapping[str, Any]) -> set[str]:
        return {str(value) for value in (state.get("building_types") or [])}

    @staticmethod
    def _available(state: Mapping[str, Any]) -> set[str]:
        return {str(value) for value in (state.get("available_production") or [])}

    async def step(self, game: Any, state: Mapping[str, Any]) -> dict[str, Any] | None:
        tick = int(state.get("tick", 0) or 0)
        if tick < self.retry_after_tick:
            return None

        units = _units(state, "units_summary")
        buildings = self._building_types(state)
        production = self._production_items(state)
        available = self._available(state)

        if "fact" not in buildings:
            mcv = next((item for item in units if item.get("type") == "mcv"), None)
            if tick >= self.last_deploy_attempt_tick + 100 and mcv is not None and _unit_id(mcv) is not None:
                unit_id = _unit_id(mcv)
                result = await game.call_tool("deploy_unit", unit_id=unit_id)
                self.last_deploy_attempt_tick = tick
                self._record_result(tick, result)
                return {"phase": "deploy_mcv", "unit_id": unit_id, "result": result}
            return None

        barracks = next((name for name in ("tent", "barr") if name in buildings), None)
        if "powr" not in buildings and "powr" not in production and "powr" in available:
            result = await game.call_tool("build_and_place", building_type="powr")
            self._record_result(tick, result)
            return {"phase": "build_base", "building": "powr", "result": result}
        if barracks is None:
            barracks_type = next((name for name in ("tent", "barr") if name in available), None)
            if barracks_type and barracks_type not in production:
                result = await game.call_tool("build_and_place", building_type=barracks_type)
                self._record_result(tick, result)
                return {"phase": "build_base", "building": barracks_type, "result": result}
            return None
        if "proc" not in buildings and "proc" not in production and "proc" in available:
            result = await game.call_tool("build_and_place", building_type="proc")
            self._record_result(tick, result)
            return {"phase": "build_base", "building": "proc", "result": result}
        if "weap" not in buildings and "weap" not in production and "weap" in available:
            result = await game.call_tool("build_and_place", building_type="weap")
            self._record_result(tick, result)
            return {"phase": "build_base", "building": "weap", "result": result}

        own_units = _units(state, "units_summary")
        for item_type, target_count in self.training_targets.items():
            count = sum(1 for unit in own_units if unit.get("type") == item_type)
            if count >= target_count or item_type in production or item_type not in available:
                continue
            result = await game.call_tool("build_unit", unit_type=item_type, count=1)
            self._record_result(tick, result)
            return {"phase": "train_army", "unit_type": item_type, "count": count + 1, "result": result}
        return None

    def _record_result(self, tick: int, result: Any) -> None:
        if isinstance(result, Mapping) and result.get("error"):
            self.retry_after_tick = tick + 250


class DecisionEngine:
    def __init__(self, vlm: RCDVLM | None, output_dir: Path) -> None:
        self.vlm = vlm
        self.output_dir = output_dir
        self.log_path = output_dir / "events.jsonl"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def log(self, record: Mapping[str, Any]) -> None:
        row = {"timestamp": datetime.now(timezone.utc).isoformat(), **dict(record)}
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")

    def decide(self, state: Mapping[str, Any]) -> Action:
        started = time.perf_counter()
        raw_text = ""
        error = None
        if self.vlm is None:
            action = fallback_action(state)
        else:
            try:
                action, raw_text = self.vlm.choose(state)
            except Exception as exc:  # keep the episode alive if the model fails
                action, error = fallback_action(state), str(exc)
        action = sanitize_action(action, state)
        self.log(
            {
                "event": "decision",
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
                "state": state,
                "action": asdict(action),
                "raw_model_response": raw_text[:2000],
                "error": error,
            }
        )
        return action


def _load_credentials() -> str:
    env_file = os.getenv("VLM_ENV_FILE")
    if env_file:
        load_dotenv(env_file, override=False)
    else:
        load_dotenv(override=False)
    key = os.getenv("RCD_LLM_API_KEY") or os.getenv("COURSE_LLM_API_KEY") or os.getenv("LLM_API_KEY")
    if not key:
        raise SystemExit("RCD key missing: set RCD_LLM_API_KEY or COURSE_LLM_API_KEY")
    return key


def _server_port(server_url: str) -> int:
    parsed = urlparse(server_url)
    if parsed.port:
        return parsed.port
    return 443 if parsed.scheme == "https" else 80


def _server_bot_type(difficulty: str) -> str:
    return OPENRA_BOT_TYPES.get(difficulty.lower(), difficulty)


def _server_ready(server_url: str) -> bool:
    try:
        response = requests.get(server_url.rstrip("/") + "/health", timeout=3)
        return response.ok
    except requests.RequestException:
        return False


def _ensure_server(server_url: str, difficulty: str, start_server: bool) -> bool:
    if _server_ready(server_url):
        return False
    if not start_server:
        raise SystemExit(
            f"OpenRA-RL server is not reachable at {server_url}. "
            "Start it with `openra-rl server start` or pass --start-server."
        )
    from openra_env.cli.docker_manager import start_server as docker_start_server, wait_for_health

    if "localhost" not in server_url and "127.0.0.1" not in server_url:
        raise SystemExit("--start-server only supports a local OpenRA-RL server URL")
    if not docker_start_server(port=_server_port(server_url), difficulty=_server_bot_type(difficulty)):
        raise SystemExit("Could not start the OpenRA-RL Docker server")
    if not wait_for_health(port=_server_port(server_url), timeout=120):
        raise SystemExit("OpenRA-RL server did not become healthy")
    return True


def _configure_local_server(server_url: str, map_name: str, difficulty: str) -> None:
    """Configure the legacy OpenRA-RL image used by the Docker server.

    The current image creates environments from its container-side config and
    does not forward map_name/map_data through reset(), so install the scenario
    and config before opening the agent WebSocket.
    """
    parsed = urlparse(server_url)
    if parsed.hostname not in {"localhost", "127.0.0.1"}:
        return

    config = (
        "game:\n"
        "  openra_path: /opt/openra\n"
        f"  map_name: {json.dumps(map_name)}\n"
        "  headless: true\n"
        "opponent:\n"
        f"  bot_type: {json.dumps(_server_bot_type(difficulty))}\n"
        "  ai_slot: Multi0\n"
        "planning:\n"
        "  enabled: false\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
        handle.write(config)
        config_path = handle.name
    try:
        result = subprocess.run(
            ["docker", "cp", config_path, "openra-rl-server:/app/config.yaml"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise SystemExit(f"Could not configure the OpenRA-RL Docker server: {result.stderr.strip()}")

        if map_name == DEFAULT_MAP and BUNDLED_MAP.is_file():
            result = subprocess.run(
                ["docker", "cp", str(BUNDLED_MAP), f"openra-rl-server:/opt/openra/mods/ra/maps/{DEFAULT_MAP}"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                raise SystemExit(f"Could not install the bundled OpenRA map: {result.stderr.strip()}")
    finally:
        Path(config_path).unlink(missing_ok=True)


async def _prepare_game(game: Any) -> dict[str, Any]:
    state = await game.call_tool("get_game_state")
    try:
        planning = await game.call_tool("start_planning_phase")
    except Exception:
        planning = {"planning_enabled": False}
    if isinstance(planning, Mapping) and planning.get("planning_active"):
        await game.call_tool(
            "end_planning_phase",
            strategy="Use only ATTACK, MOVE, RETREAT, and HOLD; prioritize visible enemy units.",
        )
    else:
        await game.call_tool("advance", ticks=1)
    return await game.call_tool("get_game_state")


async def _issue_action(game: Any, action: Action) -> Any:
    if not action.unit_ids:
        return {"note": "no friendly units"}
    unit_ids = ",".join(str(unit_id) for unit_id in action.unit_ids)
    if action.action == "ATTACK" and action.target_id is not None:
        return await game.call_tool("attack_target", unit_ids=unit_ids, target_actor_id=action.target_id)
    if action.action in {"MOVE", "RETREAT"} and action.target is not None:
        return await game.call_tool(
            "move_units",
            unit_ids=unit_ids,
            target_x=round(action.target[0]),
            target_y=round(action.target[1]),
        )
    return await game.call_tool("stop_units", unit_ids=unit_ids)


async def run_episode(args: argparse.Namespace, engine: DecisionEngine) -> int:
    from openra_env.mcp_ws_client import OpenRAMCPClient

    bootstrap = BootstrapController() if args.bootstrap else None
    reset_args: dict[str, Any] = {"bot_type": args.difficulty, "map_name": args.map}
    if args.map == DEFAULT_MAP and BUNDLED_MAP.is_file():
        reset_args["map_data"] = base64.b64encode(BUNDLED_MAP.read_bytes()).decode("ascii")
    async with OpenRAMCPClient(
        args.server_url,
        message_timeout_s=max(120.0, float(args.max_seconds) + 60.0),
    ) as game:
        await game.reset(**reset_args)
        state = await _prepare_game(game)
        engine.log({"event": "game_started", "state": state, "server_url": args.server_url})
        started = time.monotonic()
        turns = 0
        while not state.get("done"):
            if args.max_seconds > 0 and time.monotonic() - started >= args.max_seconds:
                engine.log({"event": "time_limit", "turns": turns, "tick": state.get("tick", 0)})
                return 0
            if args.max_turns > 0 and turns >= args.max_turns:
                engine.log({"event": "turn_limit", "turns": turns, "tick": state.get("tick", 0)})
                return 0

            bootstrap_result = None
            if bootstrap is not None:
                bootstrap_result = await bootstrap.step(game, state)
                if bootstrap_result is not None:
                    engine.log({"event": "bootstrap", "tick": state.get("tick", 0), **bootstrap_result})
                    state = await game.call_tool("get_game_state")

            # Do not issue a combat order on the same tick as MCV deployment;
            # Stop/Move can cancel the deployment before it becomes a fact.
            if bootstrap_result and bootstrap_result.get("phase") == "deploy_mcv":
                action = Action("HOLD", [], reason="bootstrap deployment")
                action_result = {"note": "combat order skipped during MCV deployment"}
            else:
                action = await asyncio.to_thread(engine.decide, state)
                action_result = await _issue_action(game, action)
            advance_result = await game.call_tool("advance", ticks=args.advance_ticks)
            engine.log(
                {
                    "event": "step",
                    "turn": turns,
                    "tick": advance_result.get("tick") if isinstance(advance_result, Mapping) else None,
                    "action": asdict(action),
                    "action_result": action_result,
                    "advance_result": advance_result,
                }
            )
            state = await game.call_tool("get_game_state")
            turns += 1

        engine.log({"event": "game_finished", "result": state.get("result", ""), "tick": state.get("tick", 0)})
        print(json.dumps({"result": state.get("result", ""), "tick": state.get("tick", 0), "turns": turns}))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Constrained VLM agent for OpenRA-RL")
    parser.add_argument("--server-url", default=os.getenv("OPENRA_URL", DEFAULT_SERVER_URL))
    parser.add_argument("--difficulty", "--bot-type", default=os.getenv("BOT_TYPE", "normal"))
    parser.add_argument("--map", default=os.getenv("OPENRA_MAP", DEFAULT_MAP))
    parser.add_argument("--model", default=os.getenv("RCD_VLM_MODEL") or os.getenv("RCD_LLM_MODEL") or DEFAULT_MODEL)
    parser.add_argument("--base-url", default=os.getenv("RCD_LLM_BASE_URL", RCD_BASE_URL))
    parser.add_argument("--request-timeout", type=float, default=45.0)
    parser.add_argument("--max-seconds", type=int, default=60)
    parser.add_argument("--max-turns", type=int, default=0)
    parser.add_argument("--advance-ticks", type=int, default=25)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-vlm", action="store_true", help="use deterministic fallback actions")
    parser.add_argument("--no-bootstrap", dest="bootstrap", action="store_false", help="disable automatic MCV deployment and base/army setup")
    parser.set_defaults(bootstrap=True)
    parser.add_argument("--start-server", action="store_true", help="start the official OpenRA-RL Docker server")
    parser.add_argument("--headless", action="store_true", help="accepted for compatibility; OpenRA-RL is headless by default")
    args = parser.parse_args(argv)
    args.advance_ticks = max(1, min(args.advance_ticks, 50))

    output_dir = args.output_dir or Path("runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    started_server = _ensure_server(args.server_url, args.difficulty, args.start_server)
    del started_server  # The official server is intentionally left running for reuse.
    _configure_local_server(args.server_url, args.map, args.difficulty)
    vlm = None if args.no_vlm else RCDVLM(_load_credentials(), args.model, args.base_url, args.request_timeout)
    return asyncio.run(run_episode(args, DecisionEngine(vlm, output_dir)))


if __name__ == "__main__":
    raise SystemExit(main())
