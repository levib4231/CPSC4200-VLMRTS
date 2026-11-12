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
ALLOWED_ACTIONS = {"ATTACK", "MOVE", "RETREAT", "HOLD", "BUILD_UNIT", "BUILD_STRUCTURE", "DEPLOY"}
PERSONALITIES = {"selfish", "helpful", "balanced"}
PERSONALITY_PROFILES = {
    "selfish": {"ally_threat_radius": 0.0, "own_base_priority_radius": 18.0, "retreat_when_unthreatened": True},
    "helpful": {"ally_threat_radius": 12.0, "own_base_priority_radius": 18.0, "retreat_when_unthreatened": False},
    "balanced": {"ally_threat_radius": 8.0, "own_base_priority_radius": 18.0, "retreat_when_unthreatened": False},
}
ECONOMY_CONTROLS = {"bootstrap", "hybrid", "vlm"}
MAX_OWN_UNITS = 32
MIN_SPEND_RESERVE = 500
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
Choose exactly one action: ATTACK, MOVE, RETREAT, HOLD, BUILD_UNIT,
BUILD_STRUCTURE, or DEPLOY. Return one JSON object and no markdown:
{"action":"ATTACK|MOVE|RETREAT|HOLD|BUILD_UNIT|BUILD_STRUCTURE|DEPLOY","unit_ids":[101],"target_id":201,"target":{"x":0,"y":0},"unit_type":"e1","structure_type":"powr","count":1,"reason":"brief observable reason"}
Use only supplied actor ids and available_production. BUILD_UNIT needs unit_type,
BUILD_STRUCTURE needs structure_type, and DEPLOY needs an owned deployable unit id.
ATTACK needs a visible enemy; MOVE needs a map cell; RETREAT returns to base;
HOLD stops units. Never invent ids. Keep reason under 20 words.
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
    unit_type: str | None = None
    structure_type: str | None = None
    count: int = 1


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

    count = obj.get("count", 1)
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 5:
        raise ActionParseError("count must be an integer from 1 to 5")
    return Action(
        action=action,
        unit_ids=unit_ids,
        target_id=target_id,
        target=target,
        reason=str(obj.get("reason", "")).strip()[:300],
        unit_type=str(obj["unit_type"]).lower() if obj.get("unit_type") else None,
        structure_type=str(obj["structure_type"]).lower() if obj.get("structure_type") else None,
        count=count,
    )


def _units(state: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    values = state.get(key) or []
    if isinstance(values, list):
        return [item for item in values if isinstance(item, Mapping)]
    return []


def _unit_id(item: Mapping[str, Any]) -> int | None:
    value = item.get("id", item.get("actor_id"))
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _own_units(state: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return _units(state, "units_summary") or _units(state, "units")


def _friendly_ids(state: Mapping[str, Any]) -> list[int]:
    return [value for item in _own_units(state) if (value := _unit_id(item)) is not None]


def _economy_cash(state: Mapping[str, Any]) -> float | None:
    economy = state.get("economy")
    cash = economy.get("cash") if isinstance(economy, Mapping) else state.get("cash", state.get("money"))
    return float(cash) if isinstance(cash, (int, float)) and not isinstance(cash, bool) else None


def _item_cost(action: Action, state: Mapping[str, Any]) -> int | None:
    advertised = state.get("production_costs") or state.get("item_costs") or {}
    key = action.unit_type if action.action == "BUILD_UNIT" else action.structure_type
    if isinstance(advertised, Mapping) and key in advertised and isinstance(advertised[key], (int, float)):
        return int(advertised[key])
    try:
        from openra_env.game_data import RA_BUILDINGS, RA_UNITS
        item = (RA_UNITS if action.action == "BUILD_UNIT" else RA_BUILDINGS).get(key or "")
        cost = item.get("cost") if item else None
        return int(cost) if isinstance(cost, (int, float)) else None
    except (ImportError, AttributeError):
        return None


def _enemy_ids(state: Mapping[str, Any]) -> list[int]:
    own_team = state.get("team", state.get("team_id"))
    own_player = state.get("player", state.get("player_id"))
    own_owners = {str(item.get("owner")) for item in (_units(state, "units_summary") or _units(state, "units")) if item.get("owner")}
    ally_ids = {_unit_id(item) for item in (_units(state, "allies_summary") or _units(state, "allied_units"))}
    result = []
    seen: set[int] = set()
    for key in ("enemy_summary", "enemy_buildings_summary", "visible_enemies", "visible_enemy_buildings"):
        for item in _units(state, key):
            team = item.get("team", item.get("team_id"))
            player = item.get("player", item.get("player_id"))
            if team is not None and own_team is not None and team == own_team:
                continue
            if player is not None and own_player is not None and player == own_player:
                continue
            if item.get("owner") and str(item.get("owner")) in own_owners:
                continue
            value = _unit_id(item)
            if value is not None and value not in seen and value not in ally_ids:
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


def _cell(item: Mapping[str, Any]) -> tuple[float, float] | None:
    try:
        x, y = float(item.get("cell_x", item.get("x"))), float(item.get("cell_y", item.get("y")))
        return (x, y) if math.isfinite(x) and math.isfinite(y) else None
    except (TypeError, ValueError):
        return None


def _distance(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    pa, pb = _cell(a), _cell(b)
    return math.dist(pa, pb) if pa and pb else math.inf


def fallback_action(state: Mapping[str, Any], personality: str = "balanced") -> Action:
    combat = _combat_ids(state)
    if not combat:
        return Action("HOLD", [], reason="no combat-capable units")
    enemies = [item for key in ("enemy_summary", "enemy_buildings_summary", "visible_enemies", "visible_enemy_buildings") for item in _units(state, key) if _unit_id(item) in _enemy_ids(state)]
    allies = _units(state, "allies_summary") or _units(state, "allied_units")
    own_base = _base_target(state)
    profile = PERSONALITY_PROFILES.get(personality, PERSONALITY_PROFILES["balanced"])
    threatened_base = any((_cell(e) and math.dist(_cell(e), own_base) <= profile["own_base_priority_radius"]) for e in enemies)
    if enemies and threatened_base:
        target = min(enemies, key=lambda enemy: _distance(enemy, {"cell_x": own_base[0], "cell_y": own_base[1]}))
        return Action("ATTACK", combat, target_id=_unit_id(target), reason="defend own base")
    ally_threats = [(ally, enemy) for ally in allies for enemy in enemies
                    if profile["ally_threat_radius"] > 0 and _distance(ally, enemy) <= profile["ally_threat_radius"]]
    if ally_threats:
        # Only use ally observations explicitly supplied by the environment.
        target = min(ally_threats, key=lambda pair: _distance(pair[0], pair[1]))[1]
        return Action("ATTACK", combat, target_id=_unit_id(target), reason="protect observed ally under threat")
    if enemies:
        if profile["retreat_when_unthreatened"]:
            return Action("RETREAT", combat, reason="preserve own forces")
        return Action("ATTACK", combat, target_id=_unit_id(enemies[0]), reason="safe local fallback")
    return Action("MOVE", combat, target=_scout_target(state), reason="scout for enemy")


def sanitize_action(action: Action, state: Mapping[str, Any]) -> Action:
    own = _own_units(state)
    by_id = {_unit_id(item): item for item in own if _unit_id(item) is not None}
    enemy = _enemy_ids(state)
    if action.action in {"BUILD_UNIT", "BUILD_STRUCTURE"}:
        key = action.unit_type if action.action == "BUILD_UNIT" else action.structure_type
        available = {str(value).lower() for value in (state.get("available_production") or [])}
        cost = _item_cost(action, state)
        cash = _economy_cash(state)
        # Team observations retain exact remaining costs; legacy MCP exposes
        # only production_items, for which full costs are a conservative bound.
        queued = state.get("production") or state.get("production_items") or []
        queued_types = {str(q.get("item", q.get("type", ""))).lower() for q in queued if isinstance(q, Mapping)}
        queued_types.update(str(q).split("@", 1)[0].lower() for q in queued if isinstance(q, str))
        if not key or key.lower() not in available or key.lower() in queued_types or cost is None or cash is None:
            return Action("HOLD", [], reason="economy order unavailable or cost unknown")
        pending_cost = 0
        for entry in queued:
            if isinstance(entry, Mapping):
                remaining = entry.get("remaining_cost")
                if not isinstance(remaining, (int, float)) or isinstance(remaining, bool):
                    return Action("HOLD", [], reason="queued production cost unknown")
                pending_cost += max(0, int(remaining))
            elif isinstance(entry, str):
                pending_item = entry.split("@", 1)[0].lower()
                pending_action = Action("BUILD_UNIT", [], unit_type=pending_item)
                pending_unit_cost = _item_cost(pending_action, state)
                if pending_unit_cost is None:
                    pending_action = Action("BUILD_STRUCTURE", [], structure_type=pending_item)
                    pending_unit_cost = _item_cost(pending_action, state)
                if pending_unit_cost is None:
                    return Action("HOLD", [], reason="queued production cost unknown")
                pending_cost += pending_unit_cost
        total_cost = cost * (action.count if action.action == "BUILD_UNIT" else 1)
        if cash - pending_cost - total_cost < MIN_SPEND_RESERVE:
            return Action("HOLD", [], reason="economy reserve would be breached")
        if action.action == "BUILD_UNIT":
            queued_units = 0
            for entry in queued:
                item = (entry.split("@", 1)[0] if isinstance(entry, str)
                        else entry.get("item", entry.get("type", "")))
                pending = Action("BUILD_UNIT", [], unit_type=str(item).lower())
                if _item_cost(pending, state) is not None:
                    count = entry.get("count", 1) if isinstance(entry, Mapping) else 1
                    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                        return Action("HOLD", [], reason="queued unit count unknown")
                    queued_units += count
            if len(own) + queued_units + action.count > MAX_OWN_UNITS:
                return Action("HOLD", [], reason="projected unit cap exceeded")
        width, height = _map_size(state)
        if action.target is not None:
            if width <= 0 or height <= 0 or not (0 <= action.target[0] < width and 0 <= action.target[1] < height):
                return Action("HOLD", [], reason="economy placement outside known map")
        return replace(action, unit_ids=[], count=action.count)

    if action.action == "DEPLOY":
        deployable = {i for i, item in by_id.items() if item.get("deployable") or item.get("type") == "mcv"}
        chosen = next((i for i in action.unit_ids if i in deployable), None)
        return replace(action, unit_ids=[chosen]) if chosen is not None else Action("HOLD", [], reason="no owned deployable unit")

    if action.action in {"ATTACK", "MOVE", "RETREAT", "HOLD"}:
        explicit = bool(action.unit_ids)
        if explicit and any(i not in by_id for i in action.unit_ids):
            return Action("HOLD", [], reason="unknown friendly actor id")
        if action.action == "ATTACK":
            allowed = {i for i, item in by_id.items() if item.get("can_attack", False)}
            selected = [i for i in action.unit_ids if i in allowed] if explicit else list(allowed)
        elif action.action in {"MOVE", "RETREAT"}:
            # Explicitly named own units may be noncombat (for example a
            # harvester); implicit selection is deliberately combat-only.
            allowed = set(by_id) if explicit else {i for i, item in by_id.items() if item.get("can_attack", False)}
            selected = [i for i in action.unit_ids if i in allowed] if explicit else list(allowed)
        else:
            selected = [i for i in action.unit_ids if i in by_id] if explicit else [i for i, item in by_id.items() if item.get("can_attack", False)]
        if not selected:
            return Action("HOLD", [], reason="no eligible owned units")
        if action.action == "ATTACK":
            target = action.target_id if action.target_id in enemy else (enemy[0] if enemy else None)
            if target is None:
                return Action("HOLD", selected, reason="no visible enemy")
            return replace(action, unit_ids=selected, target_id=target, target=None)
        if action.action == "RETREAT":
            return replace(action, unit_ids=selected, target=_base_target(state), target_id=None)
        if action.action == "MOVE":
            target = action.target or _base_target(state)
            width, height = _map_size(state)
            if width <= 0 or height <= 0 or not all(math.isfinite(v) for v in target):
                return Action("HOLD", [], reason="move requires known map coordinates")
            return replace(action, unit_ids=selected, target=(max(0.0, min(width - 1.0, target[0])), max(0.0, min(height - 1.0, target[1]))))
        return replace(action, unit_ids=selected)
    return Action("HOLD", [], reason="unsupported sanitized action")

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

    def choose(self, state: Mapping[str, Any], personality: str = "balanced") -> tuple[Action, str]:
        guidance = {
            "selfish": "Prioritize own base and army; retreat from enemies outside 18 cells of your base, and never assume ally support.",
            "helpful": "Defend own base if an enemy is within 18 cells; otherwise help explicitly observed allies threatened within 12 cells.",
            "balanced": "Defend own base within 18 cells; otherwise help explicitly observed allies threatened within 8 cells."
        }[personality]
        prompt = (
            f"Personality: {guidance}\\n"
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
        cash = _economy_cash(state)
        if cash is None or cash < MIN_SPEND_RESERVE:
            return None

        if "powr" not in buildings and "powr" not in production and "powr" in available and sanitize_action(Action("BUILD_STRUCTURE", [], structure_type="powr"), state).action == "BUILD_STRUCTURE":
            result = await game.call_tool("build_and_place", building_type="powr")
            self._record_result(tick, result)
            return {"phase": "build_base", "building": "powr", "result": result}
        if barracks is None:
            barracks_type = next((name for name in ("tent", "barr") if name in available), None)
            if barracks_type and barracks_type not in production and sanitize_action(Action("BUILD_STRUCTURE", [], structure_type=barracks_type), state).action == "BUILD_STRUCTURE":
                result = await game.call_tool("build_and_place", building_type=barracks_type)
                self._record_result(tick, result)
                return {"phase": "build_base", "building": barracks_type, "result": result}
            return None
        if "proc" not in buildings and "proc" not in production and "proc" in available and sanitize_action(Action("BUILD_STRUCTURE", [], structure_type="proc"), state).action == "BUILD_STRUCTURE":
            result = await game.call_tool("build_and_place", building_type="proc")
            self._record_result(tick, result)
            return {"phase": "build_base", "building": "proc", "result": result}
        if "weap" not in buildings and "weap" not in production and "weap" in available and sanitize_action(Action("BUILD_STRUCTURE", [], structure_type="weap"), state).action == "BUILD_STRUCTURE":
            result = await game.call_tool("build_and_place", building_type="weap")
            self._record_result(tick, result)
            return {"phase": "build_base", "building": "weap", "result": result}

        own_units = _units(state, "units_summary")
        for item_type, target_count in self.training_targets.items():
            count = sum(1 for unit in own_units if unit.get("type") == item_type)
            if len(own_units) >= MAX_OWN_UNITS:
                return None
            if count >= target_count or item_type in production or item_type not in available:
                continue
            if sanitize_action(Action("BUILD_UNIT", [], unit_type=item_type), state).action != "BUILD_UNIT":
                continue
            result = await game.call_tool("build_unit", unit_type=item_type, count=1)
            self._record_result(tick, result)
            return {"phase": "train_army", "unit_type": item_type, "count": count + 1, "result": result}
        return None

    def _record_result(self, tick: int, result: Any) -> None:
        if isinstance(result, Mapping) and result.get("error"):
            self.retry_after_tick = tick + 250


class DecisionEngine:
    def __init__(self, vlm: RCDVLM | None, output_dir: Path, personality: str = "balanced", economy_control: str = "bootstrap") -> None:
        if personality not in PERSONALITIES or economy_control not in ECONOMY_CONTROLS:
            raise ValueError("invalid personality or economy control")
        self.personality = personality
        self.economy_control = economy_control
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
            action = fallback_action(state, self.personality)
            source = "fallback"
        else:
            try:
                action, raw_text = self.vlm.choose(state, self.personality)
                source = "vlm"
            except Exception as exc:  # keep the episode alive if the model fails
                action, error = fallback_action(state, self.personality), str(exc)
                source = "fallback"
        proposed_action = action
        action = sanitize_action(action, state)
        if self.economy_control == "bootstrap" and action.action in {"BUILD_UNIT", "BUILD_STRUCTURE", "DEPLOY"}:
            action = Action("HOLD", [], reason="economy controlled by bootstrap")
        sanitized = proposed_action != action
        decision_source = "vlm_rejected" if sanitized and source == "vlm" else source
        self.log(
            {
                "event": "decision",
                "decision_source": decision_source,
                "personality": self.personality,
                "observation_counts": {"own_units": len(_own_units(state)), "visible_enemies": len(_enemy_ids(state)), "explicit_allies": len(_units(state, "allies_summary") or _units(state, "allied_units"))},
                "proposed_action": asdict(proposed_action),
                "action_sanitized": sanitized,
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


async def probe_server_capabilities(server_url: str) -> dict[str, Any]:
    """Report actual advertised tools and live observation fields without inventing support."""
    from openra_env.mcp_ws_client import OpenRAMCPClient
    async with OpenRAMCPClient(server_url) as game:
        tools = await game.list_tools(use_cache=False)
        try:
            state = await game.call_tool("get_game_state")
            if isinstance(state, Mapping) and state.get("error"):
                raise RuntimeError(str(state["error"]))
        except Exception as exc:
            state = {}
            observation_error = str(exc)
        else:
            observation_error = None
        fields = sorted(state.keys()) if isinstance(state, Mapping) else []
        allies = any(k in fields for k in ("allies_summary", "allied_units", "ally_observations"))
        return {"tools": sorted(tool.name for tool in tools), "observation_fields": fields,
                "explicit_ally_observations": allies, "multi_slot_team_control": False,
                "observation_error": observation_error}


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


def _defer_vlm_economy(action: Action, bootstrap_result: Mapping[str, Any] | None) -> Action:
    if bootstrap_result and action.action in {"BUILD_UNIT", "BUILD_STRUCTURE", "DEPLOY"}:
        return Action("HOLD", [], reason="VLM economy order deferred during bootstrap mutation")
    return action


def _suppress_order_during_deploy(action: Action, bootstrap_result: Mapping[str, Any] | None) -> Action:
    if bootstrap_result and str(bootstrap_result.get("phase", "")).startswith("deploy"):
        return Action("HOLD", [], reason="bootstrap deployment")
    return action


async def _issue_action(game: Any, action: Action) -> Any:
    if action.action == "BUILD_UNIT" and action.unit_type:
        return await game.call_tool("build_unit", unit_type=action.unit_type, count=action.count)
    if action.action == "BUILD_STRUCTURE" and action.structure_type:
        args: dict[str, Any] = {"building_type": action.structure_type}
        if action.target is not None:
            args.update(cell_x=round(action.target[0]), cell_y=round(action.target[1]))
        return await game.call_tool("build_and_place", **args)
    if action.action == "DEPLOY" and action.unit_ids:
        return await game.call_tool("deploy_unit", unit_id=action.unit_ids[0])
    if not action.unit_ids:
        return {"note": "no eligible friendly units"}
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

    bootstrap = BootstrapController() if args.economy_control in {"bootstrap", "hybrid"} and args.bootstrap else None
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
            if bootstrap_result and str(bootstrap_result.get("phase", "")).startswith("deploy"):
                action = Action("HOLD", [], reason="bootstrap deployment")
                action_result = {"note": "all other orders skipped during deployment"}
            else:
                action = await asyncio.to_thread(engine.decide, state)
                deferred = _defer_vlm_economy(action, bootstrap_result)
                if deferred != action:
                    engine.log({"event": "action_deferred", "decision_source": "deferred", "reason": deferred.reason,
                               "proposed_action": asdict(action), "bootstrap_phase": bootstrap_result.get("phase")})
                    action, action_result = deferred, {"note": "VLM economy order deferred; bootstrap mutated economy this tick"}
                else:
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
    parser.add_argument("--personality", choices=sorted(PERSONALITIES), default="balanced")
    parser.add_argument("--economy-control", choices=sorted(ECONOMY_CONTROLS), default="bootstrap")
    parser.add_argument("--no-bootstrap", dest="bootstrap", action="store_false", help="disable automatic MCV deployment and base/army setup")
    parser.set_defaults(bootstrap=True)
    parser.add_argument("--start-server", action="store_true", help="start the official OpenRA-RL Docker server")
    parser.add_argument("--headless", action="store_true", help="accepted for compatibility; OpenRA-RL is headless by default")
    parser.add_argument("--probe-capabilities", action="store_true", help="list live MCP tools and observed team capabilities; do not start a game")
    args = parser.parse_args(argv)
    args.advance_ticks = max(1, min(args.advance_ticks, 50))
    if args.probe_capabilities:
        print(json.dumps(asyncio.run(probe_server_capabilities(args.server_url)), sort_keys=True))
        return 0

    output_dir = args.output_dir or Path("runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    started_server = _ensure_server(args.server_url, args.difficulty, args.start_server)
    del started_server  # The official server is intentionally left running for reuse.
    _configure_local_server(args.server_url, args.map, args.difficulty)
    vlm = None if args.no_vlm else RCDVLM(_load_credentials(), args.model, args.base_url, args.request_timeout)
    return asyncio.run(run_episode(args, DecisionEngine(vlm, output_dir, args.personality, args.economy_control)))


if __name__ == "__main__":
    raise SystemExit(main())
