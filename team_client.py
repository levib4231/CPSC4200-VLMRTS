"""Native gRPC client and per-player adapters for coordinated OpenRA team games."""
from __future__ import annotations

import importlib
from typing import Any, Mapping



class TeamActionRejected(RuntimeError):
    """The server rejected an entire batch before applying any of its orders."""


def _value(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)


def _unit(raw: Any, owner: str | None = None) -> dict[str, Any]:
    actor_id = int(_value(raw, "actor_id", 0))
    result = {
        "id": actor_id, "actor_id": actor_id, "type": str(_value(raw, "type", "")),
        "owner": str(_value(raw, "owner", owner or "")),
        "x": int(_value(raw, "cell_x", 0)), "y": int(_value(raw, "cell_y", 0)),
        "cell_x": int(_value(raw, "cell_x", 0)), "cell_y": int(_value(raw, "cell_y", 0)),
        "hp_percent": float(_value(raw, "hp_percent", 0.0)),
        "can_attack": bool(_value(raw, "can_attack", False)),
        "is_idle": bool(_value(raw, "is_idle", False)),
    }
    for key in ("ammo", "attack_range", "is_building"):
        result[key] = _value(raw, key, False if key == "is_building" else 0)
    return result


def normalize_observation(obs: Any) -> dict[str, Any]:
    """Convert one protobuf observation to the state shape consumed by agent.py."""
    controlled = str(_value(obs, "controlled_player_id", ""))
    roster = list(_value(obs, "player_roster", []))
    relations = {str(_value(entry, "player_id", "")): str(_value(entry, "relationship", "")) for entry in roster}
    own_units = [_unit(raw, controlled) for raw in _value(obs, "units", [])]
    own_buildings = [_unit(raw, controlled) for raw in _value(obs, "buildings", [])]
    allies = [_unit(raw) for raw in _value(obs, "allied_units", [])]
    ally_buildings = [_unit(raw) for raw in _value(obs, "allied_buildings", [])]
    enemies = [_unit(raw) for raw in _value(obs, "visible_enemies", [])]
    enemy_buildings = [_unit(raw) for raw in _value(obs, "visible_enemy_buildings", [])]
    economy = _value(obs, "economy", None)
    map_info = _value(obs, "map_info", None)
    production = []
    production_items = []
    for item in _value(obs, "production", []):
        entry = {"item": str(_value(item, "item", "")), "queue_type": str(_value(item, "queue_type", "")),
                 "progress": float(_value(item, "progress", 0)), "remaining_ticks": int(_value(item, "remaining_ticks", 0)),
                 "remaining_cost": int(_value(item, "remaining_cost", 0)), "paused": bool(_value(item, "paused", False))}
        production.append(entry)
        production_items.append(f"{entry['item']}@{entry['progress'] * 100:.0f}%(~{entry['remaining_ticks']} ticks)")
    result = {
        "tick": int(_value(obs, "tick", 0)), "done": bool(_value(obs, "done", False)),
        "result": str(_value(obs, "result", "")), "controlled_player_id": controlled,
        "player": controlled, "player_roster": [{"player_id": p, "relationship": r} for p, r in relations.items()],
        "units": own_units, "units_summary": own_units,
        "buildings": own_buildings, "buildings_summary": own_buildings,
        "allied_units": allies, "allied_buildings": ally_buildings,
        "allies_summary": allies + ally_buildings,
        "visible_enemies": enemies, "enemy_summary": enemies,
        "visible_enemy_buildings": enemy_buildings, "enemy_buildings_summary": enemy_buildings,
        "available_production": list(_value(obs, "available_production", [])),
        "production": production, "production_items": production_items,
        "building_types": [u["type"] for u in own_buildings],
        "economy": {key: _value(economy, key, 0) for key in ("cash", "ore", "power_provided", "power_drained", "resource_capacity", "harvester_count")},
        "military": {key: _value(_value(obs, "military", None), key, 0) for key in ("units_killed", "units_lost", "army_value", "active_unit_count")},
        "map": {"width": int(_value(map_info, "width", 0)), "height": int(_value(map_info, "height", 0)), "map_name": str(_value(map_info, "map_name", ""))},
    }
    result["allies_summary"] = allies + ally_buildings
    return result


class TeamCoordinator:
    """Owns one shared world advance; adapters only append commands to its batch."""
    def __init__(self, address: str | None = None, session_id: str = "", *, stub: Any = None, pb2: Any = None,
                 expected_players: list[str] | None = None, channel: Any = None,
                 rpc_timeout_s: float = 10.0) -> None:
        if pb2 is None:
            pb2 = importlib.import_module("server_extensions.generated.rl_bridge_pb2")
        self.pb2 = pb2
        self.session_id = session_id
        self.expected_players = list(expected_players or [])
        self.rpc_timeout_s = max(0.1, float(rpc_timeout_s))
        self._channel = channel
        if stub is None:
            if not address:
                raise ValueError("gRPC address is required")
            grpc = importlib.import_module("grpc")
            self._channel = channel or grpc.insecure_channel(address)
            stub_cls = importlib.import_module("server_extensions.generated.rl_bridge_pb2_grpc").RLBridgeStub
            stub = stub_cls(self._channel)
        self.stub = stub
        self.states: dict[str, dict[str, Any]] = {}
        self.pending: dict[str, list[Any]] = {}
        self.placements: dict[str, list[dict[str, Any]]] = {}
        self.slots: dict[str, TeamPlayerAdapter] = {}

    def _request(self) -> Any:
        return self.pb2.StateRequest(session_id=self.session_id)

    def _verify(self, response: Any) -> dict[str, dict[str, Any]]:
        observations: dict[str, dict[str, Any]] = {}
        for row in _value(response, "observations", []):
            player = str(_value(row, "player_id", ""))
            if not player or player in observations:
                raise RuntimeError(f"Team RPC contains empty or duplicate player slot: {player!r}")
            observations[player] = normalize_observation(_value(row, "observation"))
        ids = set(observations)
        if not ids or (self.expected_players and ids != set(self.expected_players)):
            raise RuntimeError(f"Team RPC player slots mismatch: expected {self.expected_players}, received {sorted(ids)}")
        if any(state["controlled_player_id"] != player for player, state in observations.items()):
            raise RuntimeError("Team RPC observation identity does not match its player slot")
        ticks = {state["tick"] for state in observations.values()}
        if len(ticks) != 1:
            raise RuntimeError(f"Team RPC returned inconsistent player ticks: {sorted(ticks)}")
        return observations

    def team_state(self) -> dict[str, dict[str, Any]]:
        method = getattr(self.stub, "TeamState", None)
        if method is None:
            raise RuntimeError("server does not advertise required TeamState RPC")
        try:
            response = method(self._request(), timeout=self.rpc_timeout_s)
        except Exception as exc:
            raise RuntimeError(f"TeamState RPC failed or unsupported: {exc}") from exc
        states = self._verify(response)
        self.states.update(states)
        for player in states:
            self.pending.setdefault(player, [])
            self.placements.setdefault(player, [])
            self.slots.setdefault(player, TeamPlayerAdapter(self, player))
        return states

    def advance(self, ticks: int = 25) -> dict[str, dict[str, Any]]:
        method = getattr(self.stub, "TeamAdvance", None)
        if method is None:
            raise RuntimeError("server does not advertise required TeamAdvance RPC")
        actions = [self.pb2.TeamPlayerAction(player_id=player, action=self.pb2.AgentAction(commands=self.pending[player]))
                   for player in self.states]
        request = self.pb2.TeamAdvanceRequest(ticks=ticks, player_actions=actions, session_id=self.session_id)
        try:
            response = method(request, timeout=self.rpc_timeout_s)
        except Exception as exc:
            # Actors can die or leave visibility on the dispatch tick after the
            # frozen observation. An atomic rejection is recoverable; timeouts
            # are ambiguous and must not be automatically replayed.
            code = getattr(exc, "code", lambda: None)()
            if code == importlib.import_module("grpc").StatusCode.INVALID_ARGUMENT:
                raise TeamActionRejected(str(exc)) from exc
            raise RuntimeError(f"TeamAdvance RPC failed or unsupported: {exc}") from exc
        states = self._verify(response)
        tick = int(_value(response, "tick", 0))
        if any(state["tick"] != tick for state in states.values()):
            raise RuntimeError("TeamAdvance returned inconsistent player ticks")
        self.pending = {player: [] for player in self.states}
        for requests in self.placements.values():
            for placement in requests:
                placement["sent"] = False
        self.states.update(states)
        self._queue_completed_placements()
        return states

    def recover_rejected_batch(self) -> dict[str, dict[str, Any]]:
        self.pending = {player: [] for player in self.states}
        states = self.team_state()
        for player, requests in self.placements.items():
            active = {p["item"] for p in states[player]["production"]}
            self.placements[player] = [r for r in requests if r["item"] in active]
            for request in self.placements[player]:
                request["sent"] = False
        self._queue_completed_placements()
        return states

    def _enum(self, symbolic: str) -> int:
        # Generated protobuf modules expose constants at module level; enum values are canonical.
        return int(getattr(self.pb2, symbolic))

    def command(self, player: str, action: str, **fields: Any) -> dict[str, Any]:
        if player not in self.states:
            raise RuntimeError(f"unknown team player slot {player}")
        state = self.states[player]
        owned = {int(unit["id"]) for unit in state["units_summary"] + state["buildings_summary"]}
        actor = fields.get("actor_id")
        if actor is not None and int(actor) not in owned:
            return {"error": f"actor {actor} is not owned by {player}"}
        target = fields.get("target_actor_id")
        if action == "ATTACK" and target is not None:
            visible = {int(u["id"]) for key in ("visible_enemies", "visible_enemy_buildings") for u in state[key]}
            if int(target) not in visible:
                return {"error": f"target {target} is not a visible enemy"}
        cmd = self.pb2.Command(action=self._enum(action), **fields)
        self.pending[player].append(cmd)
        return {"queued": True, "player_id": player, "action": action}

    def _queue_completed_placements(self) -> None:
        for player, requests in self.placements.items():
            state = self.states[player]
            production = state["production"]
            building_counts: dict[str, int] = {}
            for building in state["buildings_summary"]:
                kind = building["type"]
                building_counts[kind] = building_counts.get(kind, 0) + 1
            retained = []
            for request in requests:
                kind = request["item"]
                if building_counts.get(kind, 0) > request["baseline_buildings"]:
                    continue
                completed = any(p["item"] == kind and p["progress"] >= 0.99 for p in production)
                if request["completed"] and not any(p["item"] == kind for p in production):
                    continue
                request["completed"] = request["completed"] or completed
                ready_to_place = request["completed"] and any(p["item"] == kind for p in production)
                if ready_to_place and not request["sent"]:
                    target = request["target"] or (0, 0)
                    self.command(player, "PLACE_BUILDING", item_type=kind, target_x=target[0], target_y=target[1])
                    request["sent"] = True
                retained.append(request)
            self.placements[player] = retained

    def close(self) -> None:
        if self._channel is not None:
            self._channel.close()


class TeamPlayerAdapter:
    def __init__(self, coordinator: TeamCoordinator, player_id: str) -> None:
        self.coordinator, self.player_id = coordinator, player_id

    async def call_tool(self, name: str, **kwargs: Any) -> Any:
        state = self.coordinator.states[self.player_id]
        if name == "get_game_state":
            return state
        if name in {"build_unit", "build_and_place"}:
            kind = kwargs.get("unit_type", kwargs.get("building_type", ""))
            is_building = name == "build_and_place"
            count = 1 if is_building else max(1, min(int(kwargs.get("count", 1)), 5))
            result = {"queued": True, "player_id": self.player_id, "item_type": kind}
            for _ in range(count):
                result = self.coordinator.command(self.player_id, "BUILD" if is_building else "TRAIN", item_type=kind, queued=True)
                if result.get("error"):
                    return result
            if is_building:
                x, y = kwargs.get("cell_x"), kwargs.get("cell_y")
                baseline = sum(1 for building in state["buildings_summary"] if building["type"] == kind)
                baseline += sum(1 for placement in self.coordinator.placements[self.player_id] if placement["item"] == kind)
                self.coordinator.placements[self.player_id].append({
                    "item": kind, "target": (int(x), int(y)) if x is not None and y is not None else None,
                    "baseline_buildings": baseline, "completed": False, "sent": False,
                })
            return result
        if name == "deploy_unit":
            return self.coordinator.command(self.player_id, "DEPLOY", actor_id=int(kwargs["unit_id"]))
        if name in {"attack_target", "move_units", "stop_units"}:
            raw_ids = kwargs.get("unit_ids", "")
            ids = [int(x) for x in raw_ids.split(",") if x.strip()] if isinstance(raw_ids, str) else [int(x) for x in raw_ids]
            action = {"attack_target": "ATTACK", "move_units": "MOVE", "stop_units": "STOP"}[name]
            result = {"queued": True}
            for actor in ids:
                fields: dict[str, Any] = {"actor_id": actor}
                if action == "ATTACK": fields["target_actor_id"] = int(kwargs["target_actor_id"])
                if action == "MOVE": fields.update(target_x=int(kwargs["target_x"]), target_y=int(kwargs["target_y"]))
                result = self.coordinator.command(self.player_id, action, **fields)
                if result.get("error"): return result
            return result
        raise ValueError(f"unsupported native team tool {name!r}")
