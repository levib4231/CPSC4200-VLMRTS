"""Run two DecisionEngine policies through the native coordinated team gRPC API."""
from __future__ import annotations

import argparse
import asyncio
import copy
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from agent import (Action, BootstrapController, DecisionEngine, PERSONALITIES, RCDVLM, _defer_vlm_economy,
                   _issue_action, _load_credentials, sanitize_action)
from team_client import TeamActionRejected, TeamCoordinator


async def run_team_realtime(args: argparse.Namespace, coordinator: TeamCoordinator,
                           engines: dict[str, DecisionEngine]) -> int:
    """Advance the world on a wall-clock cadence while VLM calls run in background threads."""
    states = coordinator.team_state()
    feedback: dict[str, dict[str, Any] | None] = {player: None for player in states}
    bootstraps = {player: BootstrapController() if args.economy_control in {"bootstrap", "hybrid"} and args.bootstrap else None for player in states}
    pending: dict[str, asyncio.Task[Action]] = {}
    ready: dict[str, tuple[Action, dict[str, Any]]] = {}
    started, turns = time.monotonic(), 0

    for player, state in states.items():
        engines[player].log({"event": "game_started", "player_id": player, "state": state,
                              "transport": "native_grpc", "realtime": True})

    while not all(bool(state.get("done")) for state in states.values()):
        if args.max_seconds > 0 and time.monotonic() - started >= args.max_seconds:
            break
        if args.max_turns > 0 and turns >= args.max_turns:
            break

        for player, task in list(pending.items()):
            if task.done():
                pending.pop(player)
                try:
                    ready[player] = (task.result(), dict(engines[player].last_feedback))
                except Exception as exc:
                    ready[player] = (Action("HOLD", [], reason="VLM task failed"),
                                     {"validation": "not_ready", "model_error": str(exc)})

        for player, state in states.items():
            if state.get("done") or player in pending or player in ready:
                continue
            snapshot = copy.deepcopy(state)
            previous_feedback = copy.deepcopy(feedback[player]) if feedback[player] else None
            pending[player] = asyncio.create_task(
                asyncio.to_thread(engines[player].decide, snapshot, previous_feedback)
            )

        for player, state in states.items():
            if state.get("done"):
                continue
            adapter = coordinator.slots[player]
            bootstrap_result = None
            bootstrap = bootstraps[player]
            if bootstrap is not None:
                bootstrap_result = await bootstrap.step(adapter, state)
                if bootstrap_result:
                    engines[player].log({"event": "bootstrap", "player_id": player,
                                         "tick": state["tick"], **bootstrap_result, "realtime": True})

            deployment_mutation = bootstrap_result and str(bootstrap_result.get("phase", "")).startswith("deploy")
            decision_ready = player in ready and not deployment_mutation
            if deployment_mutation:
                ready.pop(player, None)
                action = Action("HOLD", [], reason="bootstrap deployment")
                decision_feedback = {
                    "requested_action": action.__dict__,
                    "submitted_action": action.__dict__,
                    "validation": "deferred",
                }
                action_result = {"note": "all other orders skipped during deployment"}
            else:
                if decision_ready:
                    action, decision_feedback = ready.pop(player)
                    current_action = sanitize_action(action, state)
                    if current_action != action:
                        decision_feedback = dict(decision_feedback)
                        decision_feedback.update({
                            "submitted_action": current_action.__dict__,
                            "validation": "rejected",
                            "validation_reason": current_action.reason,
                        })
                    action = current_action
                    deferred = _defer_vlm_economy(action, bootstrap_result)
                    if deferred != action:
                        decision_feedback = dict(decision_feedback)
                        decision_feedback.update({
                            "submitted_action": deferred.__dict__,
                            "validation": "deferred",
                            "validation_reason": deferred.reason,
                        })
                        action = deferred
                else:
                    action = Action("HOLD", [], reason="VLM decision pending")
                    decision_feedback = {
                        "requested_action": action.__dict__,
                        "submitted_action": action.__dict__,
                        "validation": "not_ready",
                    }
                action_result = await _issue_action(adapter, action)
            feedback[player] = {**decision_feedback, "queue_result": action_result}
            if bootstrap_result:
                feedback[player]["bootstrap_result"] = bootstrap_result
            engines[player].log({"event": "team_action_queued", "player_id": player,
                                 "turn": turns, "tick": state["tick"], "action": action.__dict__,
                                 "action_result": action_result,
                                 "decision_pending": not decision_ready,
                                 "bootstrap_active": bool(bootstrap_result)})

        try:
            states = coordinator.advance(args.advance_ticks)
        except TeamActionRejected as exc:
            for player, engine in engines.items():
                feedback[player] = {**(feedback[player] or {}),
                                    "advance_result": {"status": "rejected", "error": str(exc)}}
                engine.log({"event": "team_action_rejected", "player_id": player,
                            "turn": turns, "error": str(exc), "realtime": True})
            states = coordinator.recover_rejected_batch()
        else:
            for player, engine in engines.items():
                feedback[player] = {**(feedback[player] or {}),
                                    "advance_result": {"status": "accepted", "tick": states[player]["tick"]}}
                engine.log({"event": "team_advance", "player_id": player, "turn": turns,
                            "tick": states[player]["tick"], "done": states[player]["done"],
                            "result": states[player]["result"], "realtime": True})
        turns += 1
        realtime_delay = getattr(args, "realtime_delay", args.advance_ticks / 25.0)
        if realtime_delay > 0:
            await asyncio.sleep(realtime_delay)

    for task in pending.values():
        task.cancel()
    for player, engine in engines.items():
        if states[player].get("done"):
            engine.log({"event": "game_finished", "player_id": player,
                        "result": states[player]["result"], "tick": states[player]["tick"],
                        "turns": turns, "realtime": True})
    print({"result": {p: s["result"] for p, s in states.items()},
           "tick": next(iter(states.values()))["tick"], "turns": turns, "realtime": True})
    return 0


async def run_team(args: argparse.Namespace, coordinator: TeamCoordinator, engines: dict[str, DecisionEngine]) -> int:
    if getattr(args, "realtime", False):
        return await run_team_realtime(args, coordinator, engines)
    states = coordinator.team_state()
    if set(states) != set(engines):
        raise RuntimeError(f"expected team slots {sorted(engines)}, got {sorted(states)}")
    bootstraps = {player: BootstrapController() if args.economy_control in {"bootstrap", "hybrid"} and args.bootstrap else None for player in states}
    feedback: dict[str, dict[str, Any] | None] = {player: None for player in states}
    for player, state in states.items():
        engines[player].log({"event": "game_started", "player_id": player, "state": state, "transport": "native_grpc"})
    started, turns = time.monotonic(), 0
    while not all(bool(state.get("done")) for state in states.values()):
        if args.max_seconds > 0 and time.monotonic() - started >= args.max_seconds:
            for player, engine in engines.items(): engine.log({"event": "time_limit", "player_id": player, "turns": turns, "tick": states[player]["tick"]})
            return 0
        if args.max_turns > 0 and turns >= args.max_turns:
            for player, engine in engines.items(): engine.log({"event": "turn_limit", "player_id": player, "turns": turns, "tick": states[player]["tick"]})
            return 0
        for player, state in states.items():
            if state.get("done"): continue
            adapter = coordinator.slots[player]
            bootstrap_result = None
            bootstrap = bootstraps[player]
            if bootstrap is not None:
                bootstrap_result = await bootstrap.step(adapter, state)
                if bootstrap_result:
                    engines[player].log({"event": "bootstrap", "player_id": player, "tick": state["tick"], **bootstrap_result})
            if bootstrap_result and str(bootstrap_result.get("phase", "")).startswith("deploy"):
                from agent import Action
                action = Action("HOLD", [], reason="bootstrap deployment")
                action_result = {"note": "all other orders skipped during deployment"}
            else:
                action = await asyncio.to_thread(engines[player].decide, state, feedback[player])
                deferred = _defer_vlm_economy(action, bootstrap_result)
                if deferred != action:
                    engines[player].log({"event": "action_deferred", "player_id": player, "decision_source": "deferred",
                                         "reason": deferred.reason, "proposed_action": action.__dict__})
                    action, action_result = deferred, {"note": "VLM economy order deferred during bootstrap mutation"}
                else:
                    action_result = await _issue_action(adapter, action)
            feedback[player] = {**engines[player].last_feedback, "queue_result": action_result}
            engines[player].log({"event": "team_action_queued", "player_id": player, "turn": turns,
                                 "tick": state["tick"], "action": action.__dict__, "action_result": action_result})
        try:
            states = coordinator.advance(args.advance_ticks)
        except TeamActionRejected as exc:
            for player, engine in engines.items():
                feedback[player] = {**(feedback[player] or {}),
                                    "advance_result": {"status": "rejected", "error": str(exc)}}
                engine.log({"event": "team_action_rejected", "player_id": player,
                            "turn": turns, "error": str(exc)})
            states = coordinator.recover_rejected_batch()
            turns += 1
            continue
        for player, engine in engines.items():
            feedback[player] = {**(feedback[player] or {}),
                                "advance_result": {"status": "accepted", "tick": states[player]["tick"]}}
            engine.log({"event": "team_advance", "player_id": player, "turn": turns,
                        "tick": states[player]["tick"], "done": states[player]["done"], "result": states[player]["result"]})
        turns += 1
    for player, state in states.items():
        engines[player].log({"event": "game_finished", "player_id": player, "result": state["result"], "tick": state["tick"]})
    print({"result": {p: s["result"] for p, s in states.items()}, "tick": next(iter(states.values()))["tick"], "turns": turns})
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Native gRPC coordinated OpenRA team agent")
    parser.add_argument("--grpc-address", default=os.getenv("OPENRA_GRPC_ADDRESS", "127.0.0.1:9999"))
    parser.add_argument("--session-id", default=os.getenv("OPENRA_SESSION_ID", ""))
    parser.add_argument("--players", default=os.getenv("OPENRA_TEAM_PLAYERS", "Multi1,Multi2"), help="comma-separated exact player slot IDs")
    parser.add_argument("--max-seconds", type=int, default=60)
    parser.add_argument("--max-turns", type=int, default=0)
    parser.add_argument("--advance-ticks", type=int, default=25)
    parser.add_argument("--realtime", action="store_true", help="advance on a wall-clock cadence without waiting for VLM calls")
    parser.add_argument("--realtime-delay", type=float, default=None, help="seconds between realtime batches; default is advance_ticks/25")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-vlm", action="store_true")
    parser.add_argument("--model", default=os.getenv("RCD_VLM_MODEL") or os.getenv("RCD_LLM_MODEL", "qwen3-omni-30b-a3b"))
    parser.add_argument("--base-url", default=os.getenv("RCD_LLM_BASE_URL", "https://llm.rcd.clemson.edu/v1"))
    parser.add_argument("--request-timeout", type=float, default=45.0)
    parser.add_argument("--personality", choices=sorted(PERSONALITIES), action="append")
    parser.add_argument("--economy-control", choices=("bootstrap", "hybrid", "vlm"), default="bootstrap")
    parser.add_argument("--no-bootstrap", dest="bootstrap", action="store_false")
    parser.set_defaults(bootstrap=True)
    args = parser.parse_args(argv)
    players = [value.strip() for value in args.players.split(",") if value.strip()]
    if len(players) < 2 or len(set(players)) != len(players): parser.error("--players must contain at least two unique slots")
    args.advance_ticks = max(1, min(args.advance_ticks, 50))
    if args.realtime_delay is None:
        args.realtime_delay = args.advance_ticks / 25.0
    personalities = args.personality or ["helpful", "selfish"]
    if len(personalities) == 1: personalities *= len(players)
    if len(personalities) != len(players): parser.error("provide one personality per player")
    output = args.output_dir or Path("runs") / ("team_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    vlm = None if args.no_vlm else RCDVLM(_load_credentials(), args.model, args.base_url, args.request_timeout)
    engines = {player: DecisionEngine(vlm, output / player, personalities[i], args.economy_control) for i, player in enumerate(players)}
    coordinator = TeamCoordinator(args.grpc_address, args.session_id, expected_players=players)
    try: return asyncio.run(run_team(args, coordinator, engines))
    finally: coordinator.close()


if __name__ == "__main__": raise SystemExit(main())
