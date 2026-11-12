import asyncio
import unittest
from types import SimpleNamespace

from team_client import TeamActionRejected, TeamCoordinator, normalize_observation
from run_team import run_team


class Msg(SimpleNamespace):
    def __init__(self, **kwargs): super().__init__(**kwargs)


class Proto:
    NO_OP, MOVE, ATTACK, STOP, BUILD, TRAIN, DEPLOY, PLACE_BUILDING = range(8)
    StateRequest = TeamAdvanceRequest = TeamPlayerAction = AgentAction = Command = Msg


def observation(player, tick=1, *, done=False, units=None, buildings=None, production=None, allies=None):
    return Msg(controlled_player_id=player, tick=tick, done=done, result="win" if done else "",
               player_roster=[Msg(player_id=player, relationship="self"), Msg(player_id="ally", relationship="ally")],
               units=units or [], buildings=buildings or [], allied_units=allies or [], allied_buildings=[],
               visible_enemies=[], visible_enemy_buildings=[], production=production or [], available_production=["powr"],
               economy=Msg(cash=1000), military=Msg(), map_info=Msg(width=64, height=64))


class Stub:
    def __init__(self): self.calls = []; self.timeouts = []; self.tick = 1; self.obs = {"p1": observation("p1"), "p2": observation("p2")}
    def TeamState(self, request, timeout=None):
        self.timeouts.append(timeout)
        return Msg(observations=[Msg(player_id=p, observation=o) for p, o in self.obs.items()])
    def TeamAdvance(self, request, timeout=None):
        self.timeouts.append(timeout)
        self.calls.append(request); self.tick += request.ticks
        self.obs = {p: observation(p, self.tick) for p in self.obs}
        return Msg(tick=self.tick, observations=[Msg(player_id=p, observation=o) for p, o in self.obs.items()])


class TeamClientTests(unittest.TestCase):
    def test_normalization_preserves_owner_and_team_summary(self):
        own = Msg(actor_id=1, type="e1", owner="p1", cell_x=2, cell_y=3, can_attack=True)
        ally = Msg(actor_id=9, type="1tnk", owner="p2", cell_x=4, cell_y=5, can_attack=True)
        state = normalize_observation(observation("p1", units=[own], allies=[ally]))
        self.assertEqual(state["units_summary"][0]["id"], 1)
        self.assertEqual(state["allies_summary"][0]["owner"], "p2")
        self.assertEqual(state["controlled_player_id"], "p1")

    def test_commands_batch_once_and_no_cross_owner_routing(self):
        stub = Stub(); coordinator = TeamCoordinator(stub=stub, pb2=Proto, expected_players=["p1", "p2"])
        coordinator.team_state()
        a, b = coordinator.slots["p1"], coordinator.slots["p2"]
        self.assertTrue(asyncio.run(a.call_tool("move_units", unit_ids="11", target_x=1, target_y=2))["error"])
        asyncio.run(a.call_tool("build_unit", unit_type="e1"))
        asyncio.run(b.call_tool("build_unit", unit_type="e2"))
        states = coordinator.advance(12)
        self.assertEqual(len(stub.calls), 1)
        self.assertEqual(stub.calls[0].ticks, 12)
        self.assertEqual([x.player_id for x in stub.calls[0].player_actions], ["p1", "p2"])
        self.assertEqual([len(x.action.commands) for x in stub.calls[0].player_actions], [1, 1])
        self.assertEqual(states["p1"]["tick"], states["p2"]["tick"])

    def test_completed_structure_queues_and_retries_placement_until_observed(self):
        stub = Stub(); coordinator = TeamCoordinator(stub=stub, pb2=Proto, expected_players=["p1", "p2"])
        coordinator.team_state()
        asyncio.run(coordinator.slots["p1"].call_tool("build_and_place", building_type="powr", cell_x=7, cell_y=8))
        coordinator.states["p1"]["production"] = [{"item": "powr", "progress": 0.98}]
        coordinator._queue_completed_placements()
        self.assertEqual(len(coordinator.pending["p1"]), 1)  # BUILD command only
        coordinator.states["p1"]["production"] = [{"item": "powr", "progress": 0.99}]
        coordinator._queue_completed_placements()
        placement = coordinator.pending["p1"][-1]
        self.assertEqual(placement.action, Proto.PLACE_BUILDING)
        self.assertEqual((placement.target_x, placement.target_y), (7, 8))
        coordinator.pending["p1"] = []  # simulated successful response, placement still pending in engine
        for request in coordinator.placements["p1"]: request["sent"] = False
        coordinator._queue_completed_placements()
        self.assertEqual(len(coordinator.pending["p1"]), 1)
        self.assertEqual(coordinator.pending["p2"], [])

    def test_existing_same_type_building_does_not_suppress_requested_placement(self):
        existing = Msg(actor_id=5, type="powr", owner="p1", cell_x=1, cell_y=1)
        stub = Stub(); stub.obs["p1"] = observation("p1", buildings=[existing])
        coordinator = TeamCoordinator(stub=stub, pb2=Proto, expected_players=["p1", "p2"])
        coordinator.team_state()
        asyncio.run(coordinator.slots["p1"].call_tool("build_and_place", building_type="powr"))
        coordinator.states["p1"]["production"] = [{"item": "powr", "progress": 1.0}]
        coordinator._queue_completed_placements()
        self.assertEqual(coordinator.pending["p1"][-1].action, Proto.PLACE_BUILDING)

    def test_terminal_observation_normalizes(self):
        self.assertTrue(normalize_observation(observation("p1", done=True))["done"])

    def test_missing_team_rpc_fails_closed(self):
        coordinator = TeamCoordinator(stub=object(), pb2=Proto)
        with self.assertRaisesRegex(RuntimeError, "TeamState"):
            coordinator.team_state()

    def test_rpc_calls_are_bounded_and_failed_advance_preserves_pending_commands(self):
        class FailingStub(Stub):
            def TeamAdvance(self, request, timeout=None):
                self.timeouts.append(timeout)
                raise TimeoutError("timed out")
        stub = FailingStub()
        coordinator = TeamCoordinator(stub=stub, pb2=Proto, expected_players=["p1", "p2"], rpc_timeout_s=2.5)
        coordinator.team_state()
        asyncio.run(coordinator.slots["p1"].call_tool("build_unit", unit_type="e1"))
        with self.assertRaisesRegex(RuntimeError, "TeamAdvance RPC failed"):
            coordinator.advance(10)
        self.assertEqual(stub.timeouts, [2.5, 2.5])
        self.assertEqual(len(coordinator.pending["p1"]), 1)

    def test_atomic_rejection_can_refresh_without_replaying_stale_commands(self):
        import grpc
        class Rejected(grpc.RpcError):
            def code(self): return grpc.StatusCode.INVALID_ARGUMENT
        class RejectingStub(Stub):
            def TeamAdvance(self, request, timeout=None):
                raise Rejected("stale target")
        coordinator = TeamCoordinator(stub=RejectingStub(), pb2=Proto, expected_players=["p1", "p2"])
        coordinator.team_state()
        asyncio.run(coordinator.slots["p1"].call_tool("build_and_place", building_type="powr"))
        with self.assertRaises(TeamActionRejected):
            coordinator.advance(10)
        coordinator.recover_rejected_batch()
        self.assertEqual(coordinator.pending, {"p1": [], "p2": []})
        self.assertEqual(coordinator.placements["p1"], [])

    def test_legacy_and_team_generated_protocols_can_coexist(self):
        from openra_env.generated import rl_bridge_pb2 as legacy
        from server_extensions.generated import rl_bridge_pb2 as team
        self.assertTrue(hasattr(legacy, "GameObservation"))
        self.assertTrue(hasattr(team, "TeamAdvanceRequest"))

    def test_duplicate_slots_and_initial_tick_mismatch_fail_closed(self):
        class BadStub(Stub):
            def TeamState(self, request, timeout=None):
                return Msg(observations=[Msg(player_id="p1", observation=observation("p1", 1)),
                                         Msg(player_id="p1", observation=observation("p1", 1))])
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            TeamCoordinator(stub=BadStub(), pb2=Proto, expected_players=["p1", "p2"]).team_state()
        class MismatchStub(Stub):
            def TeamState(self, request, timeout=None):
                return Msg(observations=[Msg(player_id="p1", observation=observation("p1", 1)),
                                         Msg(player_id="p2", observation=observation("p2", 2))])
        with self.assertRaisesRegex(RuntimeError, "inconsistent player ticks"):
            TeamCoordinator(stub=MismatchStub(), pb2=Proto, expected_players=["p1", "p2"]).team_state()

    def test_terminal_team_state_finishes_without_advancing(self):
        class TerminalStub(Stub):
            def TeamState(self, request, timeout=None):
                return Msg(observations=[Msg(player_id=p, observation=observation(p, done=True)) for p in ("p1", "p2")])
        class Engine:
            def __init__(self): self.records = []
            def log(self, record): self.records.append(record)
        coordinator = TeamCoordinator(stub=TerminalStub(), pb2=Proto, expected_players=["p1", "p2"])
        engines = {p: Engine() for p in ("p1", "p2")}
        args = SimpleNamespace(economy_control="bootstrap", bootstrap=True, max_seconds=0, max_turns=0, advance_ticks=1)
        self.assertEqual(asyncio.run(run_team(args, coordinator, engines)), 0)
        self.assertEqual(coordinator.stub.calls, [])
        self.assertTrue(any(row["event"] == "game_finished" for row in engines["p1"].records))


if __name__ == "__main__": unittest.main()
