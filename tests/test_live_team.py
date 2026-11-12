"""Opt-in tests against the custom native team server; advances its current match."""
import os
import time
import unittest


@unittest.skipUnless(os.getenv("OPENRA_TEAM_LIVE_TEST") == "1", "requires the custom team server")
class LiveTeamTests(unittest.TestCase):
    def setUp(self):
        import grpc
        from team_client import TeamCoordinator
        self.grpc = grpc
        self.client = TeamCoordinator(os.getenv("OPENRA_GRPC_ADDRESS", "127.0.0.1:9999"),
                                      expected_players=["Multi1", "Multi2"])
        self.states = self.client.team_state()
        self.pb = self.client.pb2

    def tearDown(self):
        self.client.close()

    def test_relationships_visibility_and_shared_step(self):
        for player, state in self.states.items():
            ally = "Multi2" if player == "Multi1" else "Multi1"
            roster = {p["player_id"]: p["relationship"] for p in state["player_roster"]}
            self.assertEqual(roster[player], "self")
            self.assertEqual(roster[ally], "ally")
            self.assertEqual(roster["Multi0"], "enemy")
            self.assertEqual(roster["Multi3"], "enemy")
            self.assertTrue(all(u["owner"] == player for u in state["units_summary"]))
            self.assertTrue(all(u["owner"] == ally for u in state["allies_summary"]))
            self.assertTrue(all(u["owner"] not in {player, ally} for u in state["enemy_summary"]))
        before = self.states["Multi1"]["tick"]
        time.sleep(0.2)  # Test that idle simulation is paused, not a readiness poll.
        again = self.client.team_state()
        self.assertEqual(again["Multi1"]["tick"], before)
        after = self.client.advance(10)
        self.assertEqual(after["Multi1"]["tick"], after["Multi2"]["tick"])
        self.assertGreater(after["Multi1"]["tick"], before)

    def test_server_rejects_cross_owner_commands_and_allied_attacks(self):
        owned = self.states["Multi1"]["units_summary"] + self.states["Multi1"]["buildings_summary"]
        allied = self.states["Multi2"]["units_summary"] + self.states["Multi2"]["buildings_summary"]
        self.assertTrue(owned and allied, "test requires living controlled players")
        for command in (
            self.pb.Command(action=self.pb.STOP, actor_id=allied[0]["id"]),
            self.pb.Command(action=self.pb.ATTACK, actor_id=owned[0]["id"], target_actor_id=allied[0]["id"]),
        ):
            request = self.pb.TeamAdvanceRequest(ticks=1, player_actions=[
                self.pb.TeamPlayerAction(player_id="Multi1", action=self.pb.AgentAction(commands=[command]))])
            with self.assertRaises(self.grpc.RpcError) as caught:
                self.client.stub.TeamAdvance(request, timeout=10)
            self.assertEqual(caught.exception.code(), self.grpc.StatusCode.INVALID_ARGUMENT)

    def test_server_rejects_unconfigured_enemy_slot(self):
        request = self.pb.TeamAdvanceRequest(ticks=1, player_actions=[
            self.pb.TeamPlayerAction(player_id="Multi0", action=self.pb.AgentAction())])
        with self.assertRaises(self.grpc.RpcError) as caught:
            self.client.stub.TeamAdvance(request, timeout=10)
        self.assertEqual(caught.exception.code(), self.grpc.StatusCode.PERMISSION_DENIED)
