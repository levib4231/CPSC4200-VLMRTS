"""Opt-in native-tool integration test; creates a new single-player match."""
import asyncio
import os
import unittest

from agent import Action, _issue_action, _prepare_game, sanitize_action


@unittest.skipUnless(os.getenv("OPENRA_LIVE_TEST") == "1", "requires a running OpenRA-RL server")
class LiveEconomyTests(unittest.TestCase):
    def test_deploy_build_and_train(self):
        asyncio.run(self._exercise())

    async def _exercise(self):
        from openra_env.mcp_ws_client import OpenRAMCPClient

        async def wait_for(game, predicate):
            for _ in range(60):
                state = await game.call_tool("get_game_state")
                if predicate(state):
                    return state
                self.assertFalse(state.get("done"), "match ended before production finished")
                await game.call_tool("advance", ticks=50)
            self.fail("production did not become available within 3000 ticks")

        async def execute(game, action, state):
            validated = sanitize_action(action, state)
            self.assertEqual(validated.action, action.action, validated.reason)
            result = await _issue_action(game, validated)
            self.assertFalse(result.get("error"), result)

        url = os.getenv("OPENRA_URL", "http://127.0.0.1:8000")
        async with OpenRAMCPClient(url) as game:
            await game.reset()
            state = await _prepare_game(game)
            mcv = next(unit for unit in state["units_summary"] if unit["type"] == "mcv")
            await execute(game, Action("DEPLOY", [mcv["id"]]), state)
            state = await wait_for(game, lambda s: "powr" in s.get("available_production", []))
            await execute(game, Action("BUILD_STRUCTURE", [], structure_type="powr"), state)
            state = await wait_for(game, lambda s: "powr" in s.get("building_types", []))
            barracks = next(name for name in ("tent", "barr") if name in state["available_production"])
            await execute(game, Action("BUILD_STRUCTURE", [], structure_type=barracks), state)
            state = await wait_for(game, lambda s: "e1" in s.get("available_production", []))
            await execute(game, Action("BUILD_UNIT", [], unit_type="e1"), state)
            await wait_for(game, lambda s: any(unit["type"] == "e1" for unit in s.get("units_summary", [])))
