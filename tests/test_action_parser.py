import json
import tempfile
import unittest
from pathlib import Path

from agent import Action, ActionParseError, DecisionEngine, _defer_vlm_economy, _issue_action, _suppress_order_during_deploy, fallback_action, parse_action, sanitize_action


class ActionTests(unittest.TestCase):
    def test_json_action(self) -> None:
        action = parse_action('{"action":"ATTACK","unit_ids":[101,102],"target_id":202}')
        self.assertEqual(action.action, "ATTACK")
        self.assertEqual(action.unit_ids, [101, 102])
        self.assertEqual(action.target_id, 202)

    def test_markdown_fence_and_coordinate(self) -> None:
        action = parse_action('```json\n{"action":"MOVE","target":{"x":4,"y":8}}\n```')
        self.assertEqual(action.target, (4.0, 8.0))

    def test_fallback_is_validated_against_openra_rl_ids(self) -> None:
        state = {
            "map": {"width": 64, "height": 64},
            "units_summary": [{"id": 101, "type": "e1", "can_attack": True}],
            "enemy_summary": [{"id": 202, "type": "1tnk"}],
        }
        action = sanitize_action(fallback_action(state), state)
        self.assertEqual(action.action, "ATTACK")
        self.assertEqual(action.unit_ids, [101])
        self.assertEqual(action.target_id, 202)

    def test_fallback_attacks_visible_enemy_building(self) -> None:
        state = {
            "map": {"width": 112, "height": 54},
            "units_summary": [{"id": 101, "type": "e1", "can_attack": True}],
            "enemy_buildings_summary": [{"id": 404, "type": "powr"}],
        }
        action = fallback_action(state)
        self.assertEqual(action.action, "ATTACK")
        self.assertEqual(action.target_id, 404)

    def test_retreat_targets_construction_yard(self) -> None:
        state = {
            "map": {"width": 64, "height": 64},
            "units_summary": [{"id": 101, "type": "e1", "can_attack": True}],
            "buildings_summary": [{"id": 303, "type": "fact", "cell_x": 7, "cell_y": 9}],
        }
        action = sanitize_action(parse_action('{"action":"RETREAT","unit_ids":[101]}'), state)
        self.assertEqual(action.target, (7.0, 9.0))

    def test_move_is_clamped_to_map(self) -> None:
        state = {
            "map": {"width": 64, "height": 64},
            "units_summary": [{"id": 101, "type": "e1", "can_attack": True}],
        }
        action = sanitize_action(parse_action('{"action":"MOVE","target":{"x":99,"y":-2}}'), state)
        self.assertEqual(action.target, (63.0, 0.0))

    def test_native_observation_shape_uses_economy_and_visible_enemy_buildings(self) -> None:
        state = {"units": [{"actor_id": 1, "type": "e1", "can_attack": True}],
                 "buildings": [{"actor_id": 10, "type": "fact", "cell_x": 2, "cell_y": 3}],
                 "visible_enemy_buildings": [{"actor_id": 20, "type": "powr", "owner": "Multi1"}],
                 "map_info": {"width": 32, "height": 32},
                 "economy": {"cash": 4000, "ore": 200},
                 "available_production": ["e1"]}
        self.assertEqual(fallback_action(state).target_id, 20)
        order = sanitize_action(parse_action('{"action":"BUILD_UNIT","unit_type":"e1","count":2}'), state)
        self.assertEqual(order.action, "BUILD_UNIT")

    def test_economy_action_parse_and_safety(self) -> None:
        action = parse_action('{"action":"BUILD_UNIT","unit_type":"e1","count":2}')
        state = {"available_production": ["e1"], "economy": {"cash": 1500, "ore": 300}, "units_summary": []}
        self.assertEqual(sanitize_action(action, state).action, "BUILD_UNIT")
        self.assertEqual(sanitize_action(action, {**state, "economy": {"cash": 100}}).action, "HOLD")
        with self.assertRaises(ActionParseError):
            parse_action('{"action":"BUILD_UNIT","unit_type":"e1","count":6}')

    def test_deploy_requires_owned_deployable_unit(self) -> None:
        action = parse_action('{"action":"DEPLOY","unit_ids":[9]}')
        self.assertEqual(sanitize_action(action, {"units_summary": [{"id": 9, "type": "e1"}]}).action, "HOLD")
        self.assertEqual(sanitize_action(action, {"units_summary": [{"id": 9, "type": "mcv"}]}).unit_ids, [9])

    def test_allied_actor_is_not_attack_target(self) -> None:
        state = {"team": "blue", "units_summary": [{"id": 1, "can_attack": True}],
                 "enemy_summary": [{"id": 2, "team": "blue"}], "map": {"width": 8, "height": 8}}
        self.assertEqual(fallback_action(state).action, "MOVE")

    def test_noncombat_explicit_orders_and_default_exclusion(self) -> None:
        state = {"map_info": {"width": 32, "height": 32}, "units": [
            {"actor_id": 1, "type": "harv", "can_attack": False},
            {"actor_id": 2, "type": "mcv", "can_attack": False},
            {"actor_id": 3, "type": "e1", "can_attack": True}],
            "buildings": [{"actor_id": 7, "type": "fact", "cell_x": 5, "cell_y": 6}]}
        default = sanitize_action(parse_action('{"action":"MOVE","target":[4,4]}'), state)
        self.assertEqual(default.unit_ids, [3])
        explicit = sanitize_action(parse_action('{"action":"MOVE","unit_ids":[1],"target":[4,4]}'), state)
        self.assertEqual(explicit.unit_ids, [1])
        invalid = sanitize_action(parse_action('{"action":"MOVE","unit_ids":[999],"target":[4,4]}'), state)
        self.assertEqual(invalid.unit_ids, [])
        self.assertEqual(invalid.action, "HOLD")
        hold = sanitize_action(parse_action('{"action":"HOLD","unit_ids":[2]}'), state)
        self.assertEqual(hold.unit_ids, [2])

    def test_economy_placement_validation_and_queue_duplication(self) -> None:
        structure = parse_action('{"action":"BUILD_STRUCTURE","structure_type":"powr","target":[5,5]}')
        state = {"available_production": ["powr"], "economy": {"cash": 5000}, "map_info": {"width": 10, "height": 10}}
        self.assertEqual(sanitize_action(structure, state).action, "BUILD_STRUCTURE")
        self.assertEqual(sanitize_action(structure, {**state, "production": [{"item": "powr"}]}).action, "HOLD")
        out = parse_action('{"action":"BUILD_STRUCTURE","structure_type":"powr","target":[50,5]}')
        self.assertEqual(sanitize_action(out, state).action, "HOLD")

    def test_raw_production_remaining_cost_takes_precedence_over_summary(self) -> None:
        state = {"available_production": ["powr"], "economy": {"cash": 850},
                 "production_items": ["e1@50%(~30 ticks)"],
                 "production": [{"item": "e1", "queue_type": "Infantry", "remaining_cost": 50}]}
        action = sanitize_action(Action("BUILD_STRUCTURE", [], structure_type="powr"), state)
        self.assertEqual(action.action, "BUILD_STRUCTURE")

    def test_string_queue_counts_toward_projected_unit_cap(self) -> None:
        state = {"available_production": ["e2"], "economy": {"cash": 10000},
                 "units_summary": [{"id": i, "type": "e1"} for i in range(31)],
                 "production_items": ["e1@20%(~40 ticks)", "e1@0%(~50 ticks)"]}
        action = sanitize_action(Action("BUILD_UNIT", [], unit_type="e2"), state)
        self.assertEqual(action.action, "HOLD")
        self.assertIn("unit cap", action.reason)

    def test_personality_fallback_differs_for_same_threat(self) -> None:
        state = {"map_info": {"width": 64, "height": 64},
                 "units": [{"actor_id": 1, "type": "e1", "can_attack": True, "cell_x": 20, "cell_y": 20}],
                 "buildings": [{"actor_id": 3, "type": "fact", "cell_x": 5, "cell_y": 5}],
                 "visible_enemies": [{"actor_id": 2, "type": "1tnk", "cell_x": 40, "cell_y": 40, "can_attack": True}],
                 "allies_summary": [{"actor_id": 4, "owner": "ally", "cell_x": 40, "cell_y": 40}]}
        self.assertEqual(fallback_action(state, "selfish").action, "RETREAT")
        helpful = fallback_action(state, "helpful")
        self.assertEqual(helpful.action, "ATTACK")
        self.assertEqual(helpful.target_id, 2)

    def test_bootstrap_arbitration_defers_repeated_economy_and_deploy_orders(self) -> None:
        build = Action("BUILD_UNIT", [], unit_type="e1")
        mutation = {"phase": "train_army"}
        self.assertEqual(_defer_vlm_economy(build, mutation).action, "HOLD")
        self.assertEqual(_defer_vlm_economy(build, None), build)
        deploy = Action("DEPLOY", [7])
        self.assertEqual(_suppress_order_during_deploy(deploy, {"phase": "deploy_mcv"}).action, "HOLD")

    def test_decision_log_marks_sanitized_vlm_response(self) -> None:
        class FakeVlm:
            def choose(self, state, personality):
                return Action("ATTACK", [999], target_id=123), "raw response"
        state = {"units": [{"actor_id": 1, "type": "e1", "can_attack": True}], "visible_enemies": []}
        with tempfile.TemporaryDirectory() as directory:
            engine = DecisionEngine(FakeVlm(), Path(directory), economy_control="vlm")
            engine.decide(state)
            row = json.loads((Path(directory) / "events.jsonl").read_text().splitlines()[0])
        self.assertEqual(row["decision_source"], "vlm_rejected")
        self.assertTrue(row["action_sanitized"])
        self.assertEqual(row["observation_counts"]["own_units"], 1)

    def test_executor_dispatches_empty_selection_economy_actions(self) -> None:
        class FakeGame:
            def __init__(self): self.calls = []
            async def call_tool(self, name, **kwargs):
                self.calls.append((name, kwargs)); return {"ok": True}
        import asyncio
        game = FakeGame()
        asyncio.run(_issue_action(game, Action("BUILD_UNIT", [], unit_type="e1", count=2)))
        asyncio.run(_issue_action(game, Action("BUILD_STRUCTURE", [], structure_type="powr", target=(4, 5))))
        self.assertEqual(game.calls, [("build_unit", {"unit_type": "e1", "count": 2}),
                                      ("build_and_place", {"building_type": "powr", "cell_x": 4, "cell_y": 5})])

    def test_decision_log_is_jsonl(self) -> None:
        state = {"units_summary": [], "enemy_summary": []}
        with tempfile.TemporaryDirectory() as directory:
            engine = DecisionEngine(None, Path(directory))
            action = engine.decide(state)
            row = json.loads((Path(directory) / "events.jsonl").read_text().splitlines()[0])
        self.assertEqual(action.action, "HOLD")
        self.assertEqual(row["event"], "decision")
        self.assertEqual(row["action"]["action"], "HOLD")


if __name__ == "__main__":
    unittest.main()
