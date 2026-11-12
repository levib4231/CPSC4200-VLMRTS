import json
import tempfile
import unittest
from pathlib import Path

from agent import DecisionEngine, fallback_action, parse_action, sanitize_action


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
            "units_summary": [{"id": 101, "type": "e1"}],
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
            "units_summary": [{"id": 101, "type": "e1"}],
            "buildings_summary": [{"id": 303, "type": "fact", "cell_x": 7, "cell_y": 9}],
        }
        action = sanitize_action(parse_action('{"action":"RETREAT","unit_ids":[101]}'), state)
        self.assertEqual(action.target, (7.0, 9.0))

    def test_move_is_clamped_to_map(self) -> None:
        state = {
            "map": {"width": 64, "height": 64},
            "units_summary": [{"id": 101, "type": "e1"}],
        }
        action = sanitize_action(parse_action('{"action":"MOVE","target":{"x":99,"y":-2}}'), state)
        self.assertEqual(action.target, (63.0, 0.0))

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
