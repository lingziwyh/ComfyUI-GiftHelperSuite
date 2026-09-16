import importlib
import json
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "gift_matting_test"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
video = importlib.import_module(PACKAGE + ".gift_video_matting")


class MattingNodeTests(unittest.TestCase):
    def test_names_and_no_research_registrations(self):
        self.assertEqual(set(video.NODE_CLASS_MAPPINGS),
                         {"GiftAutoShotSplit", "GiftMaskCheckFrames", "GiftAdaptiveMatting"})
        self.assertEqual(set(video.NODE_CLASS_MAPPINGS), set(video.NODE_DISPLAY_NAME_MAPPINGS))

    def test_sparse_indices_include_last_without_duplicates(self):
        self.assertEqual(video.checkpoint_indices(1, 12), [0])
        self.assertEqual(video.checkpoint_indices(25, 12), [0, 12, 24])
        self.assertEqual(video.checkpoint_indices(26, 12), [0, 12, 24, 25])
        self.assertEqual(video.checkpoint_indices(4, 1), [0, 1, 2, 3])
        for length, step in [(0, 12), (4, 0)]:
            with self.assertRaises(ValueError):
                video.checkpoint_indices(length, step)

    def test_samples_copy_and_preserve_order(self):
        images = torch.rand(26, 4, 3, 3)
        selected, indices = video.GiftMaskCheckFrames().sample(images, 12)
        self.assertEqual(json.loads(indices), [0, 12, 24, 25])
        self.assertTrue(torch.equal(selected, images[[0, 12, 24, 25]]))
        selected.zero_()
        self.assertGreater(images.sum().item(), 0)

    def test_disagreement_ignores_bottom_region_and_soft_edges(self):
        a = torch.zeros(100, 100)
        b = a.clone()
        b[65:] = 1
        self.assertEqual(video.disagreement_score(a, b, 0.6), 0)
        b[:] = 0.5
        self.assertEqual(video.disagreement_score(a, b, 1), 0)
        b[:50] = 1
        self.assertGreater(video.disagreement_score(a, b, 0.6), 0.1)

    def test_backfill_retains_existing_v2_behavior(self):
        self.assertEqual(video.backfill_weight(0, 0, 12), 1)
        self.assertEqual(video.backfill_weight(12, 12, 24), 0)
        self.assertEqual(video.backfill_weight(18, 12, 24), 0.5)
        self.assertEqual(video.backfill_weight(24, 12, 24), 1)

    def test_missing_backend_actionable_error(self):
        stub = types.ModuleType("nodes")
        stub.NODE_CLASS_MAPPINGS = {}
        with mock.patch.dict(sys.modules, {"nodes": stub}):
            with self.assertRaisesRegex(RuntimeError, "FuouM/ComfyUI-MatAnyone"):
                video.matanyone_backend()

    def test_download_switch_is_optional_and_defaults_on(self):
        for cls in [video.GiftAutoShotSplit, video.GiftAdaptiveMatting]:
            schema = cls.INPUT_TYPES()
            self.assertNotIn("auto_download", schema["required"])
            self.assertTrue(schema["optional"]["auto_download"][1]["default"])

    def test_portable_example_links_and_types(self):
        path = ROOT / "example_workflows/Gift_Auto_Video_Matting.json"
        text = path.read_text(encoding="utf-8")
        workflow = json.loads(text)
        self.assertNotIn("Research", text)
        self.assertNotRegex(text, r"[A-Za-z]:[/\\]")
        nodes = {n["id"]: n for n in workflow["nodes"]}
        for link_id, source, slot, target, input_slot, kind in workflow["links"]:
            self.assertIn(link_id, nodes[source]["outputs"][slot]["links"])
            self.assertEqual(nodes[target]["inputs"][input_slot]["link"], link_id)
            self.assertEqual(nodes[source]["outputs"][slot]["type"], kind)
            self.assertEqual(nodes[target]["inputs"][input_slot]["type"], kind)
        self.assertTrue(set(video.NODE_CLASS_MAPPINGS).issubset({n["type"] for n in nodes.values()}))
        for node in nodes.values():
            if node["type"] == "VHS_LoadVideo":
                self.assertTrue((ROOT / "example_workflows/assets" / node["widgets_values"]["video"]).is_file())
            if node["type"] == "LoadImage":
                self.assertTrue((ROOT / "example_workflows/assets" / node["widgets_values"][0]).is_file())


if __name__ == "__main__":
    unittest.main()
