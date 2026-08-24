from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import re
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLE_DIR = ROOT / "example_workflows"
WORKFLOW_FILES = {
    "postfx": EXAMPLE_DIR / "Gift_PostFX_Example.json",
    "chroma": EXAMPLE_DIR / "Gift_Chroma_Master_Example.json",
}
ICON_WORKFLOW_FILE = EXAMPLE_DIR / "Gift_Icon_Auto_Restore_Production.json"


def load_asset_module():
    spec = importlib.util.spec_from_file_location(
        "gift_helper_suite_example_assets_test",
        ROOT / "example_assets.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExampleWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflows = {
            name: json.loads(path.read_text(encoding="utf-8"))
            for name, path in WORKFLOW_FILES.items()
        }
        cls.assets = load_asset_module()
        cls.icon_workflow = json.loads(ICON_WORKFLOW_FILE.read_text(encoding="utf-8"))

    def test_workflows_are_portable_ui_workflows(self):
        for name, path in WORKFLOW_FILES.items():
            workflow = self.workflows[name]
            text = path.read_text(encoding="utf-8")
            self.assertEqual(workflow["version"], 0.4)
            self.assertIn("nodes", workflow)
            self.assertNotIn("class_type", text)
            self.assertNotIn("fullpath", text)
            self.assertNotIn("comfyui_mcp", text)
            self.assertNotRegex(text, re.compile(r"[A-Za-z]:\\\\"))
            self.assertNotRegex(text, re.compile(r"ProChroma\w*V5"))

    def test_workflows_reference_only_bundled_input_assets(self):
        expected_video = {
            "postfx": "GiftHelperSuite_PostFX_Source_bebfe94a.mp4",
            "chroma": "GiftHelperSuite_Chroma_Source_92532474.mp4",
        }
        referenced_assets = set()
        for name, workflow in self.workflows.items():
            load_videos = [node for node in workflow["nodes"] if node["type"] == "VHS_LoadVideo"]
            load_images = [node for node in workflow["nodes"] if node["type"] == "LoadImage"]
            self.assertEqual(len(load_videos), 1)
            self.assertEqual(len(load_images), 1)

            load_video = load_videos[0]
            video_name = load_video["widgets_values"]["video"]
            self.assertEqual(video_name, expected_video[name])
            self.assertEqual(load_video["widgets_values_named"]["video"], video_name)
            self.assertEqual(load_video["widgets_values"]["format"], "None")
            self.assertEqual(load_video["widgets_values_named"]["format"], "None")
            referenced_assets.add(video_name)

            load_image = load_images[0]
            image_name = load_image["widgets_values"][0]
            self.assertEqual(image_name, "GiftHelperSuite_Example_Background_213ee241.png")
            self.assertEqual(load_image["widgets_values_named"]["image"], image_name)
            referenced_assets.add(image_name)

            for node in workflow["nodes"]:
                if node["type"] == "VHS_VideoCombine":
                    self.assertNotIn("videopreview", node["widgets_values"])
                    self.assertNotIn("videopreview", node["widgets_values_named"])

        self.assertEqual(referenced_assets, set(self.assets.EXAMPLE_ASSET_MANIFEST))

    def test_chroma_workflow_uses_current_node_ids(self):
        node_types = {node["type"] for node in self.workflows["chroma"]["nodes"]}
        self.assertIn("GiftChromaMaster", node_types)
        self.assertIn("GiftChromaMasterPreview", node_types)
        self.assertIn("GiftChromaMasterPackRGBA", node_types)
        self.assertNotIn("ProChromaPreviewV5", node_types)
        self.assertNotIn("ProChromaPackRGBAV5", node_types)

    def test_icon_production_workflow_is_sanitized_and_uses_builtin_preview(self):
        workflow = self.icon_workflow
        text = ICON_WORKFLOW_FILE.read_text(encoding="utf-8")
        self.assertEqual(workflow["version"], 0.4)
        self.assertNotIn("class_type", text)
        self.assertNotIn("fullpath", text)
        self.assertNotIn("comfyui_mcp", text)
        self.assertNotRegex(text, re.compile(r"[A-Za-z]:\\\\"))
        self.assertNotIn("Clipboard", text)

        icon_nodes = [
            node for node in workflow["nodes"] if node["type"] == "GiftIconAutoRestore"
        ]
        self.assertEqual(len(icon_nodes), 1)
        icon = icon_nodes[0]
        input_by_name = {item["name"]: item for item in icon["inputs"]}
        self.assertIsNone(input_by_name["preview_background"]["link"])
        self.assertEqual(icon["widgets_values"][1], 0.105)
        self.assertEqual(icon["widgets_values"][10], 8)
        self.assertEqual(icon["widgets_values"][14], 1680)
        self.assertEqual(icon["widgets_values"][15], "auto")

        load_images = [node for node in workflow["nodes"] if node["type"] == "LoadImage"]
        self.assertEqual(len(load_images), 1)
        self.assertEqual(
            load_images[0]["widgets_values"][0], "GiftHelperSuite_Icon_Source.png"
        )

    def test_bundled_assets_match_manifest(self):
        for filename, metadata in self.assets.EXAMPLE_ASSET_MANIFEST.items():
            path = self.assets.EXAMPLE_ASSET_DIR / filename
            self.assertTrue(path.is_file(), filename)
            self.assertTrue(self.assets._matches_manifest(path, metadata), filename)

    def test_bundled_media_contains_no_embedded_workflow_metadata(self):
        forbidden = (
            b"workflow",
            b"prompt",
            b"comment",
            b"ai-t8-video-onekey",
            b"full-frame_00047",
            b"animatediff_00124",
            b"c:\\users\\admin",
            b"e:\\ai-t8",
        )
        for filename in self.assets.EXAMPLE_ASSET_MANIFEST:
            path = self.assets.EXAMPLE_ASSET_DIR / filename
            payload = path.read_bytes().lower()
            if path.suffix.lower() == ".mp4":
                self.assertIn(b"ftyp", payload[:64], filename)
                self.assertIn(b"moov", payload, filename)
                for marker in forbidden:
                    self.assertNotIn(marker, payload, f"{filename}: {marker!r}")
            elif path.suffix.lower() == ".png":
                self.assertTrue(payload.startswith(b"\x89png\r\n\x1a\n"), filename)

    def test_asset_install_is_idempotent_and_never_overwrites_conflicts(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GIFT_HELPER_SKIP_EXAMPLE_ASSETS", None)
            with tempfile.TemporaryDirectory() as temp_dir:
                first = self.assets.install_example_assets(temp_dir)
                self.assertEqual(set(first.values()), {"installed"})

                second = self.assets.install_example_assets(temp_dir)
                self.assertEqual(set(second.values()), {"present"})

                filename = "GiftHelperSuite_Example_Background_213ee241.png"
                target = pathlib.Path(temp_dir) / filename
                target.write_bytes(b"user-owned file")
                third = self.assets.install_example_assets(temp_dir)
                self.assertEqual(third[filename], "conflict")
                self.assertEqual(target.read_bytes(), b"user-owned file")

    def test_asset_install_opt_out_writes_nothing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.dict(os.environ, {"GIFT_HELPER_SKIP_EXAMPLE_ASSETS": "1"}):
                results = self.assets.install_example_assets(temp_dir)
            self.assertEqual(set(results.values()), {"skipped"})
            self.assertEqual(list(pathlib.Path(temp_dir).iterdir()), [])

    def test_concurrent_install_publishes_only_verified_complete_files(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GIFT_HELPER_SKIP_EXAMPLE_ASSETS", None)
            with tempfile.TemporaryDirectory() as temp_dir:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    results = list(executor.map(self.assets.install_example_assets, [temp_dir, temp_dir]))

                for filename, metadata in self.assets.EXAMPLE_ASSET_MANIFEST.items():
                    statuses = {result[filename] for result in results}
                    self.assertTrue(statuses.issubset({"installed", "present"}), filename)
                    self.assertIn("installed", statuses, filename)
                    self.assertTrue(
                        self.assets._matches_manifest(pathlib.Path(temp_dir) / filename, metadata),
                        filename,
                    )
                self.assertFalse(list(pathlib.Path(temp_dir).glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
