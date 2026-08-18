from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE_NAME = "gift_helper_suite_registration_test"


def load_suite():
    if PACKAGE_NAME in sys.modules:
        return sys.modules[PACKAGE_NAME]
    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE_NAME] = module
    spec.loader.exec_module(module)
    return module


class RegistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suite = load_suite()

    def test_new_nodes_are_registered_with_complete_display_names(self):
        expected = {
            "GiftChromaMaster",
            "GiftChromaMasterKeyer",
            "GiftChromaMasterCleaner",
            "GiftChromaMasterDespill",
            "GiftChromaMasterDiagnostics",
            "GiftChromaMasterPreview",
            "GiftChromaMasterPackRGBA",
            "GiftMaskRamp",
            "GiftMaskFadeInOut",
            "GiftFrameSlice",
            "GiftMaskBlend",
        }
        self.assertTrue(expected.issubset(self.suite.NODE_CLASS_MAPPINGS))
        self.assertEqual(
            set(self.suite.NODE_CLASS_MAPPINGS),
            set(self.suite.NODE_DISPLAY_NAME_MAPPINGS),
        )

    def test_legacy_chroma_nodes_are_not_present(self):
        legacy = {
            "ProChromaTrioV5",
            "ProChromaScreenKeyerV5",
            "ProChromaKeyCleanerV5",
            "ProChromaSpillSuppressorV5",
            "ProChromaDiagnosticsV5",
            "ProChromaPreviewV5",
            "ProChromaPackRGBAV5",
            "KeylightCoreV4",
            "KeylightCoreHubV3",
            "Key Spill/Algo Args (V2.3.6fixE2_clean)",
            "Key Protect Highlights Args (V2.3.6fixE2_clean)",
            "Key Edge Args (V2.3.6fixE2_clean)",
            "Key Matte Math Args (V2.3.6fixE2_clean)",
            "Key Sampler Args (V2.3.6fixE2_clean)",
        }
        self.assertFalse(legacy.intersection(self.suite.NODE_CLASS_MAPPINGS))

    def test_chroma_state_and_category_are_isolated(self):
        chroma = self.suite.NODE_CLASS_MAPPINGS["GiftChromaMaster"]
        self.assertEqual(chroma.CATEGORY, "GiftHelperSuite/Chroma")
        nodes_module = sys.modules[chroma.__module__]
        self.assertEqual(nodes_module.STATE_TYPE, "GIFT_CHROMA_MASTER_STATE")
        state = nodes_module.screen_key_stage(
            __import__("torch").zeros(1, 2, 2, 3),
            screen_mode="manual",
            screen_color="#00ff00",
        )[0]
        self.assertEqual(state.schema, "gift_chroma_master_v1")

    def test_v5_only_package_has_no_legacy_web_directory(self):
        self.assertFalse(hasattr(self.suite, "WEB_DIRECTORY"))


if __name__ == "__main__":
    unittest.main()
