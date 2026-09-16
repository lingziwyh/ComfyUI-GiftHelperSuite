import hashlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
from urllib.error import URLError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matting_models as models


class Response(io.BytesIO):
    status = 200


class ModelDownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.payload = b"small checkpoint fixture"
        self.asset = models.ModelAsset("test.pth", "https://example.invalid/test.pth",
                                       hashlib.sha256(self.payload).hexdigest(), ("legacy/test.pth",))
        self.patcher = mock.patch.dict(models.MODELS, {"test": self.asset})
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.env = mock.patch.dict(os.environ, {"GIFT_HELPER_AUTO_DOWNLOAD": "1"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_missing_downloads_once_and_reuses_offline(self):
        with mock.patch.object(models, "urlopen", return_value=Response(self.payload)) as fetch:
            target = models.ensure_model("test", self.root)
            self.assertEqual(target.read_bytes(), self.payload)
            self.assertEqual(models.ensure_model("test", self.root, auto_download=False), target)
            fetch.assert_called_once()
        self.assertFalse(list(self.root.rglob("*.part")))

    def test_legacy_and_extra_paths_do_not_download(self):
        old = self.root / "legacy/test.pth"
        old.parent.mkdir()
        old.write_bytes(self.payload)
        with mock.patch.object(models, "urlopen") as fetch:
            self.assertEqual(models.ensure_model("test", self.root), old)
            old.rename(self.root / "external.pth")
            self.assertEqual(models.ensure_model("test", self.root, extra_paths=[self.root / "external.pth"]),
                             self.root / "external.pth")
            fetch.assert_not_called()

    def test_disabled_reports_manual_path(self):
        with mock.patch.object(models, "urlopen") as fetch:
            with self.assertRaisesRegex(FileNotFoundError, "GiftHelperSuite"):
                models.ensure_model("test", self.root, auto_download=False)
            with mock.patch.dict(os.environ, {"GIFT_HELPER_AUTO_DOWNLOAD": "0"}):
                with self.assertRaises(FileNotFoundError):
                    models.ensure_model("test", self.root)
            fetch.assert_not_called()

    def test_invalid_existing_file_never_overwritten(self):
        target = self.root / "GiftHelperSuite/test.pth"
        target.parent.mkdir()
        target.write_bytes(b"custom or damaged checkpoint")
        with mock.patch.object(models, "urlopen") as fetch:
            with self.assertRaisesRegex(ValueError, "NOT overwritten"):
                models.ensure_model("test", self.root)
            fetch.assert_not_called()
        self.assertEqual(target.read_bytes(), b"custom or damaged checkpoint")

    def test_bad_download_not_published(self):
        with mock.patch.object(models, "urlopen", return_value=Response(b"<html>error</html>")):
            with self.assertRaises(ValueError):
                models.ensure_model("test", self.root)
        self.assertFalse((self.root / "GiftHelperSuite/test.pth").exists())
        self.assertFalse(list(self.root.rglob("*.part")))

    def test_network_failure_cleans_up(self):
        with mock.patch.object(models, "urlopen", side_effect=URLError("offline")):
            with self.assertRaisesRegex(RuntimeError, "Manual URL"):
                models.ensure_model("test", self.root)
        self.assertFalse(list(self.root.rglob("*.part")))

    def test_interrupt_cleans_up(self):
        cancel = mock.Mock(side_effect=[None, InterruptedError("cancelled")])
        with mock.patch.object(models, "urlopen", return_value=Response(self.payload)):
            with self.assertRaises(InterruptedError):
                models.ensure_model("test", self.root, cancelled=cancel)
        self.assertFalse(list(self.root.rglob("*.part")))

    def test_concurrent_publication_does_not_overwrite(self):
        original = os.link
        def publish_other_then_link(source, target):
            Path(target).write_bytes(self.payload)
            original(source, target)
        with mock.patch.object(models, "urlopen", return_value=Response(self.payload)), \
                mock.patch.object(models.os, "link", side_effect=publish_other_then_link):
            self.assertEqual(models.ensure_model("test", self.root).read_bytes(), self.payload)

    def test_unknown_model_rejected_before_network(self):
        with mock.patch.object(models, "urlopen") as fetch:
            with self.assertRaises(KeyError):
                models.ensure_model("../../other", self.root)
            fetch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
