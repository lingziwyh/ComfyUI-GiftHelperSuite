from __future__ import annotations

import pathlib
import re
import unittest
from urllib.parse import unquote, urlsplit

from tests.test_registration import load_suite


ROOT = pathlib.Path(__file__).resolve().parents[1]


class DocumentationTests(unittest.TestCase):
    def test_release_version_is_consistent(self):
        suite = load_suite()
        version = suite.__version__
        self.assertRegex(version, r"^\d+\.\d+\.\d+$")
        self.assertIn("__version__", suite.__all__)
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        self.assertIn(f"**v{version} ·", readme)
        self.assertIn(f"## v{version} —", changelog)

    def test_node_index_matches_registration(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        index = readme.split("## 节点索引\n", 1)[1].split("\n## ", 1)[0]
        ids = re.findall(r"^\| `([^`]+)` \|", index, re.MULTILINE)
        registered = set(load_suite().NODE_CLASS_MAPPINGS)
        self.assertEqual(len(ids), len(set(ids)), "Duplicate node in README index")
        self.assertEqual(set(ids), registered)
        self.assertIn(f"**{len(registered)} 个节点**", index)

    def test_local_documentation_links_exist(self):
        documents = sorted(ROOT.glob("README*.md")) + [
            ROOT / "CHANGELOG.md",
            ROOT / "THIRD_PARTY_NOTICES.md",
            ROOT / "example_workflows" / "README.md",
        ]
        for document in documents:
            content = document.read_text(encoding="utf-8")
            for target in re.findall(r"\[[^\]]*\]\(([^)\s]+)\)", content):
                parsed = urlsplit(target)
                if parsed.scheme or parsed.netloc or not parsed.path:
                    continue
                with self.subTest(document=document.name, target=target):
                    self.assertTrue((document.parent / unquote(parsed.path)).exists())


if __name__ == "__main__":
    unittest.main()
