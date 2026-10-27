import json
from pathlib import Path
import re
import tempfile
import unittest

from shorts_ai.prompts import Prompt, read_prompts


class PromptTests(unittest.TestCase):
    def test_shared_go_fixtures(self):
        cases = json.loads((Path(__file__).parent / "fixtures/migration/cases.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prompts.txt"
            for case in cases["prompts"]:
                with self.subTest(case=case["name"]):
                    path.write_bytes(case["content"].encode("utf-8"))
                    if "error_contains" in case:
                        with self.assertRaisesRegex(ValueError, re.escape(case["error_contains"])):
                            read_prompts(path)
                    else:
                        expected = [Prompt(item["number"], item["text"]) for item in case["expected"]]
                        self.assertEqual(read_prompts(path), expected)

    def test_invalid_inputs_from_go_tests(self):
        cases = {
            "empty": " \n", "wrong first": "[02]\nScene.",
            "empty last": "[01]\nFirst.\n[02]\n", "broken marker": "[01]\nFirst.\n[02\nSecond.",
            "overflow": "[99999999999999999999999]\nFirst.", "zero": "[00]\nFirst.",
            "unicode marker": "[０１]\nFirst.", "text before marker": "Before.\n[01]\nFirst.",
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prompts.txt"
            for name, content in cases.items():
                with self.subTest(name=name):
                    path.write_text(content, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        read_prompts(path)
            path.write_bytes(b"[01]\n\xff")
            with self.assertRaisesRegex(ValueError, "UTF-8"):
                read_prompts(path)

    def test_limit_counts_unicode_characters_not_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prompts.txt"
            for char in ("я", "🌲"):
                with self.subTest(char=char):
                    path.write_text("[01]\n" + char * 32000, encoding="utf-8")
                    self.assertEqual(read_prompts(path), [Prompt(1, char * 32000)])
                    path.write_text("[01]\n" + char * 32001, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "32000"):
                        read_prompts(path)


if __name__ == "__main__":
    unittest.main()
