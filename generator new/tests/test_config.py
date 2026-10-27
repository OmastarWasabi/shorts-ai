import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from shorts_ai.config import GenerationConfig, load_env, read_bool_env, read_config


FIXTURES = Path(__file__).parent / "fixtures" / "migration" / "cases.json"


class ConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = json.loads(FIXTURES.read_text(encoding="utf-8"))

    def test_defaults_and_all_generation_modes(self):
        self.assertEqual(read_config({}), GenerationConfig(True, True))
        for voice in (True, False):
            for images in (True, False):
                with self.subTest(voice=voice, images=images):
                    env = {"GENERATE_VOICE": str(voice), "GENERATE_IMAGES": str(images)}
                    self.assertEqual(read_config(env), GenerationConfig(voice, images))

    def test_go_boolean_fixtures(self):
        for case in self.cases["booleans"]:
            for name in ("GENERATE_VOICE", "GENERATE_IMAGES"):
                with self.subTest(value=case["input"], name=name):
                    env = {name: case["input"]}
                    if "error_contains" in case:
                        with self.assertRaisesRegex(ValueError, case["error_contains"]):
                            read_bool_env(name, env)
                    else:
                        self.assertIs(read_bool_env(name, env), case["expected"])

    def test_go_env_fixtures(self):
        for case in self.cases["env"]:
            with self.subTest(case=case["name"]), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / ".env"
                path.write_bytes(case["content"].encode("utf-8"))
                env = {}
                if "initial" in case:
                    env["MIGRATION_BASELINE_VALUE"] = case["initial"]
                if "error_contains" in case:
                    with self.assertRaisesRegex(ValueError, case["error_contains"]) as caught:
                        load_env(path, env)
                    self.assertNotIn("do-not-print-this", str(caught.exception))
                else:
                    load_env(path, env)
                    self.assertEqual(env["MIGRATION_BASELINE_VALUE"], case["expected"])

    def test_real_environment_priority_including_empty_values(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            path = Path(directory) / ".env"
            path.write_text("GENERATE_VOICE=false\nGENERATE_IMAGES=false\n", encoding="utf-8")
            os.environ["GENERATE_VOICE"] = ""
            load_env(path)
            self.assertEqual(os.environ["GENERATE_VOICE"], "")
            self.assertEqual(os.environ["GENERATE_IMAGES"], "false")
            with self.assertRaisesRegex(ValueError, "GENERATE_VOICE"):
                read_config()

    def test_missing_env_and_invalid_utf8(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            env = {}
            load_env(path, env)
            self.assertEqual(env, {})
            path.write_bytes(b"SECRET=do-not-print-this\xff")
            with self.assertRaisesRegex(ValueError, "UTF-8") as caught:
                load_env(path, env)
            self.assertNotIn("do-not-print-this", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
