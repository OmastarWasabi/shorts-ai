import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from shorts_ai.timeline import MAX_TIMELINE_BYTES, file_hash, read_timeline, round_half_away, validate_timeline

FIXTURES = Path(__file__).parent / "fixtures/migration"


class TimelineTests(unittest.TestCase):
    def setUp(self):
        self.timing = json.loads((FIXTURES / "timeline_valid.json").read_text())

    def test_shared_go_fixtures_and_audio_hash(self):
        cases = json.loads((FIXTURES / "cases.json").read_text())
        with tempfile.TemporaryDirectory() as temporary:
            audio = Path(temporary) / "voice.mp3"
            audio.write_bytes(cases["audio_text"].encode())
            for case in cases["timelines"]:
                with self.subTest(case=case["file"]):
                    if "expected_frames" in case:
                        timing, frames = read_timeline(FIXTURES / case["file"], audio, case["duration"], case["scene_count"])
                        self.assertEqual(frames, case["expected_frames"])
                        self.assertEqual(timing["audio_sha256"], file_hash(audio))
                    else:
                        with self.assertRaises(ValueError):
                            read_timeline(FIXTURES / case["file"], audio, case["duration"], case["scene_count"])
            audio.write_bytes(b"changed audio")
            with self.assertRaisesRegex(ValueError, "другой озвучке"):
                read_timeline(FIXTURES / "timeline_valid.json", audio, 4, 2)

    def test_go_rounding_and_neighbours(self):
        for value, expected in ((10.5, 11), (-10.5, -11), (.5, 1), (-.5, -1), (math.nextafter(.5, 0), 0)):
            self.assertEqual(round_half_away(value), expected)
        for case in json.loads((FIXTURES / "cases.json").read_text())["centiseconds"]:
            self.assertEqual(round_half_away(case["seconds"] * 100), case["expected"])

    def test_go_validation_failures(self):
        changes = [
            (("version",), 2), (("duration",), 5), (("scenes",), self.timing["scenes"][:1]),
            (("scenes", 1, "start"), 2.3), (("scenes", 1, "start"), 2.1),
            (("scenes", 1, "number"), 1), (("scenes", 1, "end"), 3.9),
            (("scenes", 1, "end"), math.nan), (("words",), []),
            (("words", 0, "end"), 0), (("words", 1, "end"), 2.5),
            (("words", 0, "scene"), 3), (("words", 2, "start"), .1),
            (("words", 3, "scene"), 1), (("words", 1, "text"), " "),
        ]
        for path, value in changes:
            with self.subTest(path=path, value=value):
                document = copy.deepcopy(self.timing)
                target = document
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                with self.assertRaises(ValueError):
                    validate_timeline(document, 4, 2)
        short = copy.deepcopy(self.timing)
        short["scenes"][0]["end"] = short["scenes"][1]["start"] = .001
        with self.assertRaisesRegex(ValueError, "короче"):
            validate_timeline(short, 4, 2)

    def test_json_schema_types_and_nonfinite_numbers(self):
        for path in (("version",), ("duration",), ("scenes", 0, "number"), ("scenes", 0, "start"),
                     ("scenes", 0, "end"), ("words", 0, "scene"), ("words", 0, "start"), ("words", 0, "end")):
            for value in (None, True, "1", [], {}, math.inf, math.nan, 10**400):
                with self.subTest(path=path, value=repr(value)[:30]):
                    document = copy.deepcopy(self.timing)
                    target = document
                    for key in path[:-1]:
                        target = target[key]
                    target[path[-1]] = value
                    with self.assertRaises(ValueError):
                        validate_timeline(document, 4, 2)
        for path in ((), ("scenes", 0), ("words", 0)):
            for change in ("extra", "missing"):
                document = copy.deepcopy(self.timing)
                target = document
                for key in path:
                    target = target[key]
                if change == "extra":
                    target["unknown"] = 1
                else:
                    target.pop(next(iter(target)))
                with self.assertRaises(ValueError):
                    validate_timeline(document, 4, 2)
        for document in (None, [], {}, 1, "timing"):
            with self.assertRaises(ValueError):
                validate_timeline(document, 4, 2)

    def test_document_size_encoding_trailing_data_and_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio, path = root / "audio", root / "timeline.json"
            audio.write_bytes(b"audio")
            for data in (b"{", b"{} {}", b"\xff", b"\xef\xbb\xbf{}", b"[]", b"null"):
                path.write_bytes(data)
                with self.assertRaises(ValueError):
                    read_timeline(path, audio, 4, 2)
            with path.open("wb") as output:
                output.truncate(MAX_TIMELINE_BYTES + 1)
            with self.assertRaisesRegex(ValueError, "16 МиБ"):
                read_timeline(path, audio, 4, 2)
            with self.assertRaisesRegex(ValueError, "обычным JSON"):
                read_timeline(root, audio, 4, 2)


if __name__ == "__main__":
    unittest.main()
