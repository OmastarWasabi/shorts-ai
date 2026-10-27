import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from shorts_ai.captions import ass_time, centiseconds, escape_ass, write_captions
from shorts_ai.ffmpeg import require_ass


class CaptionTests(unittest.TestCase):
    def test_ass_matches_go_for_grouping_pauses_unicode_and_rounding(self):
        cases = json.loads((Path(__file__).parent / "fixtures/migration/captions.json").read_text())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "captions.ass"
            for case in cases:
                with self.subTest(case=case["name"]):
                    write_captions(path, case["words"])
                    self.assertEqual(path.read_text(), case["expected"])

    def test_pause_delays_next_highlight_and_tags_are_escaped(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "captions.ass"
            write_captions(path, [{"text": "Один", "start": .1, "end": .3, "scene": 1},
                                  {"text": r"{\pos(0,0)}два", "start": .6, "end": .8, "scene": 1}])
            text = path.read_text()
            self.assertIn(r"{\k50}Один", text)
            self.assertIn(r"{\k20}｛＼pos(0,0)｝два", text)
            self.assertNotIn(r"\pos", text)
            self.assertIn("0:00:00.10,0:00:00.80", text)
        self.assertEqual(escape_ass("{x}\\N\r\ny"), "｛x｝＼N  y")

    def test_time_rollovers_and_minimum_event_duration(self):
        self.assertEqual(centiseconds(.125), 13)
        self.assertEqual(ass_time(6000), "0:01:00.00")
        self.assertEqual(ass_time(360000), "1:00:00.00")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "captions.ass"
            write_captions(path, [{"text": "Кадр", "start": 1.001, "end": 1.002, "scene": 1}])
            self.assertIn(r"0:00:01.00,0:00:01.01,Narration,,0,0,0,,{\k1}Кадр", path.read_text())

    def test_ass_capability_is_detected_by_filter_name(self):
        with patch("shorts_ai.ffmpeg.ffmpeg", return_value=b" .. ass V->V Render ASS\n"):
            require_ass("encoder")
        for output in (b"", b" .. subtitles V->V Render ass files\n", b" .. bass A->A Filter\n"):
            with self.subTest(output=output), patch("shorts_ai.ffmpeg.ffmpeg", return_value=output):
                with self.assertRaisesRegex(ValueError, "--no-subtitles"):
                    require_ass("encoder")


if __name__ == "__main__":
    unittest.main()
