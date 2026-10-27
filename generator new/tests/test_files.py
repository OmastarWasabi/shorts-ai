import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from shorts_ai.files import project_directory, project_path, read_utf8, write_bytes_atomic


class FileTests(unittest.TestCase):
    def test_paths_with_spaces_quotes_and_cyrillic(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "ролик с 'кавычками'"
            directory.mkdir()
            root = project_directory(directory)
            self.assertEqual(root, directory.resolve())
            self.assertEqual(project_path(root, "images/01.png"), root / "images" / "01.png")
            absolute = Path(temporary) / "другой файл.mp4"
            self.assertEqual(project_path(root, absolute), absolute)
            with self.assertRaisesRegex(ValueError, "Каталог проекта"):
                project_directory(directory / "missing")
            file = directory / "file.txt"
            file.write_text("text")
            with self.assertRaises(ValueError):
                project_directory(file)

    def test_reading_removes_bom_but_preserves_newlines(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "текст.txt"
            path.write_bytes("\ufeff  Ёжик\r\nпроснулся.\r\n".encode("utf-8"))
            self.assertEqual(read_utf8(path), "  Ёжик\r\nпроснулся.\r\n")
            path.write_bytes(b"secret\xff")
            with self.assertRaisesRegex(ValueError, "UTF-8") as caught:
                read_utf8(path)
            self.assertNotIn("secret", str(caught.exception))

    def test_atomic_save_and_explicit_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "новая папка" / "output.bin"
            write_bytes_atomic(path, b"first")
            with self.assertRaises(FileExistsError):
                write_bytes_atomic(path, b"second")
            self.assertEqual(path.read_bytes(), b"first")
            write_bytes_atomic(path, b"replacement", overwrite=True)
            self.assertEqual(path.read_bytes(), b"replacement")
            self.assertEqual(list(path.parent.glob(".shorts-ai-*")), [])

    def test_file_created_by_other_process_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "output.bin"
            real_link = os.link

            def publish(source, destination):
                Path(destination).write_bytes(b"other process")
                real_link(source, destination)

            with patch("shorts_ai.files.os.link", side_effect=publish), self.assertRaises(FileExistsError):
                write_bytes_atomic(path, b"our result")
            self.assertEqual(path.read_bytes(), b"other process")
            self.assertEqual(list(path.parent.glob(".shorts-ai-*")), [])

    def test_failed_publication_leaves_no_partial_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "output.bin"
            with patch("shorts_ai.files.os.link", side_effect=OSError("test failure")), self.assertRaises(OSError):
                write_bytes_atomic(path, b"result")
            self.assertFalse(path.exists())
            self.assertEqual(list(path.parent.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
