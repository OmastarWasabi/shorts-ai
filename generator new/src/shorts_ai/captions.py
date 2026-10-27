"""ASS-субтитры с группировкой и подсветкой слов, совместимые с Go-монтажом."""

from pathlib import Path

from .timeline import round_half_away

ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Narration,Arial,68,&H0000D7FF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,1,2,100,100,280,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def centiseconds(seconds: float) -> int:
    return round_half_away(seconds * 100)


def ass_time(value: int) -> str:
    return f"{value // 360000}:{value // 6000 % 60:02d}:{value // 100 % 60:02d}.{value % 100:02d}"


def escape_ass(text: str) -> str:
    return text.translate(str.maketrans({"\\": "＼", "{": "｛", "}": "｝", "\n": " ", "\r": " "}))


def _dialogue(words: list[dict]) -> str:
    start = centiseconds(words[0]["start"])
    end = max(start + 1, centiseconds(words[-1]["end"]))
    fragments = []
    for index, word in enumerate(words):
        next_start = centiseconds(words[index + 1]["start"]) if index + 1 < len(words) else end
        # Пауза включена в длительность текущего слова: следующая подсветка не спешит.
        duration = max(1, next_start - centiseconds(word["start"]))
        fragments.append(f"{{\\k{duration}}}{escape_ass(word['text'])}")
    return f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Narration,,0,0,0,,{' '.join(fragments)}\n"


def write_captions(path: Path, words: list[dict]) -> None:
    """Принимает слова из уже проверенного timeline; пишет во временную папку монтажа."""
    lines = [ASS_HEADER]
    group = []
    characters = 0
    for word in words:
        length = len(word["text"])
        if group:
            last = group[-1]
            if (len(group) == 4 or characters + 1 + length > 32 or
                    word["start"] - last["end"] > .4 or word["scene"] != last["scene"]):
                lines.append(_dialogue(group))
                group, characters = [], 0
        group.append(word)
        # Сохраняем исходный подсчёт Go, включая пробел после последнего слова.
        characters += length + 1
        if word["text"].endswith((".", "!", "?")):
            lines.append(_dialogue(group))
            group, characters = [], 0
    if group:
        lines.append(_dialogue(group))
    path.write_text("".join(lines), encoding="utf-8")
