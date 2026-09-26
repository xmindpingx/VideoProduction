"""Animated captions (ASS, burned in with libass) + an SRT copy for platforms that take caption files.

Styles: pop (1-3 big words at a time, the spoken word lights up), karaoke (a full line with the spoken word
highlighted), clean (plain sentence subtitles). All sit above the bottom UI area of TikTok/Reels/Shorts.
"""
import re

from .. import config

GOLD = "&H0037AFD4&"      # #D4AF37
AZURE = "&H00F28F4F&"     # #4F8FF2
WHITE = "&H00FFFFFF&"
PLATINUM = "&H00E2E4E5&"  # #E5E4E2
CAPTION_Y = 1340          # bottom edge of the caption block (1920-high frame)

STYLES = {
    "pop": {"font": "Anton", "size": 124, "outline": 8, "max_words": 3, "max_chars": 16, "upper": True},
    "karaoke": {"font": "Archivo Black", "size": 76, "outline": 6, "max_words": 5, "max_chars": 22, "upper": False},
    "clean": {"font": "Archivo Black", "size": 62, "outline": 0, "max_words": 7, "max_chars": 30, "upper": False},
}


def _ts(t):
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def _srt_ts(t):
    t = max(0.0, t)
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _clean(text):
    return re.sub(r"[{}\\]", "", text).strip()


def chunk_words(words, max_words, max_chars, gap=0.45):
    """Group timed words into caption chunks; break on punctuation, pauses and length."""
    chunks, cur = [], []
    for w in words:
        if cur:
            chars = sum(len(x[2]) for x in cur) + len(cur) + len(w[2])
            if len(cur) >= max_words or chars > max_chars or w[0] - cur[-1][1] > gap:
                chunks.append(cur)
                cur = []
        cur.append(w)
        if re.search(r"[.!?;:,]$", w[2]) and len(cur) >= 2:
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    return chunks


def _wrap_hook(text, width=16):
    words, lines, line = text.split(), [], ""
    for w in words:
        if line and len(line) + 1 + len(w) > width:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        lines.append(line)
    return r"\N".join(lines[:3])


def build(words, style, hook_text, hook_end, total, emphasis, ass_path, srt_path):
    """words: [(start, end, text)] on the final timeline. Writes the .ass and .srt files."""
    st = STYLES.get(style)
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {config.OUT_W}
PlayResY: {config.OUT_H}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{st['font'] if st else 'Archivo Black'},{st['size'] if st else 60},{WHITE},{WHITE},&H00000000&,&H99000000&,0,0,0,0,100,100,1,0,{3 if style == 'clean' else 1},{st['outline'] if st else 5},{3 if style == 'clean' else 0},2,80,80,{config.OUT_H - CAPTION_Y},1
Style: Hook,Anton,128,{PLATINUM},{PLATINUM},&H00000000&,&H00000000&,0,0,0,0,100,100,2,0,1,8,0,8,70,70,250,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    if hook_text and hook_end > 0.4:
        events.append(f"Dialogue: 1,{_ts(0.05)},{_ts(hook_end)},Hook,,0,0,0,,{{\\fad(180,220)}}{_wrap_hook(_clean(hook_text).upper())}")
    emph = {e.lower().strip(".,!?") for e in emphasis}
    srt = []
    if st and words:
        chunks = chunk_words(words, st["max_words"], st["max_chars"])
        for ci, chunk in enumerate(chunks):
            texts = [_clean(w[2]) for w in chunk]
            if st["upper"]:
                texts = [t.upper() for t in texts]
            c_start, c_end = chunk[0][0], min(total, chunk[-1][1] + 0.25)
            if ci + 1 < len(chunks):  # never overlap the next chunk, or libass stacks them and the line jumps
                c_end = min(c_end, chunks[ci + 1][0][0])
            srt.append((c_start, c_end, " ".join(texts)))
            if style == "clean":
                events.append(f"Dialogue: 0,{_ts(c_start)},{_ts(c_end)},Cap,,0,0,0,,{{\\fad(60,60)}}{' '.join(texts)}")
                continue
            for k, w in enumerate(chunk):
                s = c_start if k == 0 else w[0]
                e = chunk[k + 1][0] if k + 1 < len(chunk) else c_end
                if e - s < 0.05:
                    continue
                parts = []
                for j, t in enumerate(texts):
                    if j == k:
                        grow = r"\fscx112\fscy112" if style == "pop" else ""
                        parts.append(f"{{\\c{GOLD}{grow}}}{t}{{\\c{WHITE}\\fscx100\\fscy100}}")
                    elif chunk[j][2].lower().strip(".,!?") in emph:
                        parts.append(f"{{\\c{AZURE}}}{t}{{\\c{WHITE}}}")
                    else:
                        parts.append(t)
                intro = r"{\fad(50,0)}" if k == 0 else ""
                events.append(f"Dialogue: 0,{_ts(s)},{_ts(e)},Cap,,0,0,0,,{intro}{' '.join(parts)}")
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(head + "\n".join(events) + "\n")
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, (s, e, t) in enumerate(srt, 1):
            f.write(f"{i}\n{_srt_ts(s)} --> {_srt_ts(e)}\n{t}\n\n")
    return len(events)
