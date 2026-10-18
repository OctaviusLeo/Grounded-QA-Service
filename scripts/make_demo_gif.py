"""Record docs/media/demo.gif from the running service: python scripts/make_demo_gif.py

Needs the Compose stack (`docker compose up`). Sends real questions to POST /answer/stream,
records every server-sent event with its arrival time, and draws them as terminal frames. The
answers, citations, and request IDs shown are exactly what the service returned; waits longer
than 1.5 s are shortened in the animation (the real times are written to docs/media/demo.json).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx
from PIL import Image, ImageDraw, ImageFont

URL = "http://127.0.0.1:8000/answer/stream"
QUESTIONS = [
    "When was the Gerald Ratner Athletics Center constructed?",
    "Who was Edward the Confessor's half-brother?",
    "Who won the 2030 FIFA World Cup?",
]
W, H, PAD, LINE = 1100, 330, 18, 22
BG, FG, DIM, OK, WARN = (24, 26, 33), (220, 223, 228), (130, 136, 150), (120, 200, 140), (230, 180, 90)
MAX_WAIT_S = 1.5


def font(size: int = 16) -> ImageFont.FreeTypeFont:
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "/usr/share/fonts/TTF/DejaVuSansMono.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def record(question: str) -> list[dict]:
    events, start = [], time.perf_counter()
    with httpx.stream("POST", URL, json={"query": question}, timeout=120) as r:
        r.raise_for_status()
        event = None
        for line in r.iter_lines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                events.append({"t": time.perf_counter() - start, "event": event, "data": json.loads(line[6:])})
    return events


def wrap(text: str, width: int = 92) -> list[str]:
    out, line = [], ""
    for word in text.split():
        if len(line) + len(word) + 1 > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return out + ([line] if line else [])


def frame(lines: list[tuple[str, tuple[int, int, int]]], f: ImageFont.FreeTypeFont) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((PAD, 8), "Grounded QA Service  ·  POST /answer/stream  ·  answers only from cited passages", fill=DIM, font=f)
    for i, (text, color) in enumerate(lines[-(H // LINE - 3):]):
        d.text((PAD, 40 + i * LINE), text, fill=color, font=f)
    return img


def main() -> int:
    f = font()
    frames, durations, log = [], [], []
    screen: list[tuple[str, tuple[int, int, int]]] = []
    for q in QUESTIONS:
        events = record(q)
        log.append({"question": q, "events": events})
        screen.append((f"$ curl -N localhost:8000/answer/stream -d '{{\"query\": \"{q}\"}}'", FG))
        frames.append(frame(screen, f))
        durations.append(900)
        last_t, answer = 0.0, ""
        for ev in events:
            wait = min(ev["t"] - last_t, MAX_WAIT_S)
            last_t = ev["t"]
            if ev["event"] == "chunk":
                answer += ev["data"]["text"]
                body = [(f"  {line}", OK if "not found" not in answer else WARN) for line in wrap(answer)]
                frames.append(frame(screen + body, f))
                durations.append(max(80, int(wait * 1000)))
            elif ev["event"] == "done":
                d = ev["data"]
                screen += [(f"  {line}", OK if not d["refused"] else WARN) for line in wrap(d["text"])]
                cited = ", ".join(d["cited_ids"]) or "none"
                screen.append((f"  done: refused={str(d['refused']).lower()}  cited=[{cited}]  request_id={d['request_id'][:12]}…  ({ev['t']:.1f} s)", DIM))
                screen.append(("", FG))
                frames.append(frame(screen, f))
                durations.append(2200)
    out = Path("docs/media")
    out.mkdir(parents=True, exist_ok=True)
    frames[0].save(out / "demo.gif", save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    (out / "demo.json").write_text(json.dumps(log, indent=2), encoding="utf-8", newline="\n")
    print(f"wrote {out / 'demo.gif'} ({len(frames)} frames) and the raw events in {out / 'demo.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
