"""Builds the 2-minute demo video: neural TTS narration (edge-tts) + a live browser recording (Playwright)
of the running app (new messages typed live) and of the slides, assembled with ffmpeg.

    pip install edge-tts playwright imageio-ffmpeg mutagen && python -m playwright install chromium
    python app.py                       # in another terminal: the app must be running on :7860
    python presentation/make_video.py   # -> presentation/demo.mp4
"""
import asyncio
import subprocess
import time
from pathlib import Path

import edge_tts
import imageio_ffmpeg
from mutagen.mp3 import MP3
from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
OUT = HERE / "video_build"
OUT.mkdir(exist_ok=True)
APP = "http://127.0.0.1:7860"
SLIDES = (HERE / "slides.html").as_uri()
VOICE, RATE = "en-US-AndrewNeural", "+4%"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
W, H = 1280, 720

# (narration, actions) - actions run while the narration of that segment plays
SEGMENTS = [
    ("Support centers receive thousands of messages a day. My application reads each one, routes it instantly "
     "to the right team, and sends unknown requests to a human agent.",
     [("slide", 1)]),
    ("Here is the live application. I type a new message about a money transfer. The quantized, distilled BERT "
     "model routes it to the Banking team with high confidence, and the table below compares all four models, "
     "with their latency.",
     [("app",), ("type", "how do i transfer money to my sister's account")]),
    ("A harder case, with typos and a product code. The regex extracts the model code, edit distance fixes "
     "credit and declined, and the ticket goes to Card Services.",
     [("type", "my credt card got declnied at the store, model DW-6534")]),
    ("An out-of-scope question goes to a human agent. And a difficult case: appliance repair is not in the "
     "training data, so the dishwasher message also goes to a human, instead of a wrong team.",
     [("type", "how many moons does jupiter have"), ("wait", 4.5),
      ("type", "my dishwasher DW-6534 is not draining water")]),
    ("Small talk gets an ELIZA-style rule-based reply.",
     [("type", "hellooo")]),
    ("The architecture: regex normalization, spelling correction, ELIZA rules, then the INT8 BERT-small model on "
     "ONNX Runtime. The tf-idf baseline, the GloVe feed-forward network and the BERT-base teacher run for "
     "comparison. Everything runs on CPU.",
     [("slide", 4)]),
    ("On the same test set of 5,500 messages, BERT-base reaches 90.7 percent accuracy. The distilled INT8 model "
     "reaches 89.2 percent, while being 15 times smaller and 19 times faster, at 1.7 milliseconds per message.",
     [("slide", 7)]),
    ("In conclusion, distillation plus quantization is the best trade-off for a CPU-only router. Out-of-scope "
     "detection remains the hardest part. Thank you.",
     [("slide", 9)]),
]
GAP = 0.5  # seconds of silence after each segment


async def tts_all():
    for i, (text, _) in enumerate(SEGMENTS):
        f = OUT / f"seg{i}.mp3"
        if f.exists() and f.stat().st_size > 0:
            continue
        for attempt in range(8):           # the free endpoint sometimes returns no audio: retry
            try:
                await edge_tts.Communicate(text, VOICE, rate=RATE).save(str(f))
                break
            except edge_tts.exceptions.NoAudioReceived:
                await asyncio.sleep(3 + 2 * attempt)
        else:
            raise RuntimeError(f"TTS failed for segment {i}")


def run(page, action):
    kind = action[0]
    if kind == "slide":
        if not page.url.startswith("file:"):
            page.goto(SLIDES)
        page.evaluate(f"window.scrollTo(0, {(action[1] - 1) * H})")
    elif kind == "app":
        page.goto(APP)
        page.wait_for_selector("textarea, input[type=text]")
        page.wait_for_timeout(600)
    elif kind == "type":
        box = page.locator("textarea, input[type=text]").first
        box.click()
        box.fill("")
        box.press_sequentially(action[1], delay=45)
        page.wait_for_timeout(250)
        box.press("Enter")
        page.wait_for_timeout(1200)
        page.mouse.wheel(0, 0)
    elif kind == "wait":
        page.wait_for_timeout(int(action[1] * 1000))


def record(durations):
    starts = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": W, "height": H}, record_video_dir=str(OUT),
                                  record_video_size={"width": W, "height": H}, color_scheme="dark")
        page = ctx.new_page()
        t0 = time.monotonic()
        for (text, actions), dur in zip(SEGMENTS, durations):
            start = time.monotonic() - t0
            starts.append(start)
            for a in actions:
                run(page, a)
            left = dur + GAP - (time.monotonic() - t0 - start)
            if left > 0:
                page.wait_for_timeout(int(left * 1000))
        total = time.monotonic() - t0 + 0.5
        page.wait_for_timeout(500)
        video = Path(page.video.path())
        ctx.close()
        browser.close()
    return starts, total, video


def assemble(starts, total, video):
    inputs, filters = [], []
    for i, s in enumerate(starts):
        inputs += ["-i", str(OUT / f"seg{i}.mp3")]
        filters.append(f"[{i + 1}:a]adelay={int(s * 1000)}|{int(s * 1000)}[a{i}]")
    mix = "".join(f"[a{i}]" for i in range(len(starts)))
    filters.append(f"{mix}amix=inputs={len(starts)}:normalize=0[aout]")
    out = HERE / "demo.mp4"
    subprocess.run([FFMPEG, "-y", "-i", str(video), *inputs, "-filter_complex", ";".join(filters),
                    "-map", "0:v", "-map", "[aout]", "-t", f"{total:.2f}", "-c:v", "libx264", "-preset", "medium",
                    "-crf", "20", "-pix_fmt", "yuv420p", "-r", "25", "-c:a", "aac", "-b:a", "160k", str(out)],
                   check=True, capture_output=True)
    print(f"wrote {out} ({total:.1f} s)")


if __name__ == "__main__":
    asyncio.run(tts_all())
    durs = [MP3(OUT / f"seg{i}.mp3").info.length for i in range(len(SEGMENTS))]
    print("narration:", [round(d, 1) for d in durs], "total incl. gaps", round(sum(durs) + GAP * len(durs), 1), "s")
    starts, total, video = record(durs)
    assemble(starts, total, video)
