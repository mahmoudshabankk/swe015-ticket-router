"""Builds the 2-minute demo video: neural TTS narration (edge-tts) + a live browser recording (Playwright)
of the running app (new messages typed live) and of the slides, assembled with ffmpeg.

    pip install edge-tts playwright imageio-ffmpeg mutagen && python -m playwright install chromium
    python app.py                       # in another terminal: the app must be running on :7860
    python presentation/make_video.py   # -> presentation/demo.mp4
"""
import asyncio
import shutil
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
SCALE = 1.5          # -> 1920x1080 output
OW, OH = int(W * SCALE), int(H * SCALE)

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
    """Drives the browser and captures lossless-quality frames with the Chrome DevTools screencast
    (1920x1080: 1280x720 CSS viewport at device scale factor 1.5)."""
    import base64
    frames_dir = OUT / "frames"
    shutil.rmtree(frames_dir, ignore_errors=True)
    frames_dir.mkdir()
    frames, starts = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": W, "height": H}, device_scale_factor=SCALE,
                                  color_scheme="dark")
        page = ctx.new_page()
        cdp = ctx.new_cdp_session(page)

        def on_frame(ev):
            f = frames_dir / f"{len(frames):05d}.jpg"
            f.write_bytes(base64.b64decode(ev["data"]))
            frames.append((ev["metadata"]["timestamp"], f))
            cdp.send("Page.screencastFrameAck", {"sessionId": ev["sessionId"]})

        cdp.on("Page.screencastFrame", on_frame)
        page.goto(SLIDES)
        page.wait_for_timeout(300)
        cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 95, "maxWidth": OW,
                                          "maxHeight": OH, "everyNthFrame": 1})
        page.wait_for_timeout(300)
        t0 = time.time()
        m0 = time.monotonic()
        for (text, actions), dur in zip(SEGMENTS, durations):
            start = time.monotonic() - m0
            starts.append(start)
            for a in actions:
                run(page, a)
            left = dur + GAP - (time.monotonic() - m0 - start)
            if left > 0:
                page.wait_for_timeout(int(left * 1000))
        page.wait_for_timeout(500)
        total = time.monotonic() - m0
        cdp.send("Page.stopScreencast")
        ctx.close()
        browser.close()
    # frame list for ffmpeg's concat demuxer: each frame lasts until the next one arrives
    frames.sort()
    times = [max(0.0, ts - t0) for ts, _ in frames]
    lines = []
    for i, (_, f) in enumerate(frames):
        if i == 0 and times[0] > 0:
            times[0] = 0.0
        end = times[i + 1] if i + 1 < len(frames) else total
        if end - times[i] <= 0:
            continue
        lines += [f"file '{f.as_posix()}'", f"duration {end - times[i]:.4f}"]
    lines.append(f"file '{frames[-1][1].as_posix()}'")
    (OUT / "frames.txt").write_text("\n".join(lines))
    print(f"captured {len(frames)} frames")
    return starts, total, OUT / "frames.txt"


def assemble(starts, total, frame_list):
    inputs, filters = [], []
    for i, s in enumerate(starts):
        inputs += ["-i", str(OUT / f"seg{i}.mp3")]
        filters.append(f"[{i + 1}:a]adelay={int(s * 1000)}|{int(s * 1000)}[a{i}]")
    mix = "".join(f"[a{i}]" for i in range(len(starts)))
    filters.append(f"{mix}amix=inputs={len(starts)}:normalize=0[aout]")
    filters.append(f"[0:v]scale={OW}:{OH}:flags=lanczos:in_range=pc:out_range=tv,fps=30,format=yuv420p[vout]")
    out = HERE / "demo.mp4"
    subprocess.run([FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", str(frame_list), *inputs,
                    "-filter_complex", ";".join(filters), "-map", "[vout]", "-map", "[aout]",
                    "-t", f"{total:.2f}", "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-tune", "stillimage",
                    "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", str(out)],
                   check=True, capture_output=True)
    print(f"wrote {out} ({total:.1f} s, {OW}x{OH})")


if __name__ == "__main__":
    asyncio.run(tts_all())
    durs = [MP3(OUT / f"seg{i}.mp3").info.length for i in range(len(SEGMENTS))]
    print("narration:", [round(d, 1) for d in durs], "total incl. gaps", round(sum(durs) + GAP * len(durs), 1), "s")
    starts, total, frame_list = record(durs)
    assemble(starts, total, frame_list)
