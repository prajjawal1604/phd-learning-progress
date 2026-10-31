import argparse
import re
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("lesson")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = (root / args.lesson).resolve()
    if not source.is_relative_to(root) or source.suffix != ".md" or not source.is_file():
        raise ValueError("Expected a Markdown lesson inside the repository")

    text = source.read_text(encoding="utf-8")
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[#*_\x60>]", "", text).strip()
    if not text:
        raise ValueError("Lesson is empty")

    wav = source.with_suffix(".wav")
    output = source.with_suffix(".mp3")
    try:
        subprocess.run(["espeak-ng", "-v", "en-us", "-s", "150", "-w", str(wav), text], check=True)
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav), "-codec:a", "libmp3lame", "-qscale:a", "4", str(output)], check=True)
    finally:
        wav.unlink(missing_ok=True)
    if not output.is_file() or output.stat().st_size < 1000:
        raise RuntimeError("Narration file is missing or unexpectedly small")
    print(f"Created {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
