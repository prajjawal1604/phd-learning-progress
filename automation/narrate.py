import argparse
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from kokoro import KPipeline


def clean_markdown(text):
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"(?m)^#{1,6}\s+", "", text)
    text = re.sub(r"[\x60*_]", "", text)
    replacements = {"LLM": "L L M", "RAG": "rag", "API": "A P I", "CIA triad": "C I A triad"}
    for key, value in replacements.items():
        text = text.replace(key, value)
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("lesson")
    parser.add_argument("--voice", default="am_michael")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = (root / args.lesson).resolve()
    if not source.is_relative_to(root) or source.suffix != ".md" or not source.is_file():
        raise ValueError("Expected a Markdown lesson inside the repository")
    text = clean_markdown(source.read_text(encoding="utf-8"))
    if not text:
        raise ValueError("Lesson is empty")
    pipeline = KPipeline(lang_code="a")
    pieces = []
    for _, _, audio in pipeline(text, voice=args.voice, speed=0.95, split_pattern=r"\n+"):
        if len(audio):
            pieces.append(np.asarray(audio, dtype=np.float32))
            pieces.append(np.zeros(6000, dtype=np.float32))
    if not pieces:
        raise RuntimeError("No audio produced by Kokoro")
    output = source.with_suffix(".mp3")
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "narration.wav"
        sf.write(wav, np.concatenate(pieces), 24000)
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                        "-i", str(wav), "-codec:a", "libmp3lame", "-q:a", "3",
                        str(output)], check=True)
    if output.stat().st_size < 1000:
        raise RuntimeError("Output MP3 unexpectedly small")
    print(f"Created {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
