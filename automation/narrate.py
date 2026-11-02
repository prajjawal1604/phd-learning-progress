import argparse
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import local

import numpy as np
import soundfile as sf
import torch
from kokoro import KPipeline


WORKERS = 2
BATCH_CHARS = 1400
SAMPLE_RATE = 24000
PAUSE_SAMPLES = 6000


def clean_markdown(text):
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"(?m)^#{1,6}\s+", "", text)
    text = re.sub(r"[\x60*_]", "", text)
    replacements = {"LLM": "L L M", "RAG": "rag", "API": "A P I", "CIA triad": "C I A triad"}
    for key, value in replacements.items():
        text = text.replace(key, value)
    return text.strip()


def split_batches(text, max_chars=BATCH_CHARS):
    """Group whole paragraphs so independent batches can be synthesized in parallel."""
    batches = []
    paragraphs = []
    batch_size = 0

    for paragraph in re.split(r"\n+", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        added_size = len(paragraph) + (1 if paragraphs else 0)
        if paragraphs and batch_size + added_size > max_chars:
            batches.append("\n".join(paragraphs))
            paragraphs = []
            batch_size = 0
            added_size = len(paragraph)
        paragraphs.append(paragraph)
        batch_size += added_size

    if paragraphs:
        batches.append("\n".join(paragraphs))
    return batches


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

    # Two inference workers on the hosted runner avoid CPU oversubscription.
    torch.set_num_threads(1)
    pipeline = KPipeline(lang_code="a")
    model = pipeline.model
    voice_pack = pipeline.load_voice(args.voice)
    batches = split_batches(text)
    worker_state = local()

    def synthesize(batch):
        # Each thread owns its text processor; all workers share the read-only model.
        if not hasattr(worker_state, "pipeline"):
            worker_state.pipeline = KPipeline(lang_code="a", model=model)

        pieces = []
        for result in worker_state.pipeline(
            batch, voice=voice_pack, speed=0.95, split_pattern=r"\n+"
        ):
            if result.audio is not None and len(result.audio):
                pieces.append(np.asarray(result.audio, dtype=np.float32))
                pieces.append(np.zeros(PAUSE_SAMPLES, dtype=np.float32))
        if not pieces:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(pieces)

    pieces = []
    with ThreadPoolExecutor(max_workers=min(WORKERS, len(batches))) as executor:
        for audio in executor.map(synthesize, batches):
            if len(audio):
                pieces.append(audio)

    if not pieces:
        raise RuntimeError("No audio produced by Kokoro")
    output = source.with_suffix(".mp3")
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "narration.wav"
        sf.write(wav, np.concatenate(pieces), SAMPLE_RATE)
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                        "-i", str(wav), "-codec:a", "libmp3lame", "-q:a", "3",
                        str(output)], check=True)
    if output.stat().st_size < 1000:
        raise RuntimeError("Output MP3 unexpectedly small")
    print(f"Created {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
