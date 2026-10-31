import asyncio
import argparse
from pathlib import Path
import edge_tts

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("lesson")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = (root / args.lesson).resolve()
    if not source.is_relative_to(root) or source.suffix != ".md" or not source.is_file():
        raise ValueError("Expected a Markdown lesson inside the repository")
    output = source.with_suffix(".mp3")
    await edge_tts.Communicate(source.read_text(encoding="utf-8"), "en-US-AriaNeural").save(str(output))
    if output.stat().st_size < 1000:
        raise RuntimeError("Narration file is unexpectedly small")
    print(f"Created {output} ({output.stat().st_size} bytes)")

if __name__ == "__main__":
    asyncio.run(main())
