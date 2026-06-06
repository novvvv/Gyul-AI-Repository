import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from kyul_tts.model import KyulTTSModel


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--output", type=str, default="generated.wav")
    parser.add_argument("--fish-speech-dir", type=str, default="/content/fish-speech-s2")

    args = parser.parse_args()

    model = KyulTTSModel(
        fish_speech_dir=args.fish_speech_dir,
        checkpoint_dir="checkpoints/s2-pro",
        reference_audio=str(PROJECT_ROOT / "assets" / "reference" / "kyul_ref.wav"),
        reference_text=str(PROJECT_ROOT / "assets" / "reference" / "kyul_ref.txt"),
        output_dir=str(PROJECT_ROOT / "outputs"),
        use_half=True,
    )

    model.generate(args.text, args.output)


if __name__ == "__main__":
    main()
