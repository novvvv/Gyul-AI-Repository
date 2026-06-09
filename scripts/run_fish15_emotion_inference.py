import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "emotion_tags.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "fish15_emotion"
DECODER_FILENAME = "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"


def load_emotion_tags(config_path: Path) -> tuple[dict[str, str], str]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    tags = {
        name: item["tag"]
        for name, item in config["canonical_emotions"].items()
    }
    alias_map = {}

    for name, item in config["canonical_emotions"].items():
        alias_map[name.lower()] = name
        for alias in item.get("aliases", []):
            alias_map[str(alias).lower()] = name

    default_emotion = config.get("default_emotion", "neutral")
    return {alias: tags[name] for alias, name in alias_map.items()}, default_emotion


def run_checked(command: list[str], cwd: Path) -> None:
    print("$", " ".join(command))
    try:
        subprocess.run(command, cwd=cwd, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate short emotion-conditioned wav with fine-tuned Fish Speech 1.5."
    )
    parser.add_argument("--text", required=True, help="Text to synthesize.")
    parser.add_argument(
        "--emotion",
        default=None,
        help="Emotion name or alias. Examples: neutral, happy, sad, angry, anxious, hurt, embarrassed.",
    )
    parser.add_argument(
        "--fish-speech-dir",
        required=True,
        type=Path,
        help="Path to fish-speech v1.5.1 checkout.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        type=Path,
        help="Merged fine-tuned checkpoint dir containing model.pth, config.json, tokenizer.tiktoken, special_tokens.json.",
    )
    parser.add_argument(
        "--decoder-checkpoint",
        type=Path,
        default=None,
        help="Path to Fish Speech 1.5 VQGAN decoder checkpoint. Defaults to <base-checkpoint-dir>/firefly-gan-vq-fsq-8x1024-21hz-generator.pth.",
    )
    parser.add_argument(
        "--base-checkpoint-dir",
        type=Path,
        default=None,
        help="Base fish-speech-1.5 checkpoint dir used to find the VQGAN decoder when --decoder-checkpoint is omitted.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output wav path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--half", action="store_true", help="Use fp16 text2semantic inference.")
    parser.add_argument(
        "--keep-semantic",
        action="store_true",
        help="Keep generated codes_0.npy next to the output wav.",
    )

    args = parser.parse_args()

    fish_dir = args.fish_speech_dir.resolve()
    checkpoint_dir = args.checkpoint_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_path = (args.output or output_dir / "generated.wav").resolve()
    semantic_dir = output_path.parent / f"{output_path.stem}_semantic"

    if args.decoder_checkpoint:
        decoder_checkpoint = args.decoder_checkpoint.resolve()
    elif args.base_checkpoint_dir:
        decoder_checkpoint = args.base_checkpoint_dir.resolve() / DECODER_FILENAME
    else:
        decoder_checkpoint = checkpoint_dir / DECODER_FILENAME

    require_file(fish_dir / "fish_speech" / "models" / "text2semantic" / "inference.py", "text2semantic inference script")
    require_file(fish_dir / "fish_speech" / "models" / "vqgan" / "inference.py", "vqgan inference script")
    require_file(checkpoint_dir / "model.pth", "merged model")
    require_file(checkpoint_dir / "config.json", "merged config")
    require_file(checkpoint_dir / "tokenizer.tiktoken", "merged tokenizer")
    require_file(decoder_checkpoint, "decoder checkpoint")

    emotion_tags, default_emotion = load_emotion_tags(args.config)
    emotion = (args.emotion or default_emotion).lower()
    if emotion not in emotion_tags:
        known = ", ".join(sorted(emotion_tags))
        raise ValueError(f"Unknown emotion '{args.emotion}'. Known emotions/aliases: {known}")

    tagged_text = f"{emotion_tags[emotion]} {args.text.strip()}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    semantic_dir.mkdir(parents=True, exist_ok=True)
    for old_code in semantic_dir.glob("codes_*.npy"):
        old_code.unlink()

    text2semantic_cmd = [
        sys.executable,
        "fish_speech/models/text2semantic/inference.py",
        "--text",
        tagged_text,
        "--checkpoint-path",
        str(checkpoint_dir),
        "--num-samples",
        "1",
        "--output-dir",
        str(semantic_dir),
        "--device",
        args.device,
    ]
    if args.half:
        text2semantic_cmd.append("--half")

    print("tagged_text:", tagged_text)
    run_checked(text2semantic_cmd, cwd=fish_dir)

    codes = sorted(semantic_dir.glob("codes_*.npy"))
    if not codes:
        raise FileNotFoundError(f"No semantic token generated in {semantic_dir}")

    vqgan_cmd = [
        sys.executable,
        "fish_speech/models/vqgan/inference.py",
        "-i",
        str(codes[0]),
        "-o",
        str(output_path),
        "--config-name",
        "firefly_gan_vq",
        "--checkpoint-path",
        str(decoder_checkpoint),
        "--device",
        args.device,
    ]
    run_checked(vqgan_cmd, cwd=fish_dir)

    if not args.keep_semantic:
        shutil.rmtree(semantic_dir, ignore_errors=True)

    print("saved:", output_path)


if __name__ == "__main__":
    main()
