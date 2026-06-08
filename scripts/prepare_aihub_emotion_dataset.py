import argparse
import csv
import json
import random
import re
import shutil
import wave
from dataclasses import dataclass
from pathlib import Path

import yaml


AUDIO_EXTENSIONS = {".raw", ".wav", ".flac", ".mp3"}
TEXT_EXTENSIONS = {".txt", ".lab"}
TEXT_KEYS = (
    "text",
    "transcript",
    "transcription",
    "sentence",
    "utterance",
    "origin_text",
    "script",
)
FILE_KEYS = ("file", "filename", "file_name", "path", "audio", "raw")
EMOTION_KEYS = ("emotion", "label", "감정")
ID_KEYS = ("id", "idx", "index", "no", "number", "문장번호")


@dataclass(frozen=True)
class EmotionSpec:
    name: str
    tag: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class AudioRecord:
    source_audio: Path
    source_text: str
    emotion: str


def normalize_key(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z가-힣]+", "", value).lower()


def read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return path.read_text(encoding=encoding).strip()
        except UnicodeDecodeError:
            continue

    return path.read_text(errors="ignore").strip()


def load_emotion_specs(config_path: Path) -> dict[str, EmotionSpec]:
    config = yaml.safe_load(read_text(config_path))
    specs = {}

    for name, item in config["canonical_emotions"].items():
        aliases = tuple(item.get("aliases", []))
        specs[name] = EmotionSpec(name=name, tag=item["tag"], aliases=aliases)

    return specs


def build_alias_map(specs: dict[str, EmotionSpec]) -> dict[str, str]:
    alias_map = {}

    for emotion, spec in specs.items():
        alias_map[normalize_key(emotion)] = emotion
        for alias in spec.aliases:
            alias_map[normalize_key(alias)] = emotion

    return dict(sorted(alias_map.items(), key=lambda item: len(item[0]), reverse=True))


def parse_emotion_from_name(path: Path, alias_map: dict[str, str]) -> str | None:
    normalized = normalize_key(path.stem)

    for alias, canonical in alias_map.items():
        if alias and alias in normalized:
            return canonical

    return None


def parse_record_id(path: Path) -> str | None:
    matches = re.findall(r"\d+", path.stem)
    return matches[-1] if matches else None


def read_sidecar_text(audio_path: Path) -> str | None:
    for extension in TEXT_EXTENSIONS:
        candidate = audio_path.with_suffix(extension)
        if candidate.exists():
            return read_text(candidate)

    return None


def collect_text_sidecars(input_dir: Path) -> dict[str, str]:
    index = {}

    for path in input_dir.rglob("*"):
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            continue

        text = read_text(path)
        if not text:
            continue

        index[normalize_key(path.stem)] = text
        record_id = parse_record_id(path)
        if record_id:
            index.setdefault(record_id, text)

        for line in text.splitlines():
            match = re.match(r"^\s*([^,\t| ]+)[,\t| ]+(.+?)\s*$", line)
            if not match:
                continue

            line_key, line_text = match.groups()
            line_id = parse_record_id(Path(line_key))
            if not line_id:
                continue

            index.setdefault(normalize_key(Path(line_key).stem), line_text)
            index.setdefault(line_id, line_text)

    return index


def iter_json_objects(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from iter_json_objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_json_objects(child)


def first_value(row: dict, keys: tuple[str, ...]) -> str | None:
    lowered = {str(key).lower(): value for key, value in row.items()}

    for key in keys:
        if key in row and row[key]:
            return str(row[key])

        lowered_key = key.lower()
        if lowered_key in lowered and lowered[lowered_key]:
            return str(lowered[lowered_key])

    return None


def add_metadata_row(index: dict[str, str], row: dict, alias_map: dict[str, str]) -> None:
    text = first_value(row, TEXT_KEYS)
    if not text:
        return

    filename = first_value(row, FILE_KEYS)
    record_id = first_value(row, ID_KEYS)
    emotion_value = first_value(row, EMOTION_KEYS)

    if filename:
        stem = Path(filename).stem
        index[normalize_key(stem)] = text.strip()

        filename_id = parse_record_id(Path(filename))
        if filename_id:
            index.setdefault(filename_id, text.strip())

    if record_id:
        index.setdefault(normalize_key(record_id), text.strip())

    if record_id and emotion_value:
        canonical = alias_map.get(normalize_key(emotion_value))
        if canonical:
            index.setdefault(f"{canonical}:{normalize_key(record_id)}", text.strip())


def collect_metadata_texts(input_dir: Path, alias_map: dict[str, str]) -> dict[str, str]:
    index = {}

    for path in input_dir.rglob("*"):
        suffix = path.suffix.lower()

        if suffix == ".json":
            try:
                data = json.loads(read_text(path))
            except json.JSONDecodeError:
                continue

            for row in iter_json_objects(data):
                add_metadata_row(index, row, alias_map)

        elif suffix == ".jsonl":
            for line in read_text(path).splitlines():
                if not line.strip():
                    continue

                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if isinstance(row, dict):
                    add_metadata_row(index, row, alias_map)

        elif suffix in {".csv", ".tsv"}:
            delimiter = "\t" if suffix == ".tsv" else ","
            for encoding in ("utf-8-sig", "utf-8", "cp949"):
                try:
                    with path.open(encoding=encoding, newline="") as fp:
                        for row in csv.DictReader(fp, delimiter=delimiter):
                            add_metadata_row(index, row, alias_map)
                    break
                except UnicodeDecodeError:
                    continue

    return index


def resolve_text(
    audio_path: Path,
    emotion: str,
    text_index: dict[str, str],
    metadata_index: dict[str, str],
) -> str | None:
    sidecar_text = read_sidecar_text(audio_path)
    if sidecar_text:
        return sidecar_text

    lookup_keys = [normalize_key(audio_path.stem)]
    record_id = parse_record_id(audio_path)

    if record_id:
        lookup_keys.extend([record_id, f"{emotion}:{record_id}"])

    for key in lookup_keys:
        if key in metadata_index:
            return metadata_index[key]
        if key in text_index:
            return text_index[key]

    return None


def discover_records(
    input_dir: Path,
    alias_map: dict[str, str],
    emotions: set[str],
) -> tuple[dict[str, list[AudioRecord]], list[Path], list[Path]]:
    text_index = collect_text_sidecars(input_dir)
    metadata_index = collect_metadata_texts(input_dir, alias_map)
    grouped_records = {emotion: [] for emotion in emotions}
    unmapped_audio = []
    missing_text = []

    for audio_path in sorted(input_dir.rglob("*")):
        if audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue

        emotion = parse_emotion_from_name(audio_path, alias_map)
        if emotion not in emotions:
            unmapped_audio.append(audio_path)
            continue

        text = resolve_text(audio_path, emotion, text_index, metadata_index)
        if not text:
            missing_text.append(audio_path)
            continue

        grouped_records[emotion].append(
            AudioRecord(source_audio=audio_path, source_text=text, emotion=emotion)
        )

    return grouped_records, unmapped_audio, missing_text


def convert_raw_to_wav(source: Path, destination: Path) -> None:
    raw_bytes = source.read_bytes()

    with wave.open(str(destination), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(raw_bytes)


def write_audio(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if source.suffix.lower() == ".raw":
        convert_raw_to_wav(source, destination)
        return destination
    elif source.suffix.lower() == ".wav":
        shutil.copy2(source, destination)
        return destination
    else:
        output_path = destination.with_suffix(source.suffix.lower())
        shutil.copy2(source, output_path)
        return output_path


def write_manifest(manifest_path: Path, rows: list[dict[str, str]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "emotion",
                "audio",
                "lab",
                "source_audio",
                "text",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def prepare_dataset(
    input_dir: Path,
    output_dir: Path,
    config_path: Path,
    samples_per_emotion: int,
    seed: int,
    emotions: list[str] | None,
    overwrite: bool,
) -> dict[str, int]:
    specs = load_emotion_specs(config_path)
    alias_map = build_alias_map(specs)
    target_emotions = set(emotions or specs.keys())
    unknown_targets = target_emotions - set(specs.keys())

    if unknown_targets:
        raise ValueError(f"Unknown target emotions: {sorted(unknown_targets)}")

    grouped_records, unmapped_audio, missing_text = discover_records(
        input_dir=input_dir,
        alias_map=alias_map,
        emotions=target_emotions,
    )

    missing_emotions = [
        emotion for emotion, records in grouped_records.items() if not records
    ]
    if missing_emotions:
        raise RuntimeError(
            "No usable audio/text records found for emotions: "
            f"{', '.join(sorted(missing_emotions))}"
        )

    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)

    rng = random.Random(seed)
    manifest_rows = []
    counts = {}

    for emotion in sorted(target_emotions):
        records = grouped_records[emotion]
        rng.shuffle(records)

        selected = (
            records
            if samples_per_emotion <= 0
            else records[: min(samples_per_emotion, len(records))]
        )
        counts[emotion] = len(selected)

        for idx, record in enumerate(selected, start=1):
            stem = f"{emotion}_{idx:04d}"
            audio_output = output_dir / emotion / f"{stem}.wav"
            lab_output = output_dir / emotion / f"{stem}.lab"
            tagged_text = f"{specs[emotion].tag} {record.source_text.strip()}"

            actual_audio_output = write_audio(record.source_audio, audio_output)
            lab_output.write_text(tagged_text + "\n", encoding="utf-8")

            manifest_rows.append(
                {
                    "emotion": emotion,
                    "audio": str(actual_audio_output),
                    "lab": str(lab_output),
                    "source_audio": str(record.source_audio),
                    "text": tagged_text,
                }
            )

    write_manifest(output_dir / "manifest.csv", manifest_rows)

    report = {
        "counts": counts,
        "unmapped_audio_count": len(unmapped_audio),
        "missing_text_count": len(missing_text),
    }
    (output_dir / "prepare_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return counts


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert AI Hub emotional TTS data into Fish Speech fine-tuning format."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/emotion_tags.yaml"),
        help="Path to emotion tag mapping YAML.",
    )
    parser.add_argument("--samples-per-emotion", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--emotions",
        nargs="*",
        default=None,
        help="Canonical emotions to export. Defaults to all configured emotions.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    counts = prepare_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        samples_per_emotion=args.samples_per_emotion,
        seed=args.seed,
        emotions=args.emotions,
        overwrite=args.overwrite,
    )

    print("Prepared Fish Speech dataset:")
    for emotion, count in sorted(counts.items()):
        print(f"- {emotion}: {count}")


if __name__ == "__main__":
    main()
