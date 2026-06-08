import tempfile
import unittest
import wave
from pathlib import Path

from scripts.prepare_aihub_emotion_dataset import prepare_dataset


class EmotionDatasetTest(unittest.TestCase):
    def write_config(self, root: Path) -> Path:
        config_path = root / "emotion_tags.yaml"
        config_path.write_text(
            """
canonical_emotions:
  happy:
    tag: "(happy)"
    aliases:
      - happy
      - 기쁨
  sad:
    tag: "(sad)"
    aliases:
      - sad
      - 슬픔
default_emotion: happy
""".strip()
            + "\n",
            encoding="utf-8",
        )
        return config_path

    def test_prepare_raw_audio_and_lab(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.write_config(root)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()

            raw_path = input_dir / "acriil_기쁨_0001.raw"
            raw_path.write_bytes(b"\x00\x00" * 160)
            raw_path.with_suffix(".txt").write_text("오늘은 좋은 날입니다.", encoding="utf-8")

            counts = prepare_dataset(
                input_dir=input_dir,
                output_dir=output_dir,
                config_path=config,
                samples_per_emotion=1,
                seed=1,
                emotions=["happy"],
                overwrite=True,
            )

            self.assertEqual(counts, {"happy": 1})

            wav_path = output_dir / "happy" / "happy_0001.wav"
            lab_path = output_dir / "happy" / "happy_0001.lab"
            self.assertTrue(wav_path.exists())
            self.assertTrue(lab_path.exists())
            self.assertEqual(
                lab_path.read_text(encoding="utf-8").strip(),
                "(happy) 오늘은 좋은 날입니다.",
            )

            with wave.open(str(wav_path), "rb") as wav_file:
                self.assertEqual(wav_file.getnchannels(), 1)
                self.assertEqual(wav_file.getsampwidth(), 2)
                self.assertEqual(wav_file.getframerate(), 16000)

    def test_unknown_emotion_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.write_config(root)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()

            (input_dir / "acriil_분노_0001.raw").write_bytes(b"\x00\x00" * 160)
            (input_dir / "acriil_분노_0001.txt").write_text("화가 납니다.", encoding="utf-8")

            with self.assertRaises(RuntimeError):
                prepare_dataset(
                    input_dir=input_dir,
                    output_dir=output_dir,
                    config_path=config,
                    samples_per_emotion=1,
                    seed=1,
                    emotions=["happy"],
                    overwrite=True,
                )

if __name__ == "__main__":
    unittest.main()
