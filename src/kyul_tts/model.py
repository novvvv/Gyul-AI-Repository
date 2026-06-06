from pathlib import Path
import subprocess
import shutil


class KyulTTSModel:
    """
    Fish Speech S2-Pro 기반 zero-shot Korean TTS wrapper.

    Pipeline:
    1. reference audio -> prompt token(fake.npy)
    2. text + prompt token -> semantic token(codes_0.npy)
    3. semantic token -> waveform(fake.wav)
    """

    def __init__(
        self,
        fish_speech_dir: str,
        checkpoint_dir: str = "checkpoints/s2-pro",
        reference_audio: str = "assets/reference/kyul_ref.wav",
        reference_text: str = "assets/reference/kyul_ref.txt",
        output_dir: str = "outputs",
        use_half: bool = True,
    ):
        self.fish_speech_dir = Path(fish_speech_dir).resolve()
        self.checkpoint_dir = checkpoint_dir
        self.reference_audio = Path(reference_audio).resolve()
        self.reference_text = Path(reference_text).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.use_half = use_half

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.codec_path = self.fish_speech_dir / self.checkpoint_dir / "codec.pth"

        self._validate_paths()

    def _validate_paths(self):
        if not self.fish_speech_dir.exists():
            raise FileNotFoundError(f"Fish Speech directory not found: {self.fish_speech_dir}")

        if not self.codec_path.exists():
            raise FileNotFoundError(f"codec.pth not found: {self.codec_path}")

        if not self.reference_audio.exists():
            raise FileNotFoundError(f"Reference audio not found: {self.reference_audio}")

        if not self.reference_text.exists():
            raise FileNotFoundError(f"Reference text not found: {self.reference_text}")

    def _run(self, command: str):
        print(f"\n$ {command}")

        result = subprocess.run(
            command,
            shell=True,
            cwd=str(self.fish_speech_dir),
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Command failed: {command}")

    def build_prompt_tokens(self):
        command = (
            "python fish_speech/models/dac/inference.py "
            f"-i '{self.reference_audio}' "
            f"--checkpoint-path '{self.checkpoint_dir}/codec.pth'"
        )

        self._run(command)

        fake_npy = self.fish_speech_dir / "fake.npy"

        if not fake_npy.exists():
            raise FileNotFoundError("fake.npy was not generated.")

        return fake_npy

    def text_to_semantic(self, text: str):
        prompt_text = self.reference_text.read_text(encoding="utf-8").strip()
        half_flag = "--half" if self.use_half else ""

        command = (
            "rm -f codes_*.npy && rm -rf output && "
            "python fish_speech/models/text2semantic/inference.py "
            f"--text \"{text}\" "
            f"--prompt-text \"{prompt_text}\" "
            "--prompt-tokens 'fake.npy' "
            f"--checkpoint-path '{self.checkpoint_dir}' "
            f"{half_flag}"
        )

        self._run(command)

        codes_path = self.fish_speech_dir / "output" / "codes_0.npy"

        if codes_path.exists():
            return codes_path

        alt_path = self.fish_speech_dir / "codes_0.npy"

        if alt_path.exists():
            return alt_path

        raise FileNotFoundError("codes_0.npy was not generated.")

    def semantic_to_wav(self, codes_path: Path):
        command = (
            "python fish_speech/models/dac/inference.py "
            f"-i '{codes_path}' "
            f"--checkpoint-path '{self.checkpoint_dir}/codec.pth'"
        )

        self._run(command)

        wav_path = self.fish_speech_dir / "fake.wav"

        if not wav_path.exists():
            raise FileNotFoundError("fake.wav was not generated.")

        return wav_path

    def generate(self, text: str, output_name: str = "generated.wav"):
        self.build_prompt_tokens()
        codes_path = self.text_to_semantic(text)
        wav_path = self.semantic_to_wav(codes_path)

        final_path = self.output_dir / output_name
        shutil.copy2(wav_path, final_path)

        print(f"\nGenerated audio saved to: {final_path}")

        return final_path
