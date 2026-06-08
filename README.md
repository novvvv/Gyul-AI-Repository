# Kyul TTS

Fish Speech S2-Pro 기반 한국어 zero-shot voice cloning TTS 프로젝트입니다.

Reference voice를 이용해 입력 텍스트를 특정 화자 스타일의 음성으로 합성합니다.

## Model Type

이 프로젝트는 직접 fine-tuning한 checkpoint를 포함하지 않습니다.

사전학습된 Fish Speech S2-Pro 모델을 활용하고, reference audio를 prompt token으로 변환하여 음색을 반영하는 inference pipeline입니다.

## Pipeline

```text
reference wav
    ↓
prompt token(fake.npy)
    ↓
input text + prompt token
    ↓
semantic token(codes_0.npy)
    ↓
generated wav
```

## Features

- Korean text-to-speech
- Reference voice based voice cloning
- Fish Speech S2-Pro inference wrapper
- Prompt token generation
- Semantic token generation
- Waveform synthesis

## Setup

```bash
bash scripts/setup_fish_speech.sh
hf auth login
bash scripts/download_model.sh
```

---

# Kyul TTS

Fish Speech S2-Pro 기반 한국어 zero-shot voice cloning TTS 프로젝트입니다.

Reference voice를 이용해 입력 텍스트를 특정 화자 스타일의 음성으로 합성합니다.

## Model Type

이 프로젝트는 직접 fine-tuning한 checkpoint를 포함하지 않습니다.

사전학습된 Fish Speech S2-Pro 모델을 활용하고, reference audio를 prompt token으로 변환하여 음색을 반영하는 inference pipeline입니다.

## Pipeline

```text
reference wav
    ↓
prompt token(fake.npy)
    ↓
input text + prompt token
    ↓
semantic token(codes_0.npy)
    ↓
generated wav
```

## Features

- Korean text-to-speech
- Reference voice based voice cloning
- Fish Speech S2-Pro inference wrapper
- Prompt token generation
- Semantic token generation
- Waveform synthesis

## Google Colab에서 실행하기 (추천 ⭐)

macOS 로컬 환경보다 **Google Colab(GPU)** 환경에서 구동하는 것을 강력히 추천합니다. 자세한 사용법은 [Colab_Guide.md](file:///Users/nuyoes/Desktop/3-1/Gyul-AI-Repository/Colab_Guide.md) 파일을 참고하세요.

## 로컬 Setup (macOS 등)

로컬에서 가볍게 코드가 정상 작동하는지 검사하려면 다음 단위 테스트를 실행할 수 있습니다.
```bash
python -m unittest tests/test_model.py
```

만약 로컬 환경에서 직접 실물 인퍼런스를 구축하고 싶다면 아래 단계를 수행해야 합니다.

```bash
# 1. 의존성 패키지 설치
pip install --upgrade pip
pip install llvmlite numba
pip install -r requirements.txt

# 2. fish-speech 설치 및 빌드
git clone https://github.com/fishaudio/fish-speech.git fish-speech-s2
cd fish-speech-s2
pip install -e .
cd ..

# 3. huggingface 로그인 및 모델 다운로드
huggingface-cli login
mkdir -p fish-speech-s2/checkpoints/s2-pro
huggingface-cli download fishaudio/s2-pro --local-dir fish-speech-s2/checkpoints/s2-pro
```

## 로컬 Inference

로컬 디렉토리 경로를 전달하여 음성을 합성합니다.

```bash
python scripts/run_inference.py \
  --text "안녕하세요. 음성 합성 테스트입니다." \
  --output sample.wav \
  --fish-speech-dir ./fish-speech-s2
```

Generated files are saved in:

```text
outputs/
```

## Notes

Model checkpoints are not included in this repository.

Please download the model from Hugging Face using the provided script.

## Project Description

본 프로젝트는 Fish Speech S2-Pro 기반의 한국어 음성 합성 파이프라인입니다.
사전학습된 TTS 모델을 활용하고, reference audio를 prompt token으로 변환하여 특정 화자의 음색을 반영한 zero-shot voice cloning 방식으로 음성을 생성합니다.

데이터 전처리 단계에서는 직접 수집한 음성을 chunk 단위로 분할하고, Whisper 기반 STT를 통해 metadata를 생성했습니다.
최종적으로 사용자가 텍스트를 입력하면 reference token 생성, semantic token 생성, waveform 복원까지 하나의 모델처럼 실행되도록 wrapper를 구현했습니다.
