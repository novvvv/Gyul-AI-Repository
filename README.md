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

## Setup

```bash
bash scripts/setup_fish_speech.sh
hf auth login
bash scripts/download_model.sh
```

## Inference

```bash
python scripts/run_inference.py \
  --text "안녕하세요. 음성 합성 테스트입니다." \
  --output sample.wav
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
