# Emotion-responsive TTS Fine-tuning Plan

This branch is limited to model fine-tuning work for the capstone topic:
emotion-responsive TTS model development.

Runtime integration with the main FastAPI branch is intentionally out of scope here.
This branch should produce a trained checkpoint and evaluation samples that can be used by later integration work.

## Goal

```text
AI Hub emotional speech synthesis data
-> Fish Speech S1-mini LoRA fine-tuning on Colab Pro
-> merged fine-tuned checkpoint
-> emotion-labeled generation samples for evaluation
```

## Model and Data

- Dataset: AI Hub `dataSetSn=286`, emotional speech synthesis dataset.
- Base checkpoint: `fishaudio/openaudio-s1-mini`.
- Fine-tuning method: Fish Speech `text2semantic_finetune`.
- LoRA config: `r_8_alpha_16`.
- Emotion tags: S1-mini style markers such as `(happy)`, `(sad)`, `(angry)`.

The first training run uses 300 samples per emotion. Increase the sample count only after the smoke run and first full run succeed.

## Colab Pro Storage

Use the Google account that owns Colab Pro. Store all large artifacts in that account's Drive:

```text
/content/drive/MyDrive/gyul-ai/emotion-tts/
  raw/
  processed/
  protos/
  checkpoints/
  results/
  generated_samples/
```

The repository must not store datasets, protobuf shards, checkpoints, LoRA weights, or generated wav files.

## Training Notebook

Run:

```text
notebooks/colab_aihub_emotion_finetune.ipynb
```

The notebook performs:

```text
Drive mount
-> repository clone
-> Fish Speech install
-> openaudio-s1-mini download
-> AI Hub conversion to wav/lab
-> VQ token extraction
-> protobuf packing
-> LoRA fine-tuning
-> LoRA merge
-> evaluation sample generation
```

## Dataset Conversion

Use:

```bash
python scripts/prepare_aihub_emotion_dataset.py \
  --input-dir "/content/drive/MyDrive/gyul-ai/emotion-tts/raw" \
  --output-dir "/content/drive/MyDrive/gyul-ai/emotion-tts/processed" \
  --samples-per-emotion 300 \
  --overwrite
```

The converter creates Fish Speech fine-tuning files:

```text
processed/
  happy/
    happy_0001.wav
    happy_0001.lab
```

Each `.lab` contains an emotion marker plus transcription:

```text
(happy) text to speak
```

## Training Defaults

- VQ extraction batch size:
  - T4: 4
  - L4/A100: 16
- Training max steps: 1500
- Training batch size: 2
- Gradient accumulation: 8
- Precision: `bf16-true`; use `16-mixed` if the runtime does not support bf16

## Acceptance Criteria

- Smoke run:
  - 2 samples per emotion pass conversion, VQ extraction, protobuf build
  - 20 LoRA training steps complete
- First full run:
  - 300 samples per emotion are converted
  - LoRA training reaches the configured max steps
  - LoRA checkpoint is merged into a regular checkpoint
  - At least 5 generated wav samples per emotion are saved under Drive
- Evaluation artifacts:
  - generated sample paths
  - training command/config values
  - selected checkpoint path
  - brief before/after comparison notes

## Out of Scope

- main branch FastAPI integration
- local inference service implementation
- WebSocket response schema changes
- real-time demo UI behavior

Those will be handled in a later integration branch after the fine-tuned checkpoint is produced.
