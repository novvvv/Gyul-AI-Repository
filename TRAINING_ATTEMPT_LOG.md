# Emotion TTS Fine-tuning Attempt Log

이 문서는 `MaTuna/tts` 브랜치에서 감정 반응형 TTS 모델 fine-tuning을 수행하며 발생한 주요 시도, 실패 원인, 해결 방식을 정리한 기록이다. 목적은 이후 같은 문제를 반복하지 않고, 캡스톤 보고서에서 모델 학습 경로의 의사결정 근거를 설명할 수 있도록 남기는 것이다.

## Final Direction

최종 학습 경로는 다음으로 고정한다.

```text
AI Hub 감정 음성합성 데이터셋
-> scripts/prepare_aihub_emotion_dataset.py
-> Fish Speech 1.5 VQ extraction
-> protobuf dataset build
-> Fish Speech 1.5 text2semantic LoRA fine-tuning
-> LoRA merge
-> fine-tuned model.pth
```

최종 기준 모델과 코드는 다음을 사용한다.

- Code: `fishaudio/fish-speech` tag `v1.5.1`
- Base checkpoint: `fishaudio/fish-speech-1.5`
- VQ config: `firefly_gan_vq`
- VQ checkpoint: `firefly-gan-vq-fsq-8x1024-21hz-generator.pth`
- Fine-tune config: `text2semantic_finetune`
- LoRA config: `r_8_alpha_16`
- Notebook: `notebooks/colab_aihub_emotion_finetune_v15.ipynb`

## Attempt 1: Local Inference Feasibility Check

### Why We Tried

처음에는 README에서 소개하는 로컬 실물 inference 경로를 사용해, 로컬 PC 자원으로 Fish Speech 기반 TTS를 실행할 수 있는지 확인하려 했다. 목표는 `assets/reference` 디렉터리의 reference audio를 사용해 텍스트를 음성으로 생성하는 것이었다.

### Result

로컬 환경은 다음과 같았다.

```text
CPU: Ryzen 5 3600
RAM: 16 GB
GPU: RTX 2080 8 GB VRAM
```

실행 과정에서 모델 로딩과 inference에 필요한 VRAM/RAM 부담이 컸고, 특히 S1 계열 모델을 안정적으로 올리기에는 로컬 GPU VRAM이 부족하다고 판단했다.

### Decision

학습과 고비용 inference 준비 과정은 Google Colab Pro GPU에서 수행하고, 로컬은 최종적으로 학습된 모델을 받아 짧은 데모 inference를 시도하는 방향으로 분리했다.

## Attempt 2: OpenAudio S1-mini Fine-tuning Path

### Why We Tried

처음 계획은 `fishaudio/openaudio-s1-mini`를 기준 모델로 사용하는 것이었다. 이유는 다음과 같다.

- S1 계열보다 작아 로컬 추론 가능성이 높을 것으로 기대했다.
- Colab에서 LoRA fine-tuning을 수행한 뒤, merged checkpoint를 로컬 시연 파이프라인에 가져오는 구성이 가능해 보였다.
- `openaudio-s1-mini`는 최신 계열의 경량 모델로 보여 데모 목적에 적합해 보였다.

### Problems Observed

S1-mini 경로에서는 전처리, VQ 추출, protobuf 생성까지는 여러 패치로 통과했지만, LoRA 학습 단계에서 코드, checkpoint, tokenizer, dataset config의 불일치가 누적되었다.

주요 문제는 다음과 같다.

1. `pyaudio` build failure

   - Error: `fatal error: portaudio.h: No such file or directory`
   - Cause: Colab 환경에 PortAudio 개발 헤더가 없었다.
   - Temporary fix: `apt-get install portaudio19-dev libasound2-dev`

2. `torchvision::nms` runtime error

   - Error: `RuntimeError: operator torchvision::nms does not exist`
   - Cause: Colab의 `torch`, `torchvision`, `torchmetrics` 조합이 맞지 않았다.
   - Temporary fix: fine-tuning에 불필요한 `torchvision` 제거

3. GitHub main과 S1-mini checkpoint의 tokenizer mismatch

   - GitHub `fish-speech` main은 일부 경로에서 Hugging Face `AutoTokenizer` 로딩을 기대했다.
   - `openaudio-s1-mini`는 `tokenizer.tiktoken`을 포함하는 구조라 main branch 코드와 맞지 않았다.

4. S1-mini Hugging Face Space snapshot package layout issue

   - S1-mini Space revision을 고정해 사용하려 했으나, 해당 snapshot은 일반 Python package처럼 설치하기 어렵거나 필요한 dependency가 명확하지 않았다.
   - `descript-audiotools`, `descript-audio-codec`, `protobuf` 등 dependency를 수동으로 맞춰야 했다.

5. VQ extraction compatibility patch

   - `torchaudio.list_audio_backends()`가 최신 torchaudio에서 제거되어 shim을 추가했다.
   - DAC 모델의 sample rate 접근 방식이 `model.spec_transform.sample_rate`와 `model.sample_rate` 사이에서 맞지 않아 패치가 필요했다.

6. Training dataset class mismatch

   - Fine-tuning config가 `AutoTextSemanticInstructionIterableDataset`을 참조했지만, 사용 중인 snapshot에는 해당 class path가 맞지 않았다.
   - `AutoTextSemanticInstructionDataset`으로 바꾸는 임시 패치를 적용했다.

7. Final blocker: semantic token format mismatch

   - Error: `KeyError('<|semantic|>')`
   - Cause: 학습 dataset/tokenizer stack은 단일 `<|semantic|>` token을 기대했지만, S1-mini tokenizer는 indexed semantic token 형식을 사용했다.
   - This was not a harmless missing dependency. It was a semantic-format mismatch.

### Why We Stopped This Path

S1-mini 경로는 계속 패치하면 실행은 더 진행될 수 있었지만, tokenizer와 semantic token convention이 맞지 않아 학습 결과가 잘못될 위험이 있었다. 따라서 이 경로를 중단했다.

### Lesson

TTS fine-tuning에서는 모델 파일만 맞추는 것으로 충분하지 않다. 다음 네 요소가 같은 release family여야 한다.

```text
model checkpoint
tokenizer
VQ/codec config
fine-tuning dataset/config code
```

## Attempt 3: Fish Speech 1.5 Stable Fine-tuning Path

### Why We Tried

Fish Speech 1.5는 최신 모델은 아니지만, release 기준으로 inference와 fine-tuning 경로가 함께 제공되어 있다. 따라서 S1-mini보다 fine-tuning 성공 가능성이 높다고 판단했다.

### What Changed

새 기준은 다음으로 바꾸었다.

```text
S1-mini / OpenAudio path
-> Fish Speech 1.5 stable path
```

구체적으로는 다음 조합을 사용한다.

```text
fishaudio/fish-speech tag v1.5.1
fishaudio/fish-speech-1.5 checkpoint
firefly_gan_vq VQ config
text2semantic_finetune config
r_8_alpha_16 LoRA
```

### Implementation

다음 산출물을 추가했다.

- `FINETUNE_PLAN_V15.md`
- `notebooks/colab_aihub_emotion_finetune_v15.ipynb`

기존 S1-mini notebook은 deprecated로 표시했다.

- `notebooks/colab_aihub_emotion_finetune_fresh.ipynb`

### Smoke Test Result

`RUN_MODE = "smoke"`에서 다음 과정을 통과했다.

```text
감정당 10개 샘플
-> VQ extraction
-> protobuf build
-> 20 step LoRA train
-> LoRA merge
-> model.pth 생성
```

이 결과로 Fish Speech 1.5 기반 파이프라인은 최소 단위에서 end-to-end로 동작함을 확인했다.

## Attempt 4: Full Training Run and Colab Session Loss

### Why We Tried

Smoke test가 성공했기 때문에 `RUN_MODE = "full"`로 바꾸고, 감정당 300개씩 총 2,100개 샘플로 1,500 step LoRA fine-tuning을 수행했다.

### First Full Run Issue

초기 full training은 다음 상태에서 장시간 멈춘 것처럼 보였다.

```text
Epoch 0/-2 ... 0/1500
val/loss: ...
```

### Cause

학습이 train step으로 진입하기 전 validation 또는 sanity validation 단계에서 오래 머물렀다. 목적은 LoRA checkpoint 생성이므로, full run에서는 validation을 끄는 것이 더 적합하다고 판단했다.

### Fix

Train command에 validation 비활성화 옵션을 추가했다.

```bash
+trainer.num_sanity_val_steps=0
trainer.limit_val_batches=0
```

Hydra override 규칙상 두 옵션의 prefix가 다르다.

- `num_sanity_val_steps`는 config에 없어서 `++` 또는 `+`로 추가해야 한다.
- `limit_val_batches`는 config에 이미 있어서 `+` 없이 override해야 한다.

### Second Full Run Issue

학습이 실제로 진행되었지만 Colab 세션이 끊기면서 checkpoint가 남지 않았다. 원인은 checkpoint 저장 위치가 Colab 임시 파일 시스템인 `/content/...` 아래였기 때문이다.

### Fix

checkpoint 저장 위치를 Google Drive로 강제했다.

```bash
++callbacks.model_checkpoint.dirpath="{RUN_RESULTS_DIR / 'checkpoints'}"
```

정상 checkpoint 저장 경로는 다음이다.

```text
/content/drive/MyDrive/gyul-ai/emotion-tts/results/aihub_emotion_fish15_lora_full/checkpoints/
```

이후 `step_000000100.ckpt`, `step_000000200.ckpt` 등의 checkpoint가 Google Drive에 생성되는 것을 확인했다.

## Current Known Good Training Command

현재 full training에 사용하는 핵심 명령은 다음 형태다.

```python
%cd /content/fish-speech-v15

BATCH_SIZE = 2
GRAD_ACCUM = 8
gpu_name = torch.cuda.get_device_name(0).lower()
PRECISION = '16-mixed' if 't4' in gpu_name else 'bf16-true'

!python fish_speech/train.py --config-name text2semantic_finetune \
  project={PROJECT_NAME} \
  pretrained_ckpt_path="{BASE_CKPT}" \
  data.batch_size={BATCH_SIZE} \
  trainer.accumulate_grad_batches={GRAD_ACCUM} \
  trainer.max_steps={TRAIN_MAX_STEPS} \
  +trainer.num_sanity_val_steps=0 \
  trainer.limit_val_batches=0 \
  trainer.val_check_interval={CHECKPOINT_EVERY} \
  callbacks.model_checkpoint.every_n_train_steps={CHECKPOINT_EVERY} \
  ++callbacks.model_checkpoint.dirpath="{RUN_RESULTS_DIR / 'checkpoints'}" \
  trainer.precision={PRECISION} \
  hydra.run.dir="{RUN_RESULTS_DIR}" \
  +lora@model.model.lora_config=r_8_alpha_16
```

## Current Data Status

Full dataset preprocessing and VQ extraction have already succeeded.

Expected Drive outputs:

```text
/content/drive/MyDrive/gyul-ai/emotion-tts/processed/fish15_full/
  angry/*.wav, *.lab, *.npy
  anxious/*.wav, *.lab, *.npy
  embarrassed/*.wav, *.lab, *.npy
  happy/*.wav, *.lab, *.npy
  hurt/*.wav, *.lab, *.npy
  neutral/*.wav, *.lab, *.npy
  sad/*.wav, *.lab, *.npy

/content/drive/MyDrive/gyul-ai/emotion-tts/protos/fish15_full/00000000.protos
```

Observed counts:

```text
7 emotions
300 wav per emotion
300 lab per emotion
300 npy per emotion
1 protobuf shard
```

Therefore, if Colab disconnects during training, it is not necessary to rerun:

- `Prepare Dataset`
- `Extract VQ Tokens`
- `Build Protobuf Dataset`

Only the following need to be restored:

```text
Drive mount
Fish Speech 1.5 install
base checkpoint availability
data/protos symlink
LoRA Train
LoRA Merge
```

## Recovery Procedure After Colab Disconnect

If a Colab session disconnects, follow this order.

1. Mount Drive.

```python
from google.colab import drive
drive.mount('/content/drive')
```

2. Verify preprocessed data and protobuf.

```python
from pathlib import Path

DRIVE_ROOT = Path('/content/drive/MyDrive/gyul-ai/emotion-tts')
RUN_PROCESSED_DIR = DRIVE_ROOT / 'processed' / 'fish15_full'
RUN_PROTO_DIR = DRIVE_ROOT / 'protos' / 'fish15_full'

print(RUN_PROCESSED_DIR.exists(), RUN_PROCESSED_DIR)
print(RUN_PROTO_DIR.exists(), RUN_PROTO_DIR)
print(list(RUN_PROTO_DIR.glob('*.protos')))
```

3. Reinstall Fish Speech 1.5 stack using the notebook install cells.

4. Restore `data/protos` symlink.

```python
%cd /content/fish-speech-v15

!rm -rf data/protos
!mkdir -p data
!ln -s "/content/drive/MyDrive/gyul-ai/emotion-tts/protos/fish15_full" data/protos
!ls -la data/protos
```

5. Resume by rerunning LoRA Train. The current notebook does not resume optimizer state automatically; it restarts training from base checkpoint unless an explicit resume option is added. Because checkpoints are now saved to Drive every 100 steps, the latest saved LoRA checkpoint can at least be merged if another disconnect occurs.

## Checkpoint Verification

During full training, checkpoints should appear here:

```text
/content/drive/MyDrive/gyul-ai/emotion-tts/results/aihub_emotion_fish15_lora_full/checkpoints/
```

Expected examples:

```text
step_000000100.ckpt
step_000000200.ckpt
step_000000300.ckpt
step_000000400.ckpt
...
step_000001500.ckpt
```

Each LoRA checkpoint is expected to be around 35-36 MB.

If the latest checkpoint exists, it can be merged even if full training did not complete, but the best target is `step_000001500.ckpt`.

## Final Merge Target

After training, merge the latest LoRA checkpoint into a final model directory.

Expected final output:

```text
/content/drive/MyDrive/gyul-ai/emotion-tts/checkpoints/fish-speech-1.5-aihub-emotion-full/model.pth
```

This `model.pth` is the fine-tuned model artifact to use for later inference experiments.

## Key Lessons Learned

1. Use one consistent model family.

   `checkpoint`, `tokenizer`, `VQ config`, and `fine-tuning code` must come from the same release family.

2. Avoid ad hoc semantic-token patches.

   Dependency errors can be patched, but tokenizer semantic convention mismatch should be treated as a stop condition.

3. Always separate smoke and full runs.

   Smoke mode verified the end-to-end pipeline before spending Colab time on full training.

4. Force checkpoint output to Drive.

   Colab `/content` is temporary. Training checkpoints must be written to Google Drive.

5. Do not rerun expensive preprocessing if Drive artifacts are valid.

   Once `processed/fish15_full` and `protos/fish15_full` exist, rerun only the training environment and LoRA train.

6. Disable validation for the training-only capstone run.

   The project goal at this stage is producing a fine-tuned TTS model artifact, not running validation metrics inside Lightning.

