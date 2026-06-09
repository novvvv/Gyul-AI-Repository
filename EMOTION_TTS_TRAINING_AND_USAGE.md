# Emotion TTS Training And Usage

이 문서는 `MaTuna/tts` 브랜치에서 수행한 감정 반응형 TTS fine-tuning의 학습 과정, 학습 원리, 산출물 구성, 추론 사용 방법을 정리한다.

## Objective

캡스톤 목표는 "감정 반응형 TTS 모델 개발"이다. 이 브랜치에서는 완전 신규 TTS 모델을 처음부터 학습하지 않고, 사전학습된 Fish Speech 1.5 모델을 AI Hub 감정 음성합성 데이터셋으로 LoRA fine-tuning하여 감정 태그 조건에 반응하는 TTS 모델을 만드는 것을 목표로 한다.

최종 서비스 개념은 다음과 같다.

```text
사용자 음성
-> Whisper: 사용자 발화 텍스트
-> SER: 사용자 감정
-> LLM: 답변 텍스트 + 응답 감정 결정
-> fine-tuned Fish Speech 1.5: 감정 태그 기반 답변 음성 생성
```

이 브랜치의 직접 범위는 마지막 단계인 `fine-tuned Fish Speech 1.5` 모델 학습과 추론 준비다.

## Model Choice

초기에는 `fishaudio/openaudio-s1-mini`를 고려했지만, fine-tuning 경로에서 코드, tokenizer, checkpoint, semantic token 형식이 맞지 않아 중단했다. 최종적으로는 fine-tuning 절차가 안정적으로 맞는 Fish Speech 1.5 계열을 사용한다.

최종 기준:

```text
Code: fishaudio/fish-speech tag v1.5.1
Base checkpoint: fishaudio/fish-speech-1.5
VQ config: firefly_gan_vq
VQ decoder checkpoint: firefly-gan-vq-fsq-8x1024-21hz-generator.pth
Fine-tuning config: text2semantic_finetune
LoRA: r_8_alpha_16
```

## Training Data

학습 데이터는 AI Hub 감정 음성합성 데이터셋을 사용한다. 감정 분류는 다음 canonical label로 정리한다.

```text
neutral
happy
sad
angry
anxious
hurt
embarrassed
```

학습에 사용하는 실제 감정 태그는 `configs/emotion_tags.yaml`에 정의되어 있다.

```text
neutral     -> (indifferent)
happy       -> (happy)
sad         -> (sad)
angry       -> (angry)
anxious     -> (anxious)
hurt        -> (painful)
embarrassed -> (embarrassed)
```

전처리 스크립트는 원본 음성/텍스트를 Fish Speech fine-tuning 형식으로 변환한다.

```text
scripts/prepare_aihub_emotion_dataset.py
```

전처리 결과:

```text
processed/fish15_full/
  angry/*.wav, *.lab, *.npy
  anxious/*.wav, *.lab, *.npy
  embarrassed/*.wav, *.lab, *.npy
  happy/*.wav, *.lab, *.npy
  hurt/*.wav, *.lab, *.npy
  neutral/*.wav, *.lab, *.npy
  sad/*.wav, *.lab, *.npy

protos/fish15_full/00000000.protos
```

각 `.lab` 파일은 텍스트 앞에 감정 태그를 붙인다.

```text
(happy) 오늘 정말 잘했어. 조금만 더 힘내보자.
(sad) 괜찮아. 지금은 천천히 쉬어도 돼.
```

## Training Principle

Fish Speech 계열의 TTS는 크게 두 단계를 가진다.

```text
text
-> text2semantic model
-> semantic audio tokens
-> VQGAN decoder
-> waveform wav
```

학습에서는 원본 음성을 바로 waveform으로 예측하도록 학습하지 않는다. 먼저 VQGAN encoder로 음성을 discrete semantic token으로 바꾸고, text2semantic 모델이 입력 텍스트에서 해당 semantic token을 생성하도록 학습한다.

이번 fine-tuning의 핵심은 텍스트 앞에 감정 태그를 붙이는 것이다.

```text
(happy) 텍스트
(sad) 텍스트
(angry) 텍스트
```

모델은 이 감정 태그가 붙은 텍스트를 입력으로 보고, 그 감정 데이터셋에서 추출된 semantic token을 목표로 학습한다. 따라서 추론 시에도 같은 태그를 붙이면 모델이 해당 감정 조건을 반영한 semantic token을 생성하도록 유도된다.

## LoRA Fine-tuning

LoRA는 사전학습 모델 전체를 새로 학습하지 않고, 일부 weight 변화량을 낮은 rank의 adapter로 학습하는 방식이다. 이 프로젝트에서는 `r_8_alpha_16` LoRA 설정을 사용했다.

장점:

- 전체 모델 재학습보다 GPU 메모리와 시간이 적게 든다.
- base model의 일반 음성 생성 능력을 유지하면서 감정 태그 조건을 추가 학습할 수 있다.
- 학습 후 LoRA를 base model에 merge하여 하나의 `model.pth` 산출물로 사용할 수 있다.

학습 명령의 핵심 설정:

```text
trainer.max_steps=1500
data.batch_size=2
trainer.accumulate_grad_batches=8
+lora@model.model.lora_config=r_8_alpha_16
```

Colab 세션이 끊겨도 checkpoint를 보존하기 위해 checkpoint 저장 경로는 Google Drive로 강제한다.

```text
callbacks.model_checkpoint.dirpath=/content/drive/MyDrive/gyul-ai/emotion-tts/results/aihub_emotion_fish15_lora_full/checkpoints
```

## Training Pipeline

전체 학습 파이프라인은 다음 순서다.

```text
AI Hub 감정 음성합성 데이터셋
-> 감정 label 정규화
-> .wav + 감정 태그 포함 .lab 생성
-> Fish Speech 1.5 VQ token 추출
-> protobuf dataset build
-> text2semantic LoRA fine-tuning
-> step_000001500.ckpt 생성
-> LoRA merge
-> fine-tuned model.pth 생성
```

사용 노트북:

```text
notebooks/colab_aihub_emotion_finetune_v15.ipynb
```

학습 완료 후 최종 산출물:

```text
/content/drive/MyDrive/gyul-ai/emotion-tts/checkpoints/fish-speech-1.5-aihub-emotion-full/
  config.json
  model.pth
  special_tokens.json
  tokenizer.tiktoken
```

## Inference Principle

추론은 학습과 반대 방향으로 진행한다.

```text
emotion + text
-> emotion tag prefix 추가
-> fine-tuned text2semantic model
-> codes_0.npy
-> Fish Speech 1.5 VQGAN decoder
-> wav
```

예를 들어 사용자가 다음 입력을 주면:

```text
emotion = happy
text = 오늘 정말 잘했어. 조금만 더 힘내보자.
```

스크립트는 내부적으로 다음 텍스트를 모델에 입력한다.

```text
(happy) 오늘 정말 잘했어. 조금만 더 힘내보자.
```

그 결과 semantic token이 생성되고, VQGAN decoder가 이를 wav로 복원한다.

## Required Inference Artifacts

추론에는 fine-tuned 모델 폴더와 base decoder 폴더가 모두 필요하다.

```text
checkpoints/fish-speech-1.5/
  config.json
  model.pth
  special_tokens.json
  tokenizer.tiktoken
  firefly-gan-vq-fsq-8x1024-21hz-generator.pth

checkpoints/fish-speech-1.5-aihub-emotion-full/
  config.json
  model.pth
  special_tokens.json
  tokenizer.tiktoken
```

`fish-speech-1.5-aihub-emotion-full/model.pth`만으로는 wav를 생성할 수 없다. VQGAN decoder checkpoint인 `firefly-gan-vq-fsq-8x1024-21hz-generator.pth`도 필요하다.

## Inference Notebook

팀원이 학습 없이 바로 추론만 수행하려면 다음 노트북을 사용한다.

```text
notebooks/colab_fish15_emotion_inference_only.ipynb
```

실행 순서:

```text
1. Colab GPU runtime 선택
2. Drive mount
3. repo clone
4. Fish Speech 1.5 runtime 설치
5. model artifact 확인
6. EMOTION/TEXT 입력
7. wav 생성
```

마지막 생성 셀에서 바꾸는 값:

```python
EMOTION = "happy"
TEXT = "오늘 정말 잘했어. 조금만 더 힘내보자."
```

결과 저장 경로:

```text
/content/drive/MyDrive/gyul-ai/emotion-tts/generated_samples/team_inference/{emotion}_sample.wav
```

## Script Usage

추론 스크립트:

```text
scripts/run_fish15_emotion_inference.py
```

Colab Drive 기준 예시:

```bash
python scripts/run_fish15_emotion_inference.py \
  --fish-speech-dir /content/fish-speech-v15 \
  --checkpoint-dir /content/drive/MyDrive/gyul-ai/emotion-tts/checkpoints/fish-speech-1.5-aihub-emotion-full \
  --base-checkpoint-dir /content/drive/MyDrive/gyul-ai/emotion-tts/checkpoints/fish-speech-1.5 \
  --emotion happy \
  --text "오늘 정말 잘했어. 조금만 더 힘내보자." \
  --output /content/drive/MyDrive/gyul-ai/emotion-tts/generated_samples/team_inference/happy.wav \
  --half
```

로컬 또는 다른 GPU 서버 기준 예시:

```bash
python scripts/run_fish15_emotion_inference.py \
  --fish-speech-dir ./fish-speech-v15 \
  --checkpoint-dir ./checkpoints/fish-speech-1.5-aihub-emotion-full \
  --base-checkpoint-dir ./checkpoints/fish-speech-1.5 \
  --emotion sad \
  --text "괜찮아. 지금은 천천히 쉬어도 돼." \
  --output ./outputs/fish15_emotion/sad.wav \
  --half
```

CPU에서도 `--device cpu`로 실행할 수 있지만 매우 느릴 수 있다.

## Important Notes

- Repo에는 학습 데이터와 모델 checkpoint를 커밋하지 않는다.
- Colab `/content` 아래 파일은 세션 종료 시 사라진다.
- 최종 모델 산출물은 Google Drive의 `gyul-ai/emotion-tts/checkpoints` 아래에 보관해야 한다.
- 팀원이 같은 Google Drive 계정을 사용하거나 checkpoint 폴더를 공유받으면 학습 없이 바로 추론할 수 있다.
- 추론 전용 노트북은 AI Hub dataset을 필요로 하지 않는다.

