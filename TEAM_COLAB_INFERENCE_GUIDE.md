# Team Colab Emotion TTS Inference Guide

이 문서는 학습이 끝난 Fish Speech 1.5 감정 TTS 모델을 팀원이 다시 학습하지 않고 사용하는 방법을 설명한다.

## Goal

팀원이 같은 Google Drive 계정 또는 공유받은 Drive 폴더를 Colab에 mount한 뒤, 텍스트와 감정만 입력해 wav 파일을 생성한다.

다시 수행하지 않는 과정:

```text
AI Hub 데이터 전처리
VQ token 추출
protobuf 생성
LoRA 학습
LoRA merge
```

매 Colab 세션마다 수행하는 과정:

```text
Drive mount
repo clone
Fish Speech 1.5 runtime 설치
학습 산출물 확인
텍스트 + 감정으로 wav 생성
```

## Use This Notebook

팀원은 아래 노트북만 실행하면 된다.

```text
notebooks/colab_fish15_emotion_inference_only.ipynb
```

노트북은 추론 전용이다. 학습 셀은 포함하지 않는다.

## Required Drive Files

팀 Colab 계정의 Drive에 다음 폴더가 있어야 한다.

```text
/content/drive/MyDrive/gyul-ai/emotion-tts/checkpoints/fish-speech-1.5/
/content/drive/MyDrive/gyul-ai/emotion-tts/checkpoints/fish-speech-1.5-aihub-emotion-full/
```

필수 파일:

```text
fish-speech-1.5/
  config.json
  model.pth
  special_tokens.json
  tokenizer.tiktoken
  firefly-gan-vq-fsq-8x1024-21hz-generator.pth

fish-speech-1.5-aihub-emotion-full/
  config.json
  model.pth
  special_tokens.json
  tokenizer.tiktoken
```

`fish-speech-1.5-aihub-emotion-full`은 fine-tuned text2semantic 모델이다. `fish-speech-1.5`의 `firefly-gan-vq-fsq-8x1024-21hz-generator.pth`는 semantic token을 wav로 복원하는 VQGAN decoder다.

## How To Run

1. Colab에서 `notebooks/colab_fish15_emotion_inference_only.ipynb`를 연다.
2. `Runtime > Change runtime type > GPU`를 선택한다.
3. 위에서부터 순서대로 실행한다.
4. `Verify Model Artifacts` 셀에서 모든 항목이 `True`인지 확인한다.
5. `Generate Wav` 셀에서 아래 두 값만 바꾼다.

```python
EMOTION = 'happy'
TEXT = '오늘 정말 잘했어. 조금만 더 힘내보자.'
```

6. `Generate Wav` 셀을 실행한다.

생성 파일은 기본적으로 아래 경로에 저장된다.

```text
/content/drive/MyDrive/gyul-ai/emotion-tts/generated_samples/team_inference/{emotion}_sample.wav
```

## Supported Emotions

사용 가능한 감정 이름은 다음과 같다.

```text
neutral
happy
sad
angry
anxious
hurt
embarrassed
```

스크립트는 내부적으로 감정 이름을 학습 때 사용한 태그로 변환한다.

```text
neutral     -> (indifferent)
happy       -> (happy)
sad         -> (sad)
angry       -> (angry)
anxious     -> (anxious)
hurt        -> (painful)
embarrassed -> (embarrassed)
```

## Direct Script Usage

노트북 대신 명령어로 실행하려면 다음 형태를 사용한다.

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

## Important Notes

- Colab의 `/content/fish-speech-v15`는 세션이 끝나면 사라진다. 이것은 정상이며 다음 세션에서 다시 clone하면 된다.
- 학습 결과는 `/content/drive/MyDrive/gyul-ai/emotion-tts/checkpoints` 아래에 있어야 한다.
- `model.pth` 하나만으로는 wav 생성이 되지 않는다. `config.json`, `tokenizer.tiktoken`, `special_tokens.json`, VQGAN decoder checkpoint가 함께 필요하다.
- 같은 Drive 계정을 쓰지 않는 팀원은 `gyul-ai/emotion-tts/checkpoints` 폴더를 공유받거나 별도로 복사해야 한다.
- 이 노트북은 학습을 재시작하지 않으므로 AI Hub dataset은 필요하지 않다.

