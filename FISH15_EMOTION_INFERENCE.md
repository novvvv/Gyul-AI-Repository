# Fish Speech 1.5 Emotion Inference

이 문서는 Colab 학습 세션이 종료된 뒤에도 fine-tuned Fish Speech 1.5 모델로 텍스트와 감정 태그만 입력해 짧은 wav를 생성하는 방법을 정리한다.

## Required Artifacts

Colab Drive에서 다음 두 모델 폴더를 보관하거나 내려받아야 한다.

```text
/content/drive/MyDrive/gyul-ai/emotion-tts/checkpoints/
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

`fish-speech-1.5-aihub-emotion-full`은 fine-tuned text2semantic 모델이고, `fish-speech-1.5`의 `firefly-gan-vq-fsq-8x1024-21hz-generator.pth`는 semantic token을 wav로 복원하는 decoder다.

Fine-tuned `model.pth` 하나만으로는 wav 생성이 끝나지 않는다. tokenizer/config와 VQGAN decoder checkpoint가 함께 필요하다.

## Environment

다른 Colab 런타임이나 Linux GPU 환경에서 Fish Speech 1.5 코드를 준비한다.

```bash
git clone --branch v1.5.1 --depth 1 https://github.com/fishaudio/fish-speech.git fish-speech-v15
cd fish-speech-v15
python -m pip install -e . --no-deps
```

노트북 `notebooks/colab_aihub_emotion_finetune_v15.ipynb`의 `Install Fish Speech 1.5 Stack` 셀과 같은 dependency 설치를 수행한다.

## Emotion Tags

스크립트는 `configs/emotion_tags.yaml`을 읽어 감정 이름을 학습 때 사용한 태그로 변환한다.

```text
neutral     -> (indifferent)
happy       -> (happy)
sad         -> (sad)
angry       -> (angry)
anxious     -> (anxious)
hurt        -> (painful)
embarrassed -> (embarrassed)
```

## Generate Wav

예시 경로는 Colab Drive 기준이다.

```bash
python scripts/run_fish15_emotion_inference.py \
  --fish-speech-dir /content/fish-speech-v15 \
  --checkpoint-dir /content/drive/MyDrive/gyul-ai/emotion-tts/checkpoints/fish-speech-1.5-aihub-emotion-full \
  --base-checkpoint-dir /content/drive/MyDrive/gyul-ai/emotion-tts/checkpoints/fish-speech-1.5 \
  --emotion happy \
  --text "오늘 정말 잘했어. 조금만 더 힘내보자." \
  --output /content/drive/MyDrive/gyul-ai/emotion-tts/generated_samples/local_reuse/happy.wav \
  --half
```

CPU 환경에서는 `--device cpu`를 줄 수 있지만 매우 느릴 수 있다.

```bash
python scripts/run_fish15_emotion_inference.py \
  --fish-speech-dir ./fish-speech-v15 \
  --checkpoint-dir ./checkpoints/fish-speech-1.5-aihub-emotion-full \
  --base-checkpoint-dir ./checkpoints/fish-speech-1.5 \
  --emotion sad \
  --text "괜찮아. 지금은 천천히 쉬어도 돼." \
  --output ./outputs/sad.wav \
  --device cpu
```

## Pipeline

스크립트 내부 동작은 다음과 같다.

```text
emotion + text
-> emotion tag prefix 추가
-> fish_speech/models/text2semantic/inference.py
-> codes_0.npy 생성
-> fish_speech/models/vqgan/inference.py
-> wav 생성
```

Fish Speech 1.5에서는 wav 복원에 `fish_speech/models/vqgan/inference.py`를 사용한다. `dac/inference.py`는 S1/S2 계열에서 쓰이는 경로와 혼동될 수 있으므로 이 실험의 1.5 inference에서는 사용하지 않는다.

## Local Notes

로컬 PC가 Ryzen 3600, RAM 16 GB, RTX 2080 8 GB VRAM 조합이라면 짧은 문장 inference는 시도할 수 있지만, 속도나 VRAM 여유는 제한적일 수 있다. 데모 안정성이 중요하면 Colab GPU에서 wav를 생성한 뒤 결과 파일만 내려받는 방식이 더 안전하다.

