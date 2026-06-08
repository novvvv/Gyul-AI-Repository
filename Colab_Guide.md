# Google Colab에서 Kyul TTS 실행하는 방법

이 가이드는 다른 사용자가 이 레포지토리를 GitHub에서 받아와 Google Colab에서 실제 음성을 합성해보는 과정을 설명합니다.

Colab에서 새 노트(Python 3, **T4 GPU** 런타임 권장)를 열고 아래 셀들을 순서대로 복사해서 실행하시면 됩니다.

---

### Step 1. 레포지토리 클론 및 이동 (Cell 1)
GitHub 레포지토리를 Colab 가상환경으로 클론하고 해당 폴더로 이동합니다.
```bash
# GitHub 주소는 본인의 레포지토리 주소로 변경하여 실행하세요.
!git clone https://github.com/novvvv/Gyul-AI-Repository.git
%cd Gyul-AI-Repository
```

### Step 2. 의존성 및 패키지 설치 (Cell 2)
Colab 환경에서 `requirements.txt`를 설치하고 `fish-speech` 환경을 구축합니다.
```bash
# 프로젝트 의존성 라이브러리 설치
!pip install -r requirements.txt

# Fish Speech 설치 스크립트 실행 (/content/fish-speech-s2 에 설치됨)
!bash scripts/setup_fish_speech.sh
```

### Step 3. Hugging Face 로그인 (Cell 3)
모델 가중치(S2-Pro)를 Hugging Face에서 안전하게 다운로드하기 위해 허깅페이스 토큰 로그인을 수행합니다.
*(코드 실행 후 나타나는 입력창에 본인의 Hugging Face Access Token을 넣어주세요)*
```python
from huggingface_hub import notebook_login
notebook_login()
```

### Step 4. 모델 체크포인트 다운로드 (Cell 4)
사전 학습된 `s2-pro` 모델 가중치를 다운로드합니다.
```bash
!bash scripts/download_model.sh
```

### Step 5. 음성 합성 테스트 실행 (Cell 5)
원하는 텍스트를 넣어 한국어 음성을 합성합니다. 결과물은 `outputs/sample.wav`에 저장됩니다.
```bash
!python scripts/run_inference.py \
  --text "안녕하세요. 구글 코랩에서 실행한 음성 합성 테스트입니다. 정상적으로 목소리가 복제되었습니다." \
  --output sample.wav
```

### Step 6. 생성된 오디오 재생하기 (Cell 6)
Colab 노트북 내에서 생성된 오디오를 즉시 들어볼 수 있습니다.
```python
from IPython.display import Audio
Audio("outputs/sample.wav")
```
