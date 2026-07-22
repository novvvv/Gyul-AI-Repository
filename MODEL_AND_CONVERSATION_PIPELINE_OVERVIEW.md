# Model Training, Inference, And Conversation Pipeline Overview

이 문서는 `MaTuna/tts` 브랜치의 Fish Speech 1.5 감정 TTS 학습/추론 방법론과, `main` 브랜치의 WebSocket 및 LangChain 기반 대화 파이프라인을 함께 설명한다.

## 1. 전체 서비스 개념

프로젝트의 최종 목표는 사용자의 음성과 텍스트를 바탕으로 감정을 추론하고, 그 감정과 대화 맥락을 반영해 답변을 생성한 뒤, 감정이 반영된 음성으로 응답하는 것이다.

```text
사용자 음성
-> STT: 사용자 발화 텍스트
-> SER: 사용자 음성 감정 분석
-> LLM: 대화 맥락 + 감정 기반 답변 생성
-> TTS: 답변 텍스트 + 감정 태그 기반 음성 생성
-> 사용자에게 음성/텍스트 응답
```

`MaTuna/tts` 브랜치는 이 중 TTS 모델 학습과 추론을 중심으로 다룬다.

`main` 브랜치는 WebSocket, SER, LLM, 세션 메모리, 프론트엔드 데모, TTS API 연동, 얼굴 감정 분석, 세션 리포트까지 포함하는 통합 대화 파이프라인을 다룬다.

## 2. Fish Speech 1.5 감정 TTS 학습 방법론

### 2.1 모델 선택

처음에는 `fishaudio/openaudio-s1-mini`도 검토했지만, fine-tuning 과정에서 코드, tokenizer, checkpoint, semantic token 형식이 서로 맞지 않아 중단했다.

최종적으로는 Fish Speech 1.5 계열을 사용했다.

```text
Code: fishaudio/fish-speech tag v1.5.1
Base checkpoint: fishaudio/fish-speech-1.5
VQ config: firefly_gan_vq
VQ decoder checkpoint: firefly-gan-vq-fsq-8x1024-21hz-generator.pth
Fine-tuning config: text2semantic_finetune
LoRA config: r_8_alpha_16
```

이 조합은 VQ extraction, protobuf packing, text2semantic fine-tuning, LoRA merge가 같은 release family 안에서 동작하기 때문에 안정적이다.

### 2.2 학습 데이터

학습 데이터는 AI Hub 감정 음성합성 데이터셋을 사용한다.

감정 label은 다음 7개 canonical emotion으로 정리한다.

```text
neutral
happy
sad
angry
anxious
hurt
embarrassed
```

실제 모델 입력에 붙는 감정 태그는 `configs/emotion_tags.yaml`에 정의되어 있다.

```text
neutral     -> (indifferent)
happy       -> (happy)
sad         -> (sad)
angry       -> (angry)
anxious     -> (anxious)
hurt        -> (painful)
embarrassed -> (embarrassed)
```

전처리 스크립트는 원본 음성과 텍스트를 Fish Speech fine-tuning 형식으로 변환한다.

```text
scripts/prepare_aihub_emotion_dataset.py
```

각 학습 샘플은 `.wav`와 `.lab` 쌍으로 저장된다. `.lab` 파일에는 원문 텍스트 앞에 감정 태그가 붙는다.

```text
(happy) 오늘 정말 잘했어. 조금만 더 힘내보자.
(sad) 괜찮아. 지금은 천천히 쉬어도 돼.
```

즉 모델은 단순히 텍스트에서 음성을 생성하는 법만 배우는 것이 아니라, 감정 태그가 붙은 텍스트에서 해당 감정의 음성 특징을 가진 semantic token을 생성하도록 학습된다.

### 2.3 VQ의 의미

여기서 말하는 VQ는 Vector Quantization이다.

음성 waveform은 원래 연속적인 실수 값이다.

```text
0.012, -0.034, 0.108, ...
```

Fish Speech 계열 모델은 이 waveform을 직접 예측하지 않고, 먼저 VQ encoder를 통해 이산적인 audio token sequence로 바꾼다.

```text
원본 wav
-> VQ encoder
-> discrete semantic audio tokens
예: [523, 18, 901, 44, ...]
```

이 token들이 text2semantic 모델의 학습 목표가 된다.

학습 시에는 다음 방향으로 진행된다.

```text
원본 wav
-> VQ token 추출
-> 텍스트 + 감정 태그에서 해당 VQ token을 생성하도록 text2semantic 학습
```

추론 시에는 반대로 진행된다.

```text
텍스트 + 감정 태그
-> text2semantic 모델
-> VQ token 생성
-> VQGAN decoder
-> wav 복원
```

따라서 VQ는 음성을 LLM 계열 모델이 다루기 쉬운 token 형태로 바꾸는 중간 표현이라고 볼 수 있다.

### 2.4 LoRA fine-tuning

LoRA는 사전학습 모델 전체를 새로 학습하지 않고, 일부 weight 변화량만 낮은 rank의 adapter로 학습하는 방식이다.

이 프로젝트에서는 Fish Speech 1.5의 text2semantic 모델에 `r_8_alpha_16` LoRA 설정을 적용했다.

핵심 학습 설정은 다음과 같다.

```text
data.batch_size=2
trainer.accumulate_grad_batches=8
trainer.max_steps=1500
trainer.limit_val_batches=0
+lora@model.model.lora_config=r_8_alpha_16
```

full training 기준으로는 감정당 300개, 총 2,100개 샘플을 사용한다.

학습 중 Colab 세션이 끊겨도 checkpoint가 남도록 checkpoint 저장 경로는 Google Drive로 지정했다.

```text
/content/drive/MyDrive/gyul-ai/emotion-tts/results/aihub_emotion_fish15_lora_full/checkpoints
```

전체 학습 파이프라인은 다음 순서다.

```text
AI Hub 감정 음성합성 데이터셋
-> 감정 label 정규화
-> 감정 태그 포함 .lab 생성
-> Fish Speech 1.5 VQ token 추출
-> protobuf dataset 생성
-> text2semantic LoRA fine-tuning
-> LoRA checkpoint 생성
-> LoRA merge
-> fine-tuned model.pth 생성
```

학습 완료 후 최종 산출물은 다음 폴더에 저장된다.

```text
fish-speech-1.5-aihub-emotion-full/
  config.json
  model.pth
  special_tokens.json
  tokenizer.tiktoken
```

## 3. Fish Speech 1.5 감정 TTS 추론 방법론

추론은 학습과 반대 방향으로 진행된다.

```text
emotion + text
-> emotion tag prefix 추가
-> fine-tuned text2semantic model
-> codes_0.npy 생성
-> Fish Speech 1.5 VQGAN decoder
-> wav 생성
```

예를 들어 사용자가 다음 값을 입력하면:

```text
emotion = happy
text = 오늘 정말 잘했어. 조금만 더 힘내보자.
```

추론 스크립트는 내부적으로 다음 입력을 만든다.

```text
(happy) 오늘 정말 잘했어. 조금만 더 힘내보자.
```

그 다음 Fish Speech 1.5의 `text2semantic/inference.py`를 호출한다.

```text
fish_speech/models/text2semantic/inference.py
--text "(happy) ..."
--checkpoint-path fish-speech-1.5-aihub-emotion-full
--num-samples 1
--output-dir semantic_dir
```

이 단계에서 `codes_0.npy` 같은 semantic token 파일이 생성된다.

이후 VQGAN decoder가 semantic token을 waveform으로 복원한다.

```text
fish_speech/models/vqgan/inference.py
-i codes_0.npy
-o output.wav
--config-name firefly_gan_vq
--checkpoint-path firefly-gan-vq-fsq-8x1024-21hz-generator.pth
```

따라서 fine-tuned `model.pth`만으로는 wav 생성이 끝나지 않는다. 추론에는 fine-tuned text2semantic 모델과 base Fish Speech 1.5의 VQGAN decoder checkpoint가 모두 필요하다.

```text
fish-speech-1.5-aihub-emotion-full/
  model.pth
  config.json
  tokenizer.tiktoken
  special_tokens.json

fish-speech-1.5/
  firefly-gan-vq-fsq-8x1024-21hz-generator.pth
```

정리하면, fine-tuned 모델은 텍스트와 감정 태그를 semantic audio token으로 바꾸는 역할을 하고, VQGAN decoder는 그 token을 실제 wav로 복원하는 역할을 한다.

## 4. main 브랜치 대화 파이프라인

`main` 브랜치의 대화 파이프라인은 FastAPI WebSocket을 중심으로 동작한다.

```text
React Demo UI
-> 브라우저 STT + 마이크 PCM 오디오
-> WebSocket /ws/predict
-> VAD로 발화 구간 감지
-> SER 음성 감정 분류
-> 세션 메모리에서 이전 대화 조회
-> persona + history + 현재 감정으로 LLM 프롬프트 구성
-> OpenAI 또는 Kanana 로컬 LLM 응답 생성
-> 선택적으로 Fish Audio TTS로 답변 음성 생성
-> WebSocket final 응답
-> 프론트 채팅 UI / 리포트 데이터 누적
```

### 4.1 WebSocket 입력 구조

WebSocket endpoint는 다음이다.

```text
GET /ws/predict
```

클라이언트는 두 종류의 데이터를 보낸다.

```text
1. bytes: 브라우저 마이크에서 온 PCM16 mono audio
2. text: session_start, utterance_text, flush, session_end 같은 제어 메시지
```

중요한 점은 STT를 서버에서 하지 않는다는 것이다. 브라우저가 Web Speech API로 STT를 수행하고, 서버에는 텍스트를 별도 메시지로 보낸다.

```text
브라우저 마이크 오디오 -> 서버 SER용
브라우저 STT 텍스트 -> 서버 LLM 답변용
```

### 4.2 VAD와 SER

서버는 PCM audio chunk를 `numpy` 배열로 바꾼 뒤 RMS 기반 VAD를 수행한다.

```text
RMS >= threshold -> speech
speech 이후 silence가 일정 시간 지속 -> 발화 종료
```

발화 종료가 감지되면 SER 모델이 전체 발화 audio를 분석한다.

SER는 `transformers` 기반 음성 분류 모델이다.

```text
AutoFeatureExtractor
AutoModelForAudioClassification
```

출력은 다음과 같은 형태다.

```json
{
  "label": "fear",
  "confidence": 0.82,
  "probs": {
    "fear": 0.82,
    "neutral": 0.08
  }
}
```

### 4.3 세션 메모리

대화 맥락은 `InMemorySessionMemory`에 저장된다.

세션 키는 다음 조합으로 만들어진다.

```text
user_id:session_id:persona_id
```

각 세션은 최근 20턴을 저장한다.

```text
user message: content + emotion
assistant message: content
```

같은 `user_id`, `session_id`, `persona_id`로 연결하면 이전 대화가 LLM 프롬프트에 다시 포함된다.

다만 이 메모리는 Redis나 DB가 아니라 프로세스 메모리 기반이므로, 서버가 재시작되면 사라진다.

### 4.4 LLM 프롬프트 구성

LLM 입력 메시지는 다음 세 층으로 구성된다.

```text
system: persona별 역할 지시
history: 이전 user/assistant 대화
current user: 현재 발화 + 현재 감정 신호
```

persona는 다음과 같이 구분된다.

```text
default: 일반 한국어 대화 어시스턴트
gyul: 자기분석 대화 상대
interviewer: AI 면접관
```

현재 사용자 발화는 다음과 같은 형태로 LLM에 들어간다.

```text
오늘 면접 생각만 하면 좀 긴장돼요.
[현재 감정 신호: fear]
위 감정 신호는 답변의 톤 조절에만 참고해줘.
```

이전 사용자 발화에도 감정 신호가 붙는다.

```text
이전 사용자 발화
[감정 신호: sadness]
```

즉 SER 결과는 답변 내용을 강제로 결정하는 값이 아니라, 답변의 톤과 공감 방향을 조절하는 보조 신호로 사용된다.

### 4.5 LangChain 활용

LangChain은 복잡한 chain graph를 구성하는 용도라기보다, OpenAI/Gemini chat model 호출을 위한 adapter 역할로 사용된다.

내부 메시지 형식은 다음 LangChain 메시지 타입으로 변환된다.

```text
role=system    -> SystemMessage
role=user      -> HumanMessage
role=assistant -> AIMessage
```

OpenAI 호출에는 `ChatOpenAI`, Gemini 호출에는 `ChatGoogleGenerativeAI`를 사용할 수 있게 구현되어 있다.

다만 현재 `main` 브랜치의 실제 LLM 라우터는 단순하게 동작한다.

```text
OPENAI_API_KEY 있음 -> OpenAI gpt-4o-mini
OPENAI_API_KEY 없음 -> Kanana 로컬 HF 모델
```

Gemini용 LangChain adapter는 존재하지만, 현재 기본 라우터에서는 OpenAI와 Kanana fallback이 중심이다.

### 4.6 로컬 LLM fallback

OpenAI API key가 없으면 Kanana 로컬 모델을 사용한다.

기술 구성은 다음과 같다.

```text
transformers AutoTokenizer
transformers AutoModelForCausalLM
tokenizer.apply_chat_template()
torch generate()
CUDA -> MPS -> CPU 순서 device 선택
```

기본 모델은 다음이다.

```text
kakaocorp/kanana-1.5-2.1b-instruct-2505
```

### 4.7 답변 TTS

LLM 답변을 음성으로 변환하는 기능도 있다.

`FISH_AUDIO_API_KEY`가 설정되어 있으면 Fish Audio API를 호출한다.

```text
LLM reply + 사용자 SER 감정
-> 감정별 Fish Audio 태그 추가
-> Fish Audio /v1/tts 호출
-> mp3 bytes
-> base64
-> WebSocket final payload에 reply_audio_b64 포함
```

예를 들어 사용자 감정이 `fear`면 답변 앞에 다음과 같은 태그를 붙일 수 있다.

```text
[whispering][soft]
```

이 기능은 `MaTuna/tts` 브랜치의 직접 fine-tuned Fish Speech 1.5 모델과는 별개로, `main` 브랜치에서 Fish Audio API를 사용하는 방식이다.

### 4.8 얼굴 감정 분석

`main` 브랜치에는 얼굴 검출과 표정 감정 분석도 포함되어 있다.

얼굴 검출:

```text
InsightFace buffalo_l
```

표정 감정 분류:

```text
trpakov/vit-face-expression
```

프론트엔드는 주기적으로 웹캠 프레임을 캡처해 서버에 보낸다.

```text
/detect_face -> 얼굴 bbox
/predict_face -> 표정 감정
```

얼굴 감정은 실시간 LLM 응답에 직접 들어가기보다는, 세션 종료 후 리포트 생성을 위한 turn log에 저장된다.

### 4.9 세션 리포트

대화 종료 후 프론트는 누적된 turn log를 `/session/report`로 보낼 수 있다.

리포트 생성에는 다음 데이터가 사용된다.

```text
사용자 발화
음성 감정
얼굴 감정
봇 답변
감정 변화
음성/표정 불일치
```

서버는 이 데이터를 집계한 뒤 같은 LLM 라우터를 사용해 자기성찰 리포트를 생성한다.

## 5. 두 브랜치의 연결 관점

`main` 브랜치의 현재 TTS는 Fish Audio API를 통해 답변 음성을 생성한다.

`MaTuna/tts` 브랜치의 목표는 이 TTS 부분을 직접 fine-tuned Fish Speech 1.5 감정 모델로 대체하거나 확장할 수 있는 기반을 만드는 것이다.

통합 관점에서는 다음 구조가 된다.

```text
main 브랜치 대화 파이프라인
  사용자 음성
  -> SER 감정 분석
  -> LLM 답변 텍스트 생성
  -> 응답 감정 또는 사용자 감정 선택
  -> MaTuna/tts fine-tuned Fish Speech 1.5 추론
  -> 감정 태그 기반 wav 생성
```

여기서 핵심 연결점은 다음 두 값이다.

```text
text: LLM이 생성한 답변 문장
emotion: SER 또는 LLM이 결정한 응답 감정
```

이 두 값을 `scripts/run_fish15_emotion_inference.py` 같은 추론 계층에 넘기면:

```text
emotion + text
-> emotion tag prefix
-> text2semantic
-> VQ token
-> VQGAN decoder
-> wav
```

형태로 최종 음성을 생성할 수 있다.

## 6. 요약

Fish Speech 1.5 감정 TTS 학습의 핵심은 감정 태그가 붙은 텍스트를 입력으로 하여, 해당 감정 음성에서 추출한 VQ semantic token을 생성하도록 text2semantic 모델을 LoRA fine-tuning하는 것이다.

추론에서는 사용자의 감정 이름을 학습 때 사용한 태그로 변환하고, 답변 텍스트 앞에 붙인 뒤 semantic token을 생성한다. 이후 VQGAN decoder가 token을 wav로 복원한다.

`main` 브랜치의 대화 파이프라인은 WebSocket 기반 실시간 오디오 처리, 브라우저 STT, SER, 세션 메모리, persona prompt, LangChain/OpenAI/Kanana LLM, Fish Audio TTS, 얼굴 감정 분석, 세션 리포트를 결합한 멀티모달 대화 시스템이다.

최종적으로 두 흐름을 연결하면, 사용자의 음성과 감정에서 시작해 LLM 답변을 만들고, 그 답변을 감정 태그 기반 fine-tuned TTS로 음성화하는 end-to-end 감정 반응형 대화 시스템이 된다.
