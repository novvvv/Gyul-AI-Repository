uvicorn ser_api:app --reload
python3 -m http.server 5500
http://127.0.0.1:5500/web_test.html
export OPENAI_API_KEY="여기에\*진짜\_API키"

GET /health
서버/모델 로드 상태 확인
응답 : {"ok": true/fasle}

WebSocket /ws/predict
실시간 스트리밍 추론
Client가 PCM16 Audio Byte 전송
서버가 VAD로 음성/무음 판단 후 발화 단위 처리

POST /predict [Debug]
파일 업로드 방식 추론
Input : Audio FIle
Response : label, confidence, probs, filename
