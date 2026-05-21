#!/usr/bin/env python3
# 웹소켓 테스트용 스크립트 
"""WebSocket으로 '안녕하세요' 텍스트 + 최소 오디오로 final 응답 테스트."""

import asyncio
import json
import sys

import numpy as np

try:
    import websockets
except ImportError:
    print("pip install websockets 필요")
    sys.exit(1)

WS_URL = "ws://127.0.0.1:8000/ws/predict?user_id=test&session_id=s1&persona_id=gyul"
TARGET_SR = 16000
SPEECH_SECONDS = 0.5
SILENCE_SECONDS = 0.8
CHUNK_SAMPLES = 4096


def _speech_chunk() -> bytes:
    n = CHUNK_SAMPLES
    t = np.arange(n, dtype=np.float32) / TARGET_SR
    wave = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    return (wave * 32767).astype(np.int16).tobytes()


def _silence_chunk() -> bytes:
    return np.zeros(CHUNK_SAMPLES, dtype=np.int16).tobytes()


async def main() -> None:
    print(f"연결: {WS_URL}")
    async with websockets.connect(WS_URL, open_timeout=30) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "session_start",
                    "user_id": "test",
                    "session_id": "s1",
                    "persona_id": "gyul",
                },
                ensure_ascii=False,
            )
        )
        ready = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
        print("수신:", ready)

        await ws.send(
            json.dumps(
                {"type": "utterance_text", "text": "안녕하세요"},
                ensure_ascii=False,
            )
        )
        print('전송: "안녕하세요"')

        speech_chunks = int(SPEECH_SECONDS * TARGET_SR / CHUNK_SAMPLES) + 1
        silence_chunks = int(SILENCE_SECONDS * TARGET_SR / CHUNK_SAMPLES) + 1

        for _ in range(speech_chunks):
            await ws.send(_speech_chunk())
        for _ in range(silence_chunks):
            await ws.send(_silence_chunk())

        print("오디오 청크 전송 완료, final 응답 대기 중... (로컬 LLM이면 수 분 걸릴 수 있음)")

        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=600)
            data = json.loads(raw)
            print("수신:", json.dumps(data, ensure_ascii=False, indent=2))
            if data.get("type") == "final":
                print("\n--- AI 답변 ---")
                print(data.get("reply", "(없음)"))
                print("--- 감정 ---")
                print(data.get("label"), data.get("confidence"))
                break


if __name__ == "__main__":
    asyncio.run(main())
