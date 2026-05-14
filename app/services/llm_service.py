import json
import os
from urllib import error, request

from app.core.config import OPENAI_MODEL


class LLMReplyService:
    def generate_reply(self, text: str, emotion: str) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return f"OPENAI_API_KEY가 없어서 감정({emotion})만 분석했어요."

        payload = {
            "model": OPENAI_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "너는 한국어 대화 어시스턴트다. "
                        "사용자의 문장과 감정 분석 결과를 반영해 짧고 자연스럽게 답해라."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"사용자 발화: {text}\n감정: {emotion}\n한두 문장으로 답변해줘."
                    ),
                },
            ],
            "temperature": 0.7,
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=20) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"].strip()
        except error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="ignore")
            return f"AI 답변 호출 실패(HTTP {e.code}): {detail[:180]}"
        except Exception as e:
            return f"AI 답변 호출 실패: {e}"


llm_reply_service = LLMReplyService()
