from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EmotionSignal:
    label: str
    confidence: float
    probs: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> EmotionSignal | None:
        if not data or not data.get("label"):
            return None
        return cls(
            label=str(data["label"]),
            confidence=float(data.get("confidence") or 0),
            probs={str(k): float(v) for k, v in (data.get("probs") or {}).items()},
        )


@dataclass
class SessionTurn:
    user_text: str
    voice_emotion: EmotionSignal | None = None
    face_emotion: EmotionSignal | None = None
    bot_reply: str = ""
    at: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionTurn:
        return cls(
            user_text=str(data.get("user_text") or "").strip(),
            voice_emotion=EmotionSignal.from_dict(data.get("voice_emotion")),
            face_emotion=EmotionSignal.from_dict(data.get("face_emotion")),
            bot_reply=str(data.get("bot_reply") or "").strip(),
            at=data.get("at"),
        )


@dataclass
class SessionSnapshot:
    user_id: str
    session_id: str
    persona_id: str
    started_at: str | None = None
    ended_at: str | None = None
    turns: list[SessionTurn] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionSnapshot:
        session = data.get("session") or data
        turns_raw = data.get("turns") or session.get("turns") or []
        return cls(
            user_id=str(session.get("user_id") or "anonymous"),
            session_id=str(session.get("session_id") or "default"),
            persona_id=str(session.get("persona_id") or "gyul"),
            started_at=session.get("started_at"),
            ended_at=session.get("ended_at"),
            turns=[SessionTurn.from_dict(t) for t in turns_raw if t],
        )
