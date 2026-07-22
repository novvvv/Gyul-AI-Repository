"""인터뷰 세션 상태 — 리포트 발행(계약 §3)을 위한 턴 로그 축적."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from app.core.config import TARGET_SR


def new_session_id() -> str:
    """Kafka 멱등키로 쓰이는 전역 유일 sessionId (서버 생성, 클라이언트 값 무시)."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"intv-{stamp}-{uuid4().hex[:8]}"


@dataclass
class InterviewSessionState:
    email: str
    member_id: int | None
    persona_id: str = "gyul"
    session_id: str = field(default_factory=new_session_id)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    turns: list[dict] = field(default_factory=list)
    speech_samples: int = 0

    def record_turn(self, turn: dict) -> None:
        """run_final_turn 반환값을 리포트용 턴 로그(SessionTurn 호환)로 축적."""
        self.speech_samples += int(turn.pop("speech_samples", 0))
        self.turns.append(turn)

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def speech_duration_sec(self) -> int:
        return round(self.speech_samples / TARGET_SR)
