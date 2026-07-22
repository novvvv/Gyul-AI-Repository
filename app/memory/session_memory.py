"""대화 세션 메모리 (MaTuna02 브랜치 RedisSessionMemory 포팅).

MEMORY_BACKEND=redis 설정 시 Redis DB 0(REDIS_URL)에 세션을 저장한다.
키 패턴 chat:{user}:{session}:{persona} — 연동 가이드 §4. TTL로 잔존물 방지.
"""

import json
from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Lock
from typing import Protocol

from app.core.config import (
    MEMORY_BACKEND,
    REDIS_URL,
    SESSION_MEMORY_MAX_TURNS,
    SESSION_MEMORY_TTL_SECONDS,
)


@dataclass(frozen=True)
class SessionContext:
    user_id: str
    session_id: str
    persona_id: str = "default"

    @property
    def key(self) -> str:
        return f"{self.user_id}:{self.session_id}:{self.persona_id}"


@dataclass(frozen=True)
class ConversationTurn:
    role: str
    content: str
    emotion: str | None = None

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "emotion": self.emotion,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationTurn":
        return cls(
            role=str(data["role"]),
            content=str(data["content"]),
            emotion=data.get("emotion"),
        )


class SessionMemory(Protocol):
    def append_user_message(
        self,
        context: SessionContext,
        content: str,
        emotion: str | None = None,
    ) -> None: ...

    def append_assistant_message(
        self, context: SessionContext, content: str
    ) -> None: ...

    def recent_messages(self, context: SessionContext) -> list[ConversationTurn]: ...

    def clear(self, context: SessionContext) -> None: ...


class InMemorySessionMemory:
    def __init__(self, max_turns: int = SESSION_MEMORY_MAX_TURNS) -> None:
        self.max_turns = max_turns
        self._store: dict[str, deque[ConversationTurn]] = defaultdict(
            lambda: deque(maxlen=max_turns)
        )
        self._lock = Lock()

    def append_user_message(
        self,
        context: SessionContext,
        content: str,
        emotion: str | None = None,
    ) -> None:
        self._append(context, ConversationTurn("user", content, emotion))

    def append_assistant_message(self, context: SessionContext, content: str) -> None:
        self._append(context, ConversationTurn("assistant", content))

    def recent_messages(self, context: SessionContext) -> list[ConversationTurn]:
        with self._lock:
            return list(self._store[context.key])

    def clear(self, context: SessionContext) -> None:
        with self._lock:
            self._store.pop(context.key, None)

    def _append(self, context: SessionContext, turn: ConversationTurn) -> None:
        with self._lock:
            self._store[context.key].append(turn)


class RedisSessionMemory:
    def __init__(
        self,
        redis_url: str = REDIS_URL,
        max_turns: int = SESSION_MEMORY_MAX_TURNS,
        ttl_seconds: int = SESSION_MEMORY_TTL_SECONDS,
    ) -> None:
        try:
            import redis
        except ImportError as e:
            raise RuntimeError(
                "redis 패키지가 없습니다. pip install -r requirements.txt를 실행해주세요."
            ) from e

        self.redis = redis.Redis.from_url(redis_url, decode_responses=True)
        self.max_turns = max_turns
        self.ttl_seconds = ttl_seconds

    def append_user_message(
        self,
        context: SessionContext,
        content: str,
        emotion: str | None = None,
    ) -> None:
        self._append(context, ConversationTurn("user", content, emotion))

    def append_assistant_message(self, context: SessionContext, content: str) -> None:
        self._append(context, ConversationTurn("assistant", content))

    def recent_messages(self, context: SessionContext) -> list[ConversationTurn]:
        raw_turns = self.redis.lrange(self._redis_key(context), 0, -1)
        turns = []
        for raw_turn in raw_turns:
            try:
                turns.append(ConversationTurn.from_dict(json.loads(raw_turn)))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        return turns

    def clear(self, context: SessionContext) -> None:
        self.redis.delete(self._redis_key(context))

    def _append(self, context: SessionContext, turn: ConversationTurn) -> None:
        key = self._redis_key(context)
        self.redis.rpush(key, json.dumps(turn.to_dict(), ensure_ascii=False))
        self.redis.ltrim(key, -self.max_turns, -1)
        self.redis.expire(key, self.ttl_seconds)

    def _redis_key(self, context: SessionContext) -> str:
        return f"chat:{context.key}"


def create_session_memory() -> SessionMemory:
    if MEMORY_BACKEND == "redis":
        return RedisSessionMemory()
    return InMemorySessionMemory()


session_memory = create_session_memory()
