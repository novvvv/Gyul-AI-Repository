from threading import Lock
from typing import Protocol

from app.core.config import MEMORY_BACKEND, REDIS_URL, SESSION_MEMORY_TTL_SECONDS
from app.memory.session_memory import SessionContext


class SummaryMemory(Protocol):
    def get_summary(self, context: SessionContext) -> str:
        ...

    def save_summary(self, context: SessionContext, summary: str) -> None:
        ...

    def clear(self, context: SessionContext) -> None:
        ...


class InMemorySummaryMemory:
    def __init__(self) -> None:
        self._store: dict[str, str] = {}
        self._lock = Lock()

    def get_summary(self, context: SessionContext) -> str:
        with self._lock:
            return self._store.get(context.key, "")

    def save_summary(self, context: SessionContext, summary: str) -> None:
        with self._lock:
            self._store[context.key] = summary.strip()

    def clear(self, context: SessionContext) -> None:
        with self._lock:
            self._store.pop(context.key, None)


class RedisSummaryMemory:
    def __init__(
        self,
        redis_url: str = REDIS_URL,
        ttl_seconds: int = SESSION_MEMORY_TTL_SECONDS,
    ) -> None:
        try:
            import redis
        except ImportError as e:
            raise RuntimeError(
                "redis 패키지가 없습니다. pip install -r requirements.txt를 실행해주세요."
            ) from e

        self.redis = redis.Redis.from_url(redis_url, decode_responses=True)
        self.ttl_seconds = ttl_seconds

    def get_summary(self, context: SessionContext) -> str:
        return self.redis.get(self._redis_key(context)) or ""

    def save_summary(self, context: SessionContext, summary: str) -> None:
        key = self._redis_key(context)
        self.redis.set(key, summary.strip())
        self.redis.expire(key, self.ttl_seconds)

    def clear(self, context: SessionContext) -> None:
        self.redis.delete(self._redis_key(context))

    def _redis_key(self, context: SessionContext) -> str:
        return f"summary:{context.key}"


def create_summary_memory() -> SummaryMemory:
    if MEMORY_BACKEND == "redis":
        return RedisSummaryMemory()
    return InMemorySummaryMemory()


summary_memory = create_summary_memory()
