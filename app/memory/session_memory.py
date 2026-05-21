from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Lock


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


class InMemorySessionMemory:
    def __init__(self, max_turns: int = 20) -> None:
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


session_memory = InMemorySessionMemory()
