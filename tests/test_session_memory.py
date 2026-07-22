import pytest

from app.memory.session_memory import (
    ConversationTurn,
    InMemorySessionMemory,
    RedisSessionMemory,
    SessionContext,
    create_session_memory,
)

CONTEXT = SessionContext(
    user_id="ai-test@example.com", session_id="s1", persona_id="gyul"
)


class TestInMemorySessionMemory:
    def test_append_and_recent(self):
        memory = InMemorySessionMemory(max_turns=20)
        memory.append_user_message(CONTEXT, "안녕하세요", "fear")
        memory.append_assistant_message(CONTEXT, "안녕하세요, 반가워요")

        turns = memory.recent_messages(CONTEXT)
        assert turns == [
            ConversationTurn("user", "안녕하세요", "fear"),
            ConversationTurn("assistant", "안녕하세요, 반가워요"),
        ]

    def test_max_turns_trims_oldest(self):
        memory = InMemorySessionMemory(max_turns=2)
        for i in range(3):
            memory.append_user_message(CONTEXT, f"메시지{i}")

        turns = memory.recent_messages(CONTEXT)
        assert [t.content for t in turns] == ["메시지1", "메시지2"]

    def test_clear(self):
        memory = InMemorySessionMemory()
        memory.append_user_message(CONTEXT, "안녕하세요")
        memory.clear(CONTEXT)
        assert memory.recent_messages(CONTEXT) == []


class TestRedisSessionMemory:
    @pytest.fixture
    def memory(self):
        fakeredis = pytest.importorskip("fakeredis")
        mem = RedisSessionMemory(max_turns=20, ttl_seconds=86400)
        mem.redis = fakeredis.FakeRedis(decode_responses=True)
        return mem

    def test_round_trip(self, memory):
        memory.append_user_message(CONTEXT, "안녕하세요", "fear")
        memory.append_assistant_message(CONTEXT, "반가워요")

        turns = memory.recent_messages(CONTEXT)
        assert [t.role for t in turns] == ["user", "assistant"]
        assert turns[0].emotion == "fear"

    def test_key_pattern_matches_contract(self, memory):
        # 가이드 §4 키 패턴: chat:{email}:{sessionId}:{persona}
        memory.append_user_message(CONTEXT, "안녕하세요")
        keys = memory.redis.keys("*")
        assert keys == ["chat:ai-test@example.com:s1:gyul"]

    def test_ttl_is_set(self, memory):
        memory.append_user_message(CONTEXT, "안녕하세요")
        ttl = memory.redis.ttl("chat:ai-test@example.com:s1:gyul")
        assert 0 < ttl <= 86400

    def test_max_turns_trims_oldest(self, memory):
        memory.max_turns = 2
        for i in range(3):
            memory.append_user_message(CONTEXT, f"메시지{i}")

        turns = memory.recent_messages(CONTEXT)
        assert [t.content for t in turns] == ["메시지1", "메시지2"]

    def test_corrupt_entry_skipped(self, memory):
        memory.append_user_message(CONTEXT, "정상 메시지")
        memory.redis.rpush("chat:ai-test@example.com:s1:gyul", "not-json")

        turns = memory.recent_messages(CONTEXT)
        assert [t.content for t in turns] == ["정상 메시지"]

    def test_clear(self, memory):
        memory.append_user_message(CONTEXT, "안녕하세요")
        memory.clear(CONTEXT)
        assert memory.recent_messages(CONTEXT) == []


def test_factory_defaults_to_in_memory(monkeypatch):
    memory = create_session_memory()
    assert isinstance(memory, InMemorySessionMemory)
