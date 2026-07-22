import numpy as np

from app.memory.session_memory import SessionContext
from app.ws.pipeline import (
    UtteranceAssembler,
    parse_session_context,
    parse_utterance_text,
)

SR = 16000


def _pcm_bytes(amplitude: int, seconds: float) -> bytes:
    samples = np.full(int(SR * seconds), amplitude, dtype=np.int16)
    return samples.tobytes()


class TestUtteranceAssembler:
    def _assembler(self) -> UtteranceAssembler:
        return UtteranceAssembler(
            target_sr=SR,
            min_chunk_seconds=1.0,
            rms_threshold=0.01,
            min_speech_seconds=0.4,
            silence_seconds=0.7,
        )

    def test_silence_only_never_completes(self):
        assembler = self._assembler()
        assembler.feed(_pcm_bytes(0, 2.0))
        assert assembler.pop_completed_utterance() is None

    def test_speech_then_silence_completes_utterance(self):
        assembler = self._assembler()
        assembler.feed(_pcm_bytes(3000, 0.5))
        assert assembler.pop_completed_utterance() is None  # 아직 침묵 부족

        assembler.feed(_pcm_bytes(0, 0.8))
        utterance = assembler.pop_completed_utterance()
        assert utterance is not None
        # 발화 + 침묵 구간이 함께 담긴다 (기존 /ws/predict 동작과 동일)
        assert len(utterance) == int(SR * 1.3)
        # 완결 후 상태 리셋
        assert assembler.pop_completed_utterance() is None
        assert not assembler.speech_active

    def test_short_speech_finalizes_with_silence_padding(self):
        # 발화 버퍼에는 침묵 구간도 포함되므로(원본 /ws/predict 동작 패리티),
        # 짧은 발화도 침묵과 합쳐 min_speech를 넘으면 완결된다.
        assembler = self._assembler()
        assembler.feed(_pcm_bytes(3000, 0.2))
        assembler.feed(_pcm_bytes(0, 1.0))
        utterance = assembler.pop_completed_utterance()
        assert utterance is not None
        assert len(utterance) == int(SR * 1.2)

    def test_flush_returns_pending_speech(self):
        assembler = self._assembler()
        assembler.feed(_pcm_bytes(3000, 0.5))
        utterance = assembler.flush()
        assert utterance is not None
        assert len(utterance) == int(SR * 0.5)
        assert assembler.flush() is None  # 리셋 확인

    def test_flush_discards_short_speech(self):
        assembler = self._assembler()
        assembler.feed(_pcm_bytes(3000, 0.2))
        assert assembler.flush() is None

    def test_partial_window_requires_min_chunk(self):
        assembler = self._assembler()
        assembler.feed(_pcm_bytes(1000, 0.5))
        assert assembler.partial_window() is None

        assembler.feed(_pcm_bytes(1000, 0.6))
        window = assembler.partial_window()
        assert window is not None
        assert len(window) == assembler.min_samples

    def test_stream_buffer_stays_bounded(self):
        assembler = self._assembler()
        for _ in range(10):
            assembler.feed(_pcm_bytes(0, 1.0))
        assert len(assembler.stream_buffer) <= assembler.min_samples


class TestParsers:
    def test_parse_utterance_text(self):
        assert (
            parse_utterance_text('{"type": "utterance_text", "text": " 안녕 "}')
            == "안녕"
        )
        assert parse_utterance_text('{"type": "utterance_text", "text": ""}') is None
        assert parse_utterance_text("not-json") is None
        assert parse_utterance_text('{"type": "other"}') is None

    def test_parse_session_context_overrides(self):
        current = SessionContext("u", "s", "default")
        updated = parse_session_context(
            '{"type": "session_start", "session_id": "s2", "persona_id": "gyul"}',
            current,
        )
        assert updated == SessionContext("u", "s2", "gyul")

    def test_parse_session_context_ignores_other_messages(self):
        current = SessionContext("u", "s", "default")
        assert parse_session_context("flush", current) is None
        assert parse_session_context('{"type": "utterance_text"}', current) is None
