from __future__ import annotations

from collections import Counter

from scripts.session_report.schema import SessionSnapshot, SessionTurn


def _voice_label(turn: SessionTurn) -> str | None:
    return turn.voice_emotion.label if turn.voice_emotion else None


def _face_label(turn: SessionTurn) -> str | None:
    return turn.face_emotion.label if turn.face_emotion else None


def build_aggregates(snapshot: SessionSnapshot) -> dict:
    voice_labels = [_voice_label(t) for t in snapshot.turns if _voice_label(t)]
    face_labels = [_face_label(t) for t in snapshot.turns if _face_label(t)]

    voice_counter = Counter(voice_labels)
    face_counter = Counter(face_labels)

    shifts: list[str] = []
    prev: str | None = None
    for label in voice_labels:
        if prev and label != prev:
            shifts.append(f"{prev}→{label}")
        prev = label

    mismatch_count = 0
    mismatch_notes: list[str] = []
    for i, turn in enumerate(snapshot.turns, start=1):
        voice = _voice_label(turn)
        face = _face_label(turn)
        if voice and face and voice != face:
            mismatch_count += 1
            mismatch_notes.append(
                f"{i}번째 발화: 음성 {voice} / 표정 {face}"
            )

    user_turn_count = sum(1 for t in snapshot.turns if t.user_text)
    bot_turn_count = sum(1 for t in snapshot.turns if t.bot_reply)

    return {
        "user_turn_count": user_turn_count,
        "bot_turn_count": bot_turn_count,
        "voice_dominant": [label for label, _ in voice_counter.most_common(3)],
        "face_dominant": [label for label, _ in face_counter.most_common(3)],
        "voice_counts": dict(voice_counter),
        "face_counts": dict(face_counter),
        "emotion_shifts": shifts,
        "mismatch_count": mismatch_count,
        "mismatch_notes": mismatch_notes[:5],
    }
