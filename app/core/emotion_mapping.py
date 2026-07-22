"""SER 감정 → Spring 연동 계약 감정 enum 매핑 (가이드 §3.3).

계약 enum: TENSION / CONFUSION / CALM / CONFIDENCE / NEUTRAL.
합의 없는 값을 보내면 Spring이 NEUTRAL로 강등하므로 반드시 이 매핑을 거친다.

SER 모델의 sadness/angry/disgust는 대응 enum이 없어 임시로 NEUTRAL에 접는다.
자기분석 서비스에서 슬픔·분노는 핵심 신호이므로 정보 손실 보전을 위해
원본 라벨 분포를 metrics.rawEmotionCounts로 함께 발행하고, enum 확장(SADNESS 등)은
Spring 레포 integration-spec.md 스펙 PR 합의 후 이 매핑을 갱신한다.
CALM은 대응하는 SER 라벨이 없어 scores에 등장하지 않는다.
"""

from collections import defaultdict

SER_TO_CONTRACT: dict[str, str] = {
    "fear": "TENSION",
    "surprise": "CONFUSION",
    "happiness": "CONFIDENCE",
    "neutral": "NEUTRAL",
    "sadness": "NEUTRAL",
    "angry": "NEUTRAL",
    "disgust": "NEUTRAL",
}

CONTRACT_EMOTIONS = ("TENSION", "CONFUSION", "CALM", "CONFIDENCE", "NEUTRAL")


def map_ser_label(label: str) -> str:
    """단일 SER 라벨을 계약 enum으로 변환. 미지 라벨은 NEUTRAL."""
    return SER_TO_CONTRACT.get(label.strip().lower(), "NEUTRAL")


def aggregate_emotion(turns: list[dict]) -> dict:
    """턴 로그의 voice_emotion.probs를 계약 enum 분포로 집계.

    턴별 확률을 enum으로 접어 단순 평균 후 합 1로 정규화한다.
    반환: {"dominant": <enum>, "scores": {<enum 소문자>: 0~1}}
    """
    sums: dict[str, float] = defaultdict(float)
    counted = 0
    for turn in turns:
        probs = (turn.get("voice_emotion") or {}).get("probs") or {}
        if not probs:
            continue
        counted += 1
        for label, prob in probs.items():
            sums[map_ser_label(str(label))] += float(prob)

    if not counted:
        return {"dominant": "NEUTRAL", "scores": {"neutral": 1.0}}

    averaged = {enum: total / counted for enum, total in sums.items()}
    norm = sum(averaged.values()) or 1.0
    scores = {
        enum.lower(): round(value / norm, 4)
        for enum, value in averaged.items()
        if value > 0
    }
    dominant = max(averaged, key=averaged.get)  # type: ignore[arg-type]
    return {"dominant": dominant, "scores": scores}


def raw_emotion_counts(turns: list[dict]) -> dict[str, int]:
    """턴별 대표 SER 원본 라벨 카운트 — metrics.rawEmotionCounts용 (정보 보전)."""
    counts: dict[str, int] = defaultdict(int)
    for turn in turns:
        label = (turn.get("voice_emotion") or {}).get("label")
        if label:
            counts[str(label).lower()] += 1
    return dict(counts)
