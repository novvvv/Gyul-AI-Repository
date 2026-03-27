"""
SER 모델 평가 스크립트
Sample 데이터(원천 wav + 라벨 JSON)로 Wav2Vec2 감정 예측 → 정답 비교 → 시각화

실행: python evaluate.py
"""

import sys

# 깨진 tensorflow 패키지가 transformers import를 방해하므로 비활성화
for _k in list(sys.modules):
    if _k == "tensorflow" or _k.startswith("tensorflow."):
        del sys.modules[_k]

import importlib.machinery
import importlib.util

_original_find_spec = importlib.util.find_spec

def _patched_find_spec(name, *args, **kwargs):
    if name == "tensorflow" or name.startswith("tensorflow."):
        return None
    return _original_find_spec(name, *args, **kwargs)

importlib.util.find_spec = _patched_find_spec

import json
import warnings
from pathlib import Path
from collections import Counter

import soundfile as sf
from scipy.signal import resample
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager, rcParams
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

korean_fonts = ["AppleGothic", "NanumGothic", "Malgun Gothic", "Arial Unicode MS"]
installed = {f.name for f in font_manager.fontManager.ttflist}
kfont = next((f for f in korean_fonts if f in installed), None)
if kfont:
    rcParams["font.family"] = kfont
rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", message="Glyph .* missing from font")

MODEL_NAME = "superb/wav2vec2-base-superb-er"
TARGET_SR = 16000
SEGMENT_SEC = 3.0

LABEL_MAP_KO_TO_EN = {
    "분노": "angry",
    "무감정": "neutral",
    "행복": "happy",
    "슬픔": "sad",
    "혐오": "disgust",
    "공포": "fear",
    "놀람": "surprise",
}
LABEL_MAP_EN_TO_KO = {v: k for k, v in LABEL_MAP_KO_TO_EN.items()}

SER_LABEL_NORMALIZE = {
    "neu": "neutral",
    "hap": "happy",
    "ang": "angry",
    "sad": "sad",
    "dis": "disgust",
    "fea": "fear",
    "sur": "surprise",
    "neutral": "neutral",
    "happy": "happy",
    "angry": "angry",
    "sadness": "sad",
    "disgust": "disgust",
    "fear": "fear",
    "surprise": "surprise",
}


def load_model():
    print(f"모델 로딩: {MODEL_NAME}")
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return extractor, model


def load_audio(wav_path: str, target_sr: int = TARGET_SR) -> np.ndarray:
    data, sr = sf.read(wav_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        num_samples = int(len(data) * target_sr / sr)
        data = resample(data, num_samples).astype(np.float32)
    return data


def predict_wav(extractor, model, wav_path: str) -> dict:
    audio = load_audio(wav_path)

    seg_len = int(TARGET_SR * SEGMENT_SEC)
    segments = []
    for i in range(0, len(audio), seg_len):
        seg = audio[i : i + seg_len]
        if len(seg) >= int(TARGET_SR * 0.5):
            segments.append(seg)
    if not segments:
        segments = [audio]

    all_probs = []
    for seg in segments:
        inputs = extractor(seg, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
        all_probs.append(probs)

    avg_probs = np.mean(all_probs, axis=0)
    id2label = model.config.id2label
    labels = [id2label.get(i, f"label_{i}") for i in range(len(avg_probs))]
    pred_idx = int(np.argmax(avg_probs))

    raw_label = labels[pred_idx]
    norm_label = SER_LABEL_NORMALIZE.get(raw_label, raw_label)

    return {
        "pred_en": norm_label,
        "confidence": float(avg_probs[pred_idx]),
        "all_scores": {labels[i]: float(avg_probs[i]) for i in range(len(avg_probs))},
        "duration": len(audio) / TARGET_SR,
    }


def build_eval_dataset():
    label_dir = Path("Sample/02.라벨링데이터")
    wav_dir = Path("Sample/01.원천데이터")

    records = []
    for fp in sorted(label_dir.glob("*.json")):
        arr = json.loads(fp.read_text("utf-8"))
        if not arr:
            continue
        item = arr[0]
        reciter_id = (item.get("reciter") or {}).get("id")

        for s in item.get("sentences", []):
            vp = s.get("voice_piece") or {}
            style = s.get("style") or {}
            filename = vp.get("filename", "")
            emotion_ko = style.get("emotion", "")
            wav_path = wav_dir / filename

            if not wav_path.exists() or not emotion_ko:
                continue

            emotion_en = LABEL_MAP_KO_TO_EN.get(emotion_ko, emotion_ko)

            records.append({
                "wav_path": str(wav_path),
                "filename": filename,
                "gt_ko": emotion_ko,
                "gt_en": emotion_en,
                "reciter_id": reciter_id,
                "duration": vp.get("duration"),
                "text": vp.get("tr", ""),
            })

    return pd.DataFrame(records)


def run_evaluation():
    df = build_eval_dataset()
    print(f"평가 대상: {len(df)}개 발화")
    print(f"정답 분포:\n{df['gt_ko'].value_counts().to_string()}\n")

    extractor, model = load_model()

    preds_en = []
    confs = []

    for i, row in df.iterrows():
        result = predict_wav(extractor, model, row["wav_path"])
        preds_en.append(result["pred_en"])
        confs.append(result["confidence"])
        if (i + 1) % 50 == 0 or i == len(df) - 1:
            print(f"  [{i+1}/{len(df)}] 예측 완료")

    df["pred_en"] = preds_en
    df["pred_ko"] = df["pred_en"].map(LABEL_MAP_EN_TO_KO).fillna(df["pred_en"])
    df["confidence"] = confs
    df["correct"] = df["gt_en"] == df["pred_en"]

    y_true = df["gt_en"]
    y_pred = df["pred_en"]

    all_labels = sorted(set(y_true) | set(y_pred))
    all_labels_ko = [LABEL_MAP_EN_TO_KO.get(l, l) for l in all_labels]

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print("\n" + "=" * 50)
    print(f"Accuracy:    {acc:.4f}  ({acc:.1%})")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print("=" * 50)

    print("\n[Classification Report]")
    print(classification_report(y_true, y_pred, zero_division=0))

    df.to_csv("evaluation_result.csv", index=False, encoding="utf-8-sig")
    print("평가 결과 저장: evaluation_result.csv\n")

    visualize(df, y_true, y_pred, all_labels, all_labels_ko, acc, macro_f1)


def visualize(df, y_true, y_pred, labels_en, labels_ko, acc, macro_f1):
    plt.style.use("dark_background")
    sns.set_theme(
        style="dark",
        context="talk",
        rc={
            "axes.facecolor": "#111111",
            "figure.facecolor": "#0B0B0B",
            "text.color": "#F2F2F2",
        },
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 13))

    # (1) Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels_en)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="magma",
        xticklabels=labels_ko,
        yticklabels=labels_ko,
        linewidths=0.5,
        linecolor="#222",
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("Confusion Matrix")
    axes[0, 0].set_xlabel("예측")
    axes[0, 0].set_ylabel("정답")

    # (2) 정답 vs 예측 분포 비교
    gt_counts = y_true.value_counts()
    pred_counts = y_pred.value_counts()
    compare = pd.DataFrame({
        "정답": gt_counts,
        "예측": pred_counts,
    }).fillna(0).astype(int)
    compare.index = [LABEL_MAP_EN_TO_KO.get(l, l) for l in compare.index]
    compare.plot(kind="bar", ax=axes[0, 1], color=["#BB86FC", "#03DAC6"], edgecolor="none")
    axes[0, 1].set_title("정답 vs 예측 분포")
    axes[0, 1].set_ylabel("개수")
    axes[0, 1].tick_params(axis="x", rotation=30)
    axes[0, 1].legend(framealpha=0.3)

    # (3) 감정별 confidence 분포
    df_plot = df.copy()
    df_plot["gt_label"] = df_plot["gt_ko"]
    sns.boxplot(
        data=df_plot,
        x="gt_label",
        y="confidence",
        hue="correct",
        palette={True: "#4CAF50", False: "#F44336"},
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("감정별 모델 신뢰도 (맞춤/틀림)")
    axes[1, 0].set_xlabel("정답 감정")
    axes[1, 0].set_ylabel("confidence")
    axes[1, 0].tick_params(axis="x", rotation=30)
    axes[1, 0].legend(title="정답 여부", labels=["틀림", "맞춤"], framealpha=0.3)

    # (4) 요약 텍스트
    axes[1, 1].axis("off")
    total = len(df)
    correct = int(df["correct"].sum())
    wrong = total - correct

    summary_lines = [
        f"총 샘플: {total}",
        f"정답: {correct}  |  오답: {wrong}",
        "",
        f"Accuracy:    {acc:.1%}",
        f"Macro F1:    {macro_f1:.4f}",
        "",
        "정답 분포:",
    ]
    for emo, cnt in df["gt_ko"].value_counts().items():
        summary_lines.append(f"  {emo}: {cnt}")

    summary_lines.append("")
    summary_lines.append("오분류 TOP:")
    mistakes = df[~df["correct"]][["gt_ko", "pred_ko"]].value_counts().head(5)
    for (gt, pred), cnt in mistakes.items():
        summary_lines.append(f"  {gt} → {pred}: {cnt}건")

    axes[1, 1].text(
        0.05, 0.95,
        "\n".join(summary_lines),
        transform=axes[1, 1].transAxes,
        fontsize=14,
        verticalalignment="top",
        fontfamily="monospace" if not kfont else kfont,
        color="#F2F2F2",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#1A1A2E", edgecolor="#262424"),
    )
    axes[1, 1].set_title("평가 요약")

    plt.suptitle(
        f"SER 평가 결과  |  Acc={acc:.1%}  Macro-F1={macro_f1:.4f}",
        fontsize=18,
        color="#FAFAFA",
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig("evaluation_plot.png", dpi=150, bbox_inches="tight", facecolor="#0B0B0B")
    print("시각화 저장: evaluation_plot.png")
    plt.show()


if __name__ == "__main__":
    run_evaluation()
