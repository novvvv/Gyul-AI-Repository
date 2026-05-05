import pandas as pd
import matplotlib.pyplot as plt
from google.colab import drive

drive.mount("/content/drive")

# 여기만 본인 Drive 안 CSV 위치로 맞추면 됩니다.
# 예: 왼쪽 파일탭에서 5차년도_2차.csv 우클릭 -> 경로 복사
CSV_PATH = "/content/drive/MyDrive/Colab Notebooks/kyul_stt/data/5_csv/5차년도_2차.csv"

df = pd.read_csv(CSV_PATH, encoding="cp949")

print("전체 행 수:", len(df))
print("컬럼:", list(df.columns))
display(df.head())

print("\n상황 분포")
display(df["상황"].value_counts(dropna=False).to_frame("count"))

print("\n상황 비율(%)")
display((df["상황"].value_counts(normalize=True, dropna=False) * 100).round(2).to_frame("pct"))

df["상황"].value_counts(dropna=False).plot(kind="bar", figsize=(10, 4), title="상황 분포")
plt.xlabel("상황")
plt.ylabel("count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
