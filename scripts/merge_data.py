# filename: merge_data.py
# 기능:
#   1) raw_texts.csv(filename, text) 읽기
#   2) consolidated_summaries.csv(파일명, 파일 내용) 읽기
#   3) filename == 파일명 기준으로 inner join
#   4) train_kobart.csv(filename, text, summary) 생성

import os
import unicodedata
import pandas as pd

# ================== 사용자 설정 ==================
BASE_DIR = r"C:\Users\inwoo\PycharmProjects\NLP\NLP_TermProject\data"

RAW_TEXT_CSV = os.path.join(BASE_DIR, "test_data", "test_input_data.csv")
SUMMARIES_CSV = os.path.join(BASE_DIR, "test_data", "consolidated_summaries_test.csv")
TRAIN_OUTPUT_CSV = os.path.join(BASE_DIR, "test_data", "test_kobart.csv")
# =================================================


def norm(s: str) -> str:
    """파일명 비교용 정규화 (공백 제거 + 유니코드 정규화)."""
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = unicodedata.normalize("NFC", s)
    return s


def to_key(fn: str) -> str:
    """
    raw / summary 공통으로 쓸 '조인 키' 생성 함수.
    예)
      - '2015_LG유플러스_preprocessing.txt'
      - '2015_LG유플러스_preprocessing_summary.txt'
      -> '2015_LG유플러스_preprocessing'
    """
    if not isinstance(fn, str):
        return ""

    fn = norm(fn)

    # 혹시 경로가 섞여 있으면 파일명만 추출
    fn = os.path.basename(fn)

    # 확장자 .txt 제거
    if fn.lower().endswith(".txt"):
        fn = fn[:-4]

    # 끝에 _summary 붙어 있으면 제거
    if fn.endswith("_summary"):
        fn = fn[:-8]

    return fn


def main():
    print("=== Step2: raw_texts.csv + consolidated_summaries.csv -> train_kobart.csv ===")

    if not os.path.exists(RAW_TEXT_CSV):
        print(f"[ERROR] RAW_TEXT_CSV 파일이 없습니다: {RAW_TEXT_CSV}")
        return

    if not os.path.exists(SUMMARIES_CSV):
        print(f"[ERROR] SUMMARIES_CSV 파일이 없습니다: {SUMMARIES_CSV}")
        return

    # 1) raw_texts 로드
    raw_df = pd.read_csv(RAW_TEXT_CSV, encoding="utf-8-sig")
    print(f"  - raw_texts 행 수: {len(raw_df)}")
    print("  - raw_texts filename 예시:", list(raw_df["filename"].head()))

    # 2) summaries 로드
    try:
        sum_df = pd.read_csv(SUMMARIES_CSV, encoding="utf-8")
    except UnicodeDecodeError:
        sum_df = pd.read_csv(SUMMARIES_CSV, encoding="cp949")

    print(f"  - summaries 행 수: {len(sum_df)}")
    print("  - summaries '파일명' 예시:", list(sum_df["파일명"].head()))

    # 3) 컬럼 체크
    if "파일명" not in sum_df.columns or "파일 내용" not in sum_df.columns:
        print("[ERROR] consolidated_summaries.csv에 '파일명' 또는 '파일 내용' 컬럼이 없습니다.")
        print("       현재 컬럼들:", list(sum_df.columns))
        return

    # 4) filename에서 공통 key 만들기
    raw_df["key"] = raw_df["filename"].apply(to_key)
    sum_df["key"] = sum_df["파일명"].apply(to_key)

    print("  - raw key 예시:", list(raw_df["key"].head()))
    print("  - sum key 예시:", list(sum_df["key"].head()))

    # 혹시 중복 있으면 한 개만 사용
    raw_df = raw_df.drop_duplicates(subset=["key"])
    sum_df = sum_df.drop_duplicates(subset=["key"])

    # 5) inner join
    merged = pd.merge(
        raw_df,
        sum_df[["key", "파일 내용"]],
        on="key",
        how="inner",
    )

    print(f"[STEP2] merge 후 데이터 수: {len(merged)}개")

    if len(merged) == 0:
        print("[ERROR] key 기준으로 매칭되는 게 없습니다.")
        print("  raw key 예시:", list(raw_df["key"].head()))
        print("  sum key 예시:", list(sum_df["key"].head()))
        return

    # 6) 최종 학습용 CSV 저장
    out_df = pd.DataFrame(
        {
            "filename": merged["filename"],   # 원본 텍스트 파일명 (preprocessing.txt)
            "text": merged["text"],           # PDF 풀텍스트
            "summary": merged["파일 내용"],   # 요약문
        }
    )

    out_df.to_csv(TRAIN_OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[STEP2 DONE] KoBART 학습용 CSV 생성 완료.")
    print(f"             -> {TRAIN_OUTPUT_CSV}")

    print("\n[DEBUG] 매칭 예시 3개:")
    print(out_df.head(3))


if __name__ == "__main__":
    main()
