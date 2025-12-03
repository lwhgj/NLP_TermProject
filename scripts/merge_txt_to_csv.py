import os
import csv

# --- 설정: TXT들이 모여 있는 폴더 / 출력 CSV 경로 ---
TXT_DIR = r"C:\Users\inwoo\PycharmProjects\NLP\NLP_TermProject\data\test_data\test_input"
OUTPUT_CSV = r"C:\Users\inwoo\PycharmProjects\NLP\NLP_TermProject\data\test_data\test_input_data.csv"


def main():
    if not os.path.exists(TXT_DIR):
        print(f"Error: TXT directory does not exist: {TXT_DIR}")
        return

    txt_files = [f for f in os.listdir(TXT_DIR) if f.lower().endswith(".txt")]
    txt_files.sort()
    print("TXT file count:", len(txt_files))

    if not txt_files:
        print("No TXT files found.")
        return

    rows = []

    for fname in txt_files:
        path = os.path.join(TXT_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                text = f.read().strip()
        except UnicodeDecodeError:
            # 혹시 인코딩 꼬였을 때 대비
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()

        rows.append({
            "filename": fname,  # 예: LG유플러스_2016_summary.txt
            "text": text,
        })
        print(f" -> Loaded: {fname} (len={len(text)})")

    # CSV로 저장
    try:
        with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "text"])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

        print("\nDONE:", OUTPUT_CSV)
        print(f"Total rows: {len(rows)}")

    except PermissionError:
        print("\n[CRITICAL ERROR] 파일 저장 실패!")
        print(f" -> '{OUTPUT_CSV}' 파일이 켜져 있습니다. 끄고 다시 실행하세요.")


if __name__ == "__main__":
    main()
