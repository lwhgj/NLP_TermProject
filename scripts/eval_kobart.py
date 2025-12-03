import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ===== 프로젝트 루트를 sys.path에 추가 =====
# eval_kobart.py (scripts 폴더 안) 기준으로 상위 폴더가 프로젝트 루트
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
# ==========================================

from src.data_module import make_preprocess_fn
from src.metrics import build_compute_metrics

CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")



def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def to_abs(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)


def main():
    # ===== config.yaml 불러오기 =====
    cfg = load_config(CONFIG_PATH)

    # max length / batch size
    INPUT_MAX_LEN = cfg["max_length"]["input_max_len"]
    TARGET_MAX_LEN = cfg["max_length"]["target_max_len"]
    TRAIN_BS = cfg["training"]["train_batch_size"]
    BATCH_SIZE = min(TRAIN_BS * 2, 8)   # 평가용은 보통 train보다 조금 키워도 됨

    # 🔥 generate 설정 (없으면 기본값 사용)
    gen_cfg = cfg.get("generate", {})
    GEN_MAX_NEW_TOKENS = gen_cfg.get("max_new_tokens", TARGET_MAX_LEN)
    GEN_MIN_LENGTH = gen_cfg.get("min_length", 30)
    GEN_NUM_BEAMS = gen_cfg.get("num_beams", 4)
    GEN_LENGTH_PENALTY = gen_cfg.get("length_penalty", 1.5)
    GEN_NO_REPEAT_NGRAM = gen_cfg.get("no_repeat_ngram_size", 3)

    # ✅ 경로는 YAML에서 읽기
    paths_cfg = cfg.get("paths", {})
    MODEL_DIR_RAW = paths_cfg.get("model_dir")
    TEST_CSV_RAW = paths_cfg.get("test_csv")

    if not MODEL_DIR_RAW:
        print("[ERROR] config.paths.model_dir 이(가) 비어 있습니다.")
        return
    if not TEST_CSV_RAW:
        print("[ERROR] config.paths.test_csv 이(가) 비어 있습니다.")
        return

    MODEL_DIR = to_abs(MODEL_DIR_RAW)
    TEST_CSV = to_abs(TEST_CSV_RAW)

    print(f"[INFO] CONFIG loaded")
    print(f" - input_max_len       = {INPUT_MAX_LEN}")
    print(f" - target_max_len      = {TARGET_MAX_LEN}")
    print(f" - train_batch         = {TRAIN_BS}")
    print(f" - eval_batch          = {BATCH_SIZE}")
    print(f" - model_dir           = {MODEL_DIR}")
    print(f" - test_csv            = {TEST_CSV}")
    print(f" - gen.max_new_tokens  = {GEN_MAX_NEW_TOKENS}")
    print(f" - gen.min_length      = {GEN_MIN_LENGTH}")
    print(f" - gen.num_beams       = {GEN_NUM_BEAMS}")
    print(f" - gen.length_penalty  = {GEN_LENGTH_PENALTY}")
    print(f" - gen.no_repeat_ngram = {GEN_NO_REPEAT_NGRAM}")

    # ===== 경로 체크 =====
    if not os.path.exists(MODEL_DIR):
        print(f"[ERROR] model_dir 없음: {MODEL_DIR}")
        return
    if not os.path.exists(TEST_CSV):
        print(f"[ERROR] test_csv 없음: {TEST_CSV}")
        return

    # ===== 데이터 로드 =====
    df = pd.read_csv(TEST_CSV)
    if "text" not in df.columns or "summary" not in df.columns:
        print("[ERROR] test_csv에는 'text', 'summary' 컬럼이 필요합니다.")
        return

    test_ds = Dataset.from_pandas(df)

    # ===== 모델 로드 =====
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # ===== 전처리 =====
    preprocess_fn = make_preprocess_fn(
        tokenizer,
        input_max_len=INPUT_MAX_LEN,
        target_max_len=TARGET_MAX_LEN,
    )
    test_ds_token = test_ds.map(
        preprocess_fn,
        batched=True,
        remove_columns=test_ds.column_names,
    )

    # 텐서 포맷
    test_ds_token.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    loader = DataLoader(test_ds_token, batch_size=BATCH_SIZE)

    all_preds, all_labels = [], []

    print("[INFO] 테스트 셋 요약 생성 중...")

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].numpy()

        with torch.no_grad():
            gen_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                # 🔥 기존: max_length=TARGET_MAX_LEN
                # → 이제 config.generate 기반으로 생성 길이/스타일 통일
                max_new_tokens=GEN_MAX_NEW_TOKENS,
                min_length=GEN_MIN_LENGTH,
                num_beams=GEN_NUM_BEAMS,
                length_penalty=GEN_LENGTH_PENALTY,
                no_repeat_ngram_size=GEN_NO_REPEAT_NGRAM,
            )

        all_preds.append(gen_ids.cpu().numpy())
        all_labels.append(labels)

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    compute_metrics = build_compute_metrics(tokenizer)
    metrics = compute_metrics((all_preds, all_labels))

    print("\n========== TEST METRICS ==========")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print("==================================")


if __name__ == "__main__":
    main()
