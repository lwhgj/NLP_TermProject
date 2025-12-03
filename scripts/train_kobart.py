# scripts/train_kobart.py
# KoBART 파인튜닝: train/val, ROUGE-1/2/L, best(eval_loss) 모델 저장 + 그래프 + CSV 로그

import os
import json
import argparse
import time
import sys
import yaml
import torch
import csv  # ✅ CSV 저장용

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

# ===== BASE_DIR 설정 =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.data_module import load_train_val_datasets, make_preprocess_fn
from src.metrics import (
    build_compute_metrics,
    plot_metrics_from_history,
    plot_loss_from_history,
)


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_global_seed(seed: int):
    set_seed(seed)


def create_log_dir(root: str):
    if not os.path.isabs(root):
        root = os.path.join(BASE_DIR, root)

    os.makedirs(root, exist_ok=True)
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(root, ts)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def to_abs(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(BASE_DIR, "config", "config.yaml"),
        help="YAML config path",
    )
    args = parser.parse_args()

    # ===== config 로드 =====
    cfg = load_config(args.config)

    model_name = cfg["model"]["name"]
    train_csv = to_abs(cfg["data"]["train_csv"])
    val_ratio = cfg["data"]["val_ratio"]

    num_epochs = cfg["training"]["num_train_epochs"]
    train_bs = cfg["training"]["train_batch_size"]
    lr = cfg["training"]["learning_rate"]
    weight_decay = cfg["training"]["weight_decay"]
    warmup_ratio = cfg["training"]["warmup_ratio"]

    input_max_len = cfg["max_length"]["input_max_len"]
    target_max_len = cfg["max_length"]["target_max_len"]

    log_root = cfg["logging"]["log_root"]
    seed = cfg.get("seed", 42)

    # 🔥 generate 섹션 읽기 (없으면 기본값 사용)
    gen_cfg = cfg.get("generate", {})
    # 학습/평가 시 generate 길이
    gen_max_len = gen_cfg.get("max_new_tokens", target_max_len)
    gen_num_beams = gen_cfg.get("num_beams", 4)
    gen_length_penalty = gen_cfg.get("length_penalty", 1.0)
    gen_no_repeat_ngram = gen_cfg.get("no_repeat_ngram_size", 3)

    set_global_seed(seed)

    eval_bs = max(1, min(train_bs * 2, 8))
    print(f"[INFO] train_batch_size={train_bs}, eval_batch_size={eval_bs}")
    print(f"[INFO] train_csv={train_csv}")
    print(
        f"[INFO] generation_max_length={gen_max_len}, num_beams={gen_num_beams}, "
        f"length_penalty={gen_length_penalty}, no_repeat_ngram_size={gen_no_repeat_ngram}"
    )

    log_dir = create_log_dir(log_root)
    print(f"[INFO] Log dir: {log_dir}")

    # 사용한 config 백업
    with open(os.path.join(log_dir, "config_used.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)

    # ===== 데이터 =====
    train_ds, val_ds = load_train_val_datasets(train_csv, val_ratio, seed)

    # ===== 모델 & 토크나이저 =====
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # 🔥 GenerationConfig 기반으로 generate 설정을 정합하게 맞추기
    # (num_beams, length_penalty, no_repeat_ngram_size, max_length 등)
    gen_config = model.generation_config
    gen_config.max_length = gen_max_len
    gen_config.num_beams = gen_num_beams
    gen_config.length_penalty = gen_length_penalty
    gen_config.no_repeat_ngram_size = gen_no_repeat_ngram
    model.generation_config = gen_config

    preprocess_fn = make_preprocess_fn(tokenizer, input_max_len, target_max_len)
    train_ds_token = train_ds.map(
        preprocess_fn,
        batched=True,
        remove_columns=train_ds.column_names,
    )
    val_ds_token = val_ds.map(
        preprocess_fn,
        batched=True,
        remove_columns=val_ds.column_names,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    compute_metrics = build_compute_metrics(tokenizer)

    # ===== 학습 설정 (best model = validation loss 최소) =====
    training_args = Seq2SeqTrainingArguments(
        output_dir=log_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        learning_rate=lr,
        weight_decay=weight_decay,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        warmup_ratio=warmup_ratio,
        predict_with_generate=True,
        # 🔥 학습/평가 시 생성 길이도 config.generate를 최대한 반영
        generation_max_length=gen_max_len,
        generation_num_beams=gen_num_beams,
        logging_dir=log_dir,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  # ✅ eval_loss 기준
        greater_is_better=False,            # ✅ loss는 작을수록 좋음
        fp16=torch.cuda.is_available(),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds_token,
        eval_dataset=val_ds_token,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # ===== 학습 =====
    train_result = trainer.train()

    # ===== 로그 CSV 저장 =====
    csv_path = os.path.join(log_dir, "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "epoch",
                "loss",
                "eval_loss",
                "eval_rouge1",
                "eval_rouge2",
                "eval_rougeL",
                "eval_rougeLsum",
            ]
        )

        for entry in trainer.state.log_history:
            writer.writerow(
                [
                    entry.get("step"),
                    entry.get("epoch"),
                    entry.get("loss"),
                    entry.get("eval_loss"),
                    entry.get("eval_rouge1"),
                    entry.get("eval_rouge2"),
                    entry.get("eval_rougeL"),
                    entry.get("eval_rougeLsum"),
                ]
            )

    print(f"[INFO] metrics.csv saved: {csv_path}")

    # ===== best 모델 저장 (validation loss 기준) =====
    best_model_dir = os.path.join(log_dir, "best_model")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    with open(os.path.join(log_dir, "train_result.json"), "w", encoding="utf-8") as f:
        json.dump(train_result.metrics, f, indent=2, ensure_ascii=False)

    # ===== 그래프 저장 (ROUGE, loss) =====
    plot_metrics_from_history(trainer, log_dir)  # validation metric 그래프 (ROUGE-1/2/L)
    plot_loss_from_history(trainer, log_dir)     # train / eval loss 그래프

    print("[DONE] Training complete.")
    print(f"[BEST MODEL] {best_model_dir}")


if __name__ == "__main__":
    main()
