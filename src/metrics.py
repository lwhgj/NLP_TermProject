# src/metrics.py
import os
import numpy as np
import matplotlib.pyplot as plt
import evaluate
from transformers import PreTrainedTokenizerBase

# HF evaluate의 rouge metric
rouge_metric = evaluate.load("rouge")


def build_compute_metrics(tokenizer: PreTrainedTokenizerBase):
    """
    Seq2SeqTrainer용 metric 함수
    → ROUGE-1 / ROUGE-2 / ROUGE-L만 계산
    """

    def compute_metrics(eval_pred):
        preds, labels = eval_pred

        if isinstance(preds, tuple):
            preds = preds[0]

        # -100 → pad_token_id 로 되돌려서 디코딩
        labels_proc = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels_proc, skip_special_tokens=True)

        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        rouge_result = rouge_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )

        # 🔧 evaluate 버전에 따라 출력 형식이 달라져도 다 처리되도록 헬퍼 함수 정의
        def get_f1(score):
            # 예전: Score 객체( score.mid.fmeasure )
            if hasattr(score, "mid"):
                return score.mid.fmeasure
            # dict 형태: {"fmeasure": ..., "precision": ..., ...} 등
            if isinstance(score, dict):
                if "fmeasure" in score:
                    return score["fmeasure"]
                if "f" in score:  # 혹시 이렇게 오는 경우
                    return score["f"]
            # 이미 float / numpy.float64 인 경우
            return float(score)

        rouge1 = get_f1(rouge_result["rouge1"])
        rouge2 = get_f1(rouge_result["rouge2"])
        rougel = get_f1(rouge_result["rougeL"])

        return {
            "rouge1": rouge1,
            "rouge2": rouge2,
            "rougeL": rougel,
        }

    return compute_metrics


def plot_metrics_from_history(trainer, log_dir: str):
    """
    trainer.state.log_history에서 eval_* 로그 뽑아서
    ROUGE-1 / 2 / L 그래프 저장
    """
    history = trainer.state.log_history
    epochs = []
    r1_list, r2_list, rl_list = [], [], []

    for log in history:
        if "eval_rouge1" in log:
            ep = log.get("epoch")
            if ep is None:
                continue
            epochs.append(ep)
            r1_list.append(log.get("eval_rouge1", float("nan")))
            r2_list.append(log.get("eval_rouge2", float("nan")))
            rl_list.append(log.get("eval_rougeL", float("nan")))

    if not epochs:
        print("[WARN] eval 로그가 없어 그래프 못 만듦.")
        return

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, r1_list, marker="o", label="ROUGE-1")
    plt.plot(epochs, r2_list, marker="o", label="ROUGE-2")
    plt.plot(epochs, rl_list, marker="o", label="ROUGE-L")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation ROUGE scores per Epoch")
    plt.grid(True)
    plt.legend()

    os.makedirs(log_dir, exist_ok=True)
    out_path = os.path.join(log_dir, "val_metrics.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Validation metric plot saved to {out_path}")

def plot_loss_from_history(trainer, log_dir: str):
    """
    Trainer log_history에서 train_loss / eval_loss를 추출하여 PNG로 저장
    """
    history = trainer.state.log_history

    epochs = []
    train_losses = []
    val_losses = []

    cur_train_loss = None

    for log in history:
        # train 손실
        if "loss" in log and "epoch" in log:
            cur_train_loss = log["loss"]
            # 같은 epoch 안에서 여러 step이 있을 수 있음 → 가장 마지막 loss 사용
            train_losses.append((log["epoch"], cur_train_loss))

        # validation 손실
        if "eval_loss" in log and "epoch" in log:
            val_losses.append((log["epoch"], log["eval_loss"]))

    if not train_losses and not val_losses:
        print("[WARN] loss 로그가 없어 그래프 생성 불가.")
        return

    # epoch / loss 분리
    train_epochs = [ep for ep, _ in train_losses]
    train_loss_values = [ls for _, ls in train_losses]

    val_epochs = [ep for ep, _ in val_losses]
    val_loss_values = [ls for _, ls in val_losses]

    # ------------ PLOT ------------
    plt.figure(figsize=(8, 6))
    if train_losses:
        plt.plot(train_epochs, train_loss_values, marker="o", label="Train Loss")
    if val_losses:
        plt.plot(val_epochs, val_loss_values, marker="o", label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train / Validation Loss per Epoch")
    plt.legend()
    plt.grid(True)

    out_path = os.path.join(log_dir, "loss_curve.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    print(f"[INFO] Loss plot saved to {out_path}")
