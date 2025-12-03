import os
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase


def load_train_val_datasets(train_path: str, val_ratio: float, seed: int):
    """
    train_path: text, summary 컬럼을 가진 CSV 또는 JSON/JSONL 파일 경로
    val_ratio 비율로 train/val split 한 HF Dataset 반환
    """
    ext = os.path.splitext(train_path)[1].lower()

    # 확장자에 따라 csv / json 자동 선택
    if ext in [".json", ".jsonl"]:
        raw = load_dataset("json", data_files={"data": train_path})["data"]
    else:
        # 기본은 csv 취급
        raw = load_dataset("csv", data_files={"data": train_path})["data"]

    dataset = raw.train_test_split(test_size=val_ratio, seed=seed)
    train_ds = dataset["train"]
    val_ds = dataset["test"]
    return train_ds, val_ds


def make_preprocess_fn(
    tokenizer: PreTrainedTokenizerBase,
    input_max_len: int,
    target_max_len: int,
):
    """
    text → input_ids, summary → labels 변환.
    pad 토큰은 -100으로 바꿔서 loss에서 무시.
    """

    def preprocess(batch):
        model_inputs = tokenizer(
            batch["text"],
            max_length=input_max_len,
            padding="max_length",
            truncation=True,
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["summary"],
                max_length=target_max_len,
                padding="max_length",
                truncation=True,
            )["input_ids"]

        labels = [
            [(lid if lid != tokenizer.pad_token_id else -100) for lid in seq]
            for seq in labels
        ]
        model_inputs["labels"] = labels
        return model_inputs

    return preprocess
