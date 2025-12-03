# scripts/infer_kobart.py
# 학습된 KoBART(best_model)로 새로운 문서 요약
# → pdf / txt / csv 전부 입력 가능

import os
import argparse

import pdfplumber
import torch
import yaml
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def to_abs(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)


# ===== 파일 타입별 텍스트 추출 =====
def extract_text_from_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()

    # 1) PDF
    if ext == ".pdf":
        texts = []
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                page_text = page_text.strip()
                if not page_text:
                    continue
                texts.append(f"[PAGE {i+1}]\n{page_text}")
        return "\n\n".join(texts)

    # 2) TXT
    elif ext == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()

    # 3) CSV → 일단 전체 파일을 문자열로 읽어서 사용
    elif ext == ".csv":
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()

    else:
        raise ValueError(f"[ERROR] 지원하지 않는 파일 확장자입니다: {ext} (pdf/txt/csv만 지원)")


def summarize(
    model,
    tokenizer,
    text: str,
    max_input_len: int,
    max_new_tokens: int,
    min_length: int,
    num_beams: int,
    length_penalty: float,
    no_repeat_ngram_size: int,
) -> str:
    """
    실제 요약 생성 함수.
    max_new_tokens, length_penalty 등으로 길이/스타일 제어.
    """
    device = model.device

    inputs = tokenizer(
        text,
        max_length=max_input_len,
        truncation=True,
        return_tensors="pt",
    )

    allowed_keys = ["input_ids", "attention_mask"]
    inputs = {k: v.to(device) for k, v in inputs.items() if k in allowed_keys}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,      # 🔥 새로 생성할 토큰 수 기준
            min_length=min_length,              # 너무 짧은 요약 방지
            num_beams=num_beams,
            length_penalty=length_penalty,      # 🔥 1.0보다 크면 더 짧게 요약하는 쪽 선호
            no_repeat_ngram_size=no_repeat_ngram_size,
        )

    summary = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return summary.strip()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(BASE_DIR, "config", "config.yaml"),
        help="YAML config path",
    )

    # model_dir / 입력 파일 경로
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="train_kobart.py에서 저장된 best_model 디렉토리 (미지정 시 config.paths.model_dir 사용)",
    )
    parser.add_argument(
        "--pdf",
        type=str,   # 이름은 pdf지만, 이제 pdf/txt/csv 전부 가능
        default=None,
        help="요약할 파일 경로 (pdf/txt/csv). 지정 안 하면 config에 infer_pdf가 있을 때 그걸 사용",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="직접 넣는 원문 텍스트 (파일 대신)",
    )

    # generate 관련 하이퍼파라미터 (None이면 config.yaml 값 사용)
    parser.add_argument(
        "--max_input_len",
        type=int,
        default=None,
        help="입력 토큰 최대 길이 (None이면 config.max_length.input_max_len 또는 generate.max_input_len)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="모델이 새로 생성할 최대 토큰 수 (None이면 config.generate.max_new_tokens 또는 target_max_len)",
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=None,
        help="요약 최소 토큰 길이 (None이면 config.generate.min_length 또는 30)",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help="beam search의 beam 수 (None이면 config.generate.num_beams 또는 4)",
    )
    parser.add_argument(
        "--length_penalty",
        type=float,
        default=None,
        help="길이 패널티 (1.0보다 크면 짧게, 작으면 길게 생성하는 경향)",
    )
    parser.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=None,
        help="반복 방지를 위한 n-gram 크기 (None이면 config.generate.no_repeat_ngram_size 또는 3)",
    )

    args = parser.parse_args()

    # ===== config 로드 =====
    cfg = load_config(args.config)

    # ----- model_dir -----
    if args.model_dir is not None:
        model_dir = to_abs(args.model_dir)
    else:
        # config.paths.model_dir 사용
        model_dir_cfg = cfg.get("paths", {}).get("model_dir", None)
        if model_dir_cfg is None:
            print("[ERROR] model_dir이 지정되지 않았고, config.paths.model_dir도 없습니다.")
            return
        model_dir = to_abs(model_dir_cfg)

    if not os.path.exists(model_dir):
        print(f"[ERROR] model_dir 없음: {model_dir}")
        return

    # ----- 요약할 텍스트 준비 -----
    if args.text is not None:
        src_text = args.text
    else:
        # 파일 경로 결정: 인자가 우선, 없으면 config.paths.infer_pdf(있을 때만)
        file_path = None
        if args.pdf is not None:
            file_path = to_abs(args.pdf)
        else:
            infer_pdf_cfg = cfg.get("paths", {}).get("infer_pdf", None)
            if infer_pdf_cfg is not None:
                file_path = to_abs(infer_pdf_cfg)

        if file_path is None:
            print("[ERROR] --pdf / --text / config.paths.infer_pdf 중 하나는 지정되어야 합니다.")
            return

        if not os.path.exists(file_path):
            print(f"[ERROR] 파일 없음: {file_path}")
            return

        print(f"[INFO] 파일에서 텍스트 추출 중: {file_path}")
        src_text = extract_text_from_file(file_path)

    if not src_text.strip():
        print("[ERROR] 입력 텍스트가 비어 있습니다.")
        return

    # ----- generate 설정값 결정 (config + CLI override) -----
    max_len_cfg = cfg.get("max_length", {})
    gen_cfg = cfg.get("generate", {})

    # max_input_len
    if args.max_input_len is not None:
        max_input_len = args.max_input_len
    else:
        max_input_len = gen_cfg.get(
            "max_input_len",
            max_len_cfg.get("input_max_len", 512),
        )

    # max_new_tokens
    if args.max_new_tokens is not None:
        max_new_tokens = args.max_new_tokens
    else:
        max_new_tokens = gen_cfg.get(
            "max_new_tokens",
            max_len_cfg.get("target_max_len", 256),
        )

    # min_length
    if args.min_length is not None:
        min_length = args.min_length
    else:
        min_length = gen_cfg.get("min_length", 30)

    # num_beams
    if args.num_beams is not None:
        num_beams = args.num_beams
    else:
        num_beams = gen_cfg.get("num_beams", 4)

    # length_penalty
    if args.length_penalty is not None:
        length_penalty = args.length_penalty
    else:
        length_penalty = gen_cfg.get("length_penalty", 1.5)

    # no_repeat_ngram_size
    if args.no_repeat_ngram_size is not None:
        no_repeat_ngram_size = args.no_repeat_ngram_size
    else:
        no_repeat_ngram_size = gen_cfg.get("no_repeat_ngram_size", 3)

    print(f"[INFO] 모델 로드 중: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    print("[INFO] 요약 생성 설정:")
    print(f"  max_input_len      = {max_input_len}")
    print(f"  max_new_tokens     = {max_new_tokens}")
    print(f"  min_length         = {min_length}")
    print(f"  num_beams          = {num_beams}")
    print(f"  length_penalty     = {length_penalty}")
    print(f"  no_repeat_ngram_sz = {no_repeat_ngram_size}")

    print("[INFO] 요약 생성 중...")
    summary = summarize(
        model=model,
        tokenizer=tokenizer,
        text=src_text,
        max_input_len=max_input_len,
        max_new_tokens=max_new_tokens,
        min_length=min_length,
        num_beams=num_beams,
        length_penalty=length_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )

    print("\n==================== SUMMARY ====================")
    print(summary)
    print("=================================================")


if __name__ == "__main__":
    main()
