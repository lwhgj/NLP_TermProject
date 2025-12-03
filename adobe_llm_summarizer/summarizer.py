import os
import google.generativeai as genai
import time
from config import GEMINI_API_KEY

EXAMPLE_TEXT = ""


def extract_text_from_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None


def list_gemini_models():
    try:
        print("\n--- Available Gemini Models ---")
        for m in genai.list_models():
            if "generateContent" in m.supported_generation_methods:
                print(m.name)
        print("-------------------------------\n")
    except Exception as e:
        print(f"Error listing models: {e}")


def summarize_text_with_gemini(text):
    try:
        model = genai.GenerativeModel('models/gemini-2.5-flash-lite')

        prompt = "데이터에 정보가 모두 없으면, 절대 추측하거나 지어내지 마십시오." \
                 "* 이나 # 와 같은 특수문자 사용을 금지해주세요." \
                 "다음 재무 정보를 요약해주세요. 핵심적인 재무 지표와 회사의 전반적인 재무 상태에 초점을 맞춰주세요." \
                 "예시 형식과 양식을 같게 작성해주세요. 예시 형식은 다음과 같습니다." \
                 "\n\n" + EXAMPLE_TEXT + "요약해야할 재무 정보는 다음과 같습니다.\n\n" + text

        # API 호출
        response = model.generate_content(prompt)

        # [핵심 수정] 요청 성공 시 무조건 5초 대기
        # 무료 티어 한도(1분 15회)를 넘지 않도록 안전장치 (60초/15회 = 4초)
        print("  > API Call Successful. Waiting 5 seconds to respect rate limits...")
        time.sleep(4)

        return response.text

    except Exception as e:
        print(f"Error summarizing text with Gemini API: {e}")
        # 에러가 나더라도 다음 요청을 위해 잠깐 쉬어줌
        time.sleep(4)
        return ""


def main():
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY is missing.")
        return

    genai.configure(api_key=GEMINI_API_KEY)

    try:
        with open('example.txt', 'r', encoding='utf-8') as f:
            global EXAMPLE_TEXT
            EXAMPLE_TEXT = f.read()
    except FileNotFoundError:
        print("Warning: 'example.txt' not found. Creating summaries without examples.")
        EXAMPLE_TEXT = ""

    list_gemini_models()

    input_dir = 'input'
    output_dir = 'output'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        print(f"Created '{input_dir}' directory. Please put .txt files there.")
        return

    txt_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]

    if not txt_files:
        print(f"No .txt files found in {input_dir}")
        return

    for txt_file in txt_files:
        txt_path = os.path.join(input_dir, txt_file)
        base_filename = os.path.splitext(txt_file)[0]
        output_filename = base_filename + '_summary.txt'
        output_path = os.path.join(output_dir, output_filename)

        if os.path.exists(output_path):
            print(f"Summary for {txt_file} already exists. Skipping.")
            continue

        print(f"\n--- Extracting and summarizing text from {txt_file} ---")
        extracted_text = extract_text_from_txt(txt_path)

        if extracted_text:
            summary = summarize_text_with_gemini(extracted_text)
            if summary:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(summary)
                print(f"Summary saved to {output_path}")
            else:
                print(f"No summary generated for {txt_file}")
        else:
            print(f"No text extracted from {txt_file}")


if __name__ == "__main__":
    main()