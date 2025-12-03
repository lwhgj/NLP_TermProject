import json
from pathlib import Path
import logging

from adobe.pdfservices.operation.auth.service_principal_credentials import ServicePrincipalCredentials
from adobe.pdfservices.operation.exception.exceptions import ServiceApiException, ServiceUsageException, SdkException
from adobe.pdfservices.operation.pdf_services import PDFServices
from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
from adobe.pdfservices.operation.pdfjobs.jobs.extract_pdf_job import ExtractPDFJob
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_element_type import ExtractElementType
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_pdf_params import ExtractPDFParams
from adobe.pdfservices.operation.pdfjobs.result.extract_pdf_result import ExtractPDFResult


# ============================
# 🔥 여기에 네 Adobe 자격증명 입력
# ============================
CLIENT_ID = ""
CLIENT_SECRET = ""
# ============================


def extract_pdf_to_txt(input_pdf_path: Path, output_txt_path: Path, pdf_services: PDFServices) -> None:
    """한 개의 PDF를 Adobe Extract API로 텍스트만 뽑아서 .txt로 저장"""
    # PDF 파일 읽기
    with input_pdf_path.open("rb") as f:
        input_stream = f.read()

    # PDF를 Adobe 쪽 asset으로 업로드
    input_asset = pdf_services.upload(
        input_stream=input_stream,
        mime_type=PDFServicesMediaType.PDF,
    )

    # 텍스트만 추출하도록 파라미터 설정
    extract_pdf_params = ExtractPDFParams(
        elements_to_extract=[ExtractElementType.TEXT],
    )

    # 작업(Job) 생성
    extract_pdf_job = ExtractPDFJob(
        input_asset=input_asset,
        extract_pdf_params=extract_pdf_params,
    )

    # Job 실행 및 결과 받기
    location = pdf_services.submit(extract_pdf_job)
    pdf_services_response = pdf_services.get_job_result(location, ExtractPDFResult)

    # 결과 객체에서 JSON 형태의 내용 가져오기
    extract_result: ExtractPDFResult = pdf_services_response.get_result()
    content_json = extract_result.get_content_json()

    # 문자열/바이트 → dict 로 변환
    if isinstance(content_json, (bytes, str)):
        data = json.loads(content_json)
    else:
        data = content_json

    elements = data.get("elements", [])
    texts = []
    for element in elements:
        text = element.get("Text")
        if text:
            texts.append(text.strip())

    # 그냥 줄글처럼 보이게 \n으로 이어 붙이기
    plain_text = "\n".join(t for t in texts if t)

    # 출력 폴더 만들고 txt 저장
    output_txt_path.parent.mkdir(parents=True, exist_ok=True)
    with output_txt_path.open("w", encoding="utf-8") as f:
        f.write(plain_text)


def main():
    logging.basicConfig(level=logging.INFO)

    # === 경로 설정 ===
    # 1) 스크립트 기준 input/output 폴더 쓰고 싶으면:
    base_dir = Path(__file__).resolve().parent
    input_dir = base_dir / "input"
    output_dir = base_dir / "output"

    # 2) 절대경로로 쓰고 싶으면 위 두 줄 대신 아래처럼 바꿔도 됨:
    # input_dir = Path(r"C:\Users\inwoo\PycharmProjects\NLP\NLP_TermProject\data\pdf_data")
    # output_dir = Path(r"C:\Users\inwoo\PycharmProjects\NLP\NLP_TermProject\data\adobe_texts")

    # === 자격 증명 확인 ===
    if CLIENT_ID.startswith("여기에_네") or CLIENT_SECRET.startswith("여기에_네"):
        raise RuntimeError("CLIENT_ID / CLIENT_SECRET 를 코드 상단에 제대로 입력하세요!")

    # 자격 증명 + PDFServices 인스턴스 생성
    credentials = ServicePrincipalCredentials(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
    )
    pdf_services = PDFServices(credentials=credentials)

    # input 폴더의 모든 PDF 처리
    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        logging.warning("input 폴더에 PDF 파일이 없습니다: %s", input_dir)
        return

    for pdf_path in pdf_files:
        output_name = pdf_path.stem + ".txt"
        output_path = output_dir / output_name
        logging.info("처리 중: %s -> %s", pdf_path.name, output_name)
        try:
            extract_pdf_to_txt(pdf_path, output_path, pdf_services)
        except (ServiceApiException, ServiceUsageException, SdkException) as e:
            logging.exception("Adobe API 실행 중 오류: %s", e)
        except Exception as e:
            logging.exception("예상치 못한 오류: %s", e)


if __name__ == "__main__":
    main()
