# OCR PoC (심사보고서용)

이 저장소는 **스캔된 PDF(보상심사서)**를 대상으로 **PaddleOCR 중심**의 파이프라인을 실행하여
- 페이지별 텍스트 추출
- 섹션(요약/판정개요/사고경위/심사의견/결재서류) 추정
- 핵심 필드(면책/지급 판정, 금액, 주요 일자) 추출
- CSV 요약 리포트 생성

을 수행합니다. Tesseract는 **옵션**으로 포함되어 baseline 비교에만 사용합니다.

## 설치
```bash
# (선택) 가상환경 생성
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

> **pdf2image**는 OS에 따라 `poppler`(pdftoppm)가 필요할 수 있습니다.
- macOS: `brew install poppler`
- Ubuntu: `sudo apt-get install poppler-utils`
- Windows: https://github.com/oschwartz10612/poppler-windows/releases 에서 다운로드 후 PATH 설정

> **Tesseract** baseline을 쓰려면:
- macOS: `brew install tesseract-lang`
- Ubuntu: `sudo apt-get install tesseract-ocr tesseract-ocr-kor`
- Windows: https://github.com/UB-Mannheim/tesseract/wiki

## 실행 예시
```bash
python ocr_poc.py --pdf_dir ./pdfs --out_dir ./out --dpi 400 --engine paddle --lang korean
# baseline 비교까지: --engine both
```

## 출력 산출물
- `out/<파일명_폴더>/page_001.txt` … 페이지별 OCR 텍스트
- `out/<파일명_폴더>/page_001.json` … OCR bbox/score 등 원시 결과
- `out/<파일명_폴더>/section_사고경위.txt` … 섹션 추정에 따라 저장되는 텍스트
- `out/summary.csv` … 파일 단위 요약(면책/지급, 금액/일자 등)
