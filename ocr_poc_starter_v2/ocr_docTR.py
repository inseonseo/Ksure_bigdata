# -*- coding: utf-8 -*-
"""
docTR 기반 OCR PoC (PaddleOCR 대체 버전)
- PDF(스캔) → 이미지 → docTR OCR → 섹션 추출(사고경위/심사의견 등) → 결과 저장
- 폴더/단일 파일 모두 지원
- 완전 로컬 실행 (모델 가중치 최초 1회 캐시 다운로드 후 오프라인 가능)
"""

import os
import re
import io
import sys
import json
import argparse
import tempfile
from pathlib import Path
from typing import List, Dict, Optional

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from pdf2image import convert_from_path

# docTR
from doctr.io import DocumentFile
from doctr.models import ocr_predictor


# -------------------------
# 설정
# -------------------------
SECTION_KEYWORDS = {
    "요약_결재": [
        "내부결재", "결재", "요약", "지급한다", "지급 결정", "지급 예정"
    ],
    "보상판정_계약개요": [
        "보상판정 개요", "보험계약", "계약내용", "수출 이행", "통지", "청구일"
    ],
    "사고조사_경위": [
        "사고조사내용", "사고 조사", "사고경위", "거래 경위", "거래 및 사고경위", "조사내용"
    ],
    "심사의견_종합의견": [
        "심사의견", "종합의견", "변제충당 검토", "연속 수출 검토", "면책", "면책사유"
    ],
    "결재서류": [
        "결재서류", "첨부 서류", "증빙 서류", "결재"
    ],
}

PAGE_BREAK = "---- page {page:03d} ----"

def sanitize_for_filename(name: str) -> str:
    return re.sub(r'[\\/:\*\?"<>\|\s]+', "_", name)


# -------------------------
# 도우미: docTR OCR
# -------------------------
_ocr_model = None

def get_doctr_model():
    global _ocr_model
    if _ocr_model is None:
        # detection + recognition 통합 predictor (PyTorch)
        _ocr_model = ocr_predictor(pretrained=True)
    return _ocr_model

def doctr_ocr_on_pil(pil_img: Image.Image) -> List[str]:
    """
    PIL 이미지 1장에 대해 docTR로 텍스트 라인 리스트 추출
    """
    model = get_doctr_model()
    # Windows 환경에서 numpy 전달 시 백엔드 버전에 따라 오류가 발생하는 케이스가 있어
    # 임시 파일로 저장 후 경로 기반으로 처리한다.
    rgb_img = pil_img.convert("RGB")
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        rgb_img.save(tmp_path)
        doc = DocumentFile.from_images([tmp_path])  # batch = 1
        result = model(doc)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    lines = []
    # result.pages[0].blocks[].lines[].words[].value
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                txt = " ".join([w.value for w in line.words]).strip()
                if txt:
                    lines.append(txt)
    return lines


# -------------------------
# OCR 파이프라인
# -------------------------
def pdf_to_page_images(
    pdf_path: str,
    dpi: int = 400,
    poppler_path: Optional[str] = None,
) -> List[Image.Image]:
    """
    PDF → PIL 이미지 리스트
    """
    pages = convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)
    return pages

def run_ocr_for_pdf(
    pdf_path: str,
    out_dir: Path,
    dpi: int = 400,
    poppler_path: Optional[str] = None,
) -> Dict:
    """
    PDF 1개 OCR 실행 → 페이지별 텍스트, 섹션 텍스트, 요약 JSON/CSV 요소 반환
    """
    pdf_path = str(pdf_path)
    pdf_stem = Path(pdf_path).stem
    work_dir = out_dir / sanitize_for_filename(pdf_stem)
    work_dir.mkdir(parents=True, exist_ok=True)

    pages = pdf_to_page_images(pdf_path, dpi=dpi, poppler_path=poppler_path)

    page_texts: List[str] = []
    for i, pil_img in enumerate(tqdm(pages, desc=f"OCR {pdf_stem}", unit="page")):
        lines = doctr_ocr_on_pil(pil_img)
        txt = "\n".join(lines)
        page_texts.append(txt)

        # 페이지별 저장
        with open(work_dir / f"page_{i+1:03d}.txt", "w", encoding="utf-8") as f:
            f.write(txt)
        with open(work_dir / f"page_{i+1:03d}.json", "w", encoding="utf-8") as f:
            json.dump({"page": i+1, "lines": lines}, f, ensure_ascii=False, indent=2)

    # 섹션 분류(룰 기반)
    sections: Dict[str, List[str]] = {k: [] for k in SECTION_KEYWORDS.keys()}
    joined_for_dump: List[str] = []
    for i, txt in enumerate(page_texts, start=1):
        joined_for_dump.append(PAGE_BREAK.format(page=i))
        joined_for_dump.append(txt)

        lower = txt  # 한국어라 lowercase 영향 적음
        for sec, keys in SECTION_KEYWORDS.items():
            if any(k in lower for k in keys):
                sections[sec].append(PAGE_BREAK.format(page=i) + "\n" + txt)
                break  # 1개 섹션 매칭 후 다음 페이지로

    # 섹션 파일 저장
    for sec, chunks in sections.items():
        if chunks:
            sec_file = work_dir / f"section_{sanitize_for_filename(sec)}.txt"
            with open(sec_file, "w", encoding="utf-8") as f:
                f.write("\n\n".join(chunks))

    # 간단 요약(필요시 확장)
    summary = {
        "file": Path(pdf_path).name,
        "page_count": len(pages),
        "section_counts": {sec: len(chunks) for sec, chunks in sections.items()},
        "error": "",
    }
    with open(work_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def process_folder(
    pdf_dir: str,
    out_dir: str,
    dpi: int = 400,
    poppler_path: Optional[str] = None,
) -> List[Dict]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    pdf_dir = Path(pdf_dir)
    pdf_files = sorted([p for p in pdf_dir.iterdir() if p.suffix.lower() == ".pdf"])
    results = []
    for pdf in pdf_files:
        try:
            res = run_ocr_for_pdf(str(pdf), out_path, dpi=dpi, poppler_path=poppler_path)
            results.append(res)
        except Exception as e:
            res = {
                "file": pdf.name,
                "page_count": 0,
                "section_counts": {},
                "error": str(e),
            }
            results.append(res)
    # summary.csv 저장
    csv_path = out_path / "summary.csv"
    write_summary_csv(csv_path, results)
    return results

def write_summary_csv(csv_path: Path, rows: List[Dict]):
    import csv
    fields = ["file", "page_count", "section_counts", "error"]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            r2 = dict(r)
            r2["section_counts"] = json.dumps(r2.get("section_counts", {}), ensure_ascii=False)
            w.writerow(r2)


# -------------------------
# CLI
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="docTR OCR PoC (PaddleOCR 대체)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pdf_dir", type=str, help="처리할 PDF 폴더")
    g.add_argument("--pdf_file", type=str, help="단일 PDF 파일 경로")
    ap.add_argument("--out_dir", type=str, required=True, help="출력 폴더")
    ap.add_argument("--dpi", type=int, default=400, help="PDF → 이미지 변환 DPI")
    ap.add_argument("--poppler_path", type=str, default=None, help="Windows에서 Poppler bin 경로 (예: C:\\poppler-xx\\Library\\bin)")
    return ap.parse_args()

def main():
    args = parse_args()
    if args.pdf_dir:
        results = process_folder(args.pdf_dir, args.out_dir, dpi=args.dpi, poppler_path=args.poppler_path)
        print(f"[DONE] processed: {len(results)} files")
    else:
        out_path = Path(args.out_dir); out_path.mkdir(parents=True, exist_ok=True)
        res = run_ocr_for_pdf(args.pdf_file, out_path, dpi=args.dpi, poppler_path=args.poppler_path)
        print("[DONE] single file:", res["file"], "pages:", res["page_count"], "sections:", res["section_counts"])

if __name__ == "__main__":
    main()
