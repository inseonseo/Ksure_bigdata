#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EasyOCR 기반 OCR PoC (한글+영문)
- PDF(스캔) → 이미지 → EasyOCR(KO+EN) → 섹션 추출(사고경위/심사의견 등) → 결과 저장
- 폴더/단일 파일 모두 지원
- 완전 로컬 실행(모델 가중치 최초 1회 캐시 다운로드 후 오프라인 가능)
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Any

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from pdf2image import convert_from_path

try:
    import easyocr
except ImportError:
    easyocr = None


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
# EasyOCR 초기화/실행
# -------------------------
_easy_reader: Optional[Any] = None


def get_easyocr_reader(gpu: bool = False, recog_network: Optional[str] = None):
    global _easy_reader
    if easyocr is None:
        raise RuntimeError("easyocr가 설치되어 있지 않습니다. pip install easyocr")
    if _easy_reader is None:
        # 언어: 한국어+영문
        # 기본값: EasyOCR에게 언어만 전달하여 내부 기본 네트워크를 자동 선택
        # 사용자 네트워크를 쓰려면 recog_network에 정확한 네트워크명을 지정해야 하며,
        # 이 경우 .EasyOCR/user_network 경로에 YAML이 필요합니다.
        if recog_network:
            _easy_reader = easyocr.Reader(["ko", "en"], gpu=gpu, recog_network=recog_network)
        else:
            _easy_reader = easyocr.Reader(["ko", "en"], gpu=gpu)
    return _easy_reader


def easyocr_on_pil(pil_img: Image.Image, gpu: bool = False) -> Dict[str, Any]:
    try:
        reader = get_easyocr_reader(gpu=gpu)
        np_img = np.array(pil_img.convert("RGB"))
        # detail=1 → bbox, text, conf 반환
        results = reader.readtext(np_img, detail=1, paragraph=False)
        lines = []
        texts_only: List[str] = []
        for item in results:
            # item: [bbox, text, conf]
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue
            bbox, text, conf = item[0], str(item[1]), float(item[2])
            # bbox는 numpy array이므로 JSON 직렬화를 위해 list로 변환
            bbox_list = bbox.tolist() if hasattr(bbox, 'tolist') else list(bbox)
            # conf도 numpy 타입일 수 있으므로 float로 확실히 변환
            score_float = float(conf) if hasattr(conf, 'item') else conf
            lines.append({"bbox": bbox_list, "text": text, "score": score_float})
            if text:
                texts_only.append(text)
        return {"engine": "easyocr", "lines": lines, "text": "\n".join(texts_only)}
    except Exception as e:
        # 에러 발생 시 빈 결과 반환
        return {"engine": "easyocr", "lines": [], "text": "", "error": str(e)}


# -------------------------
# OCR 파이프라인
# -------------------------
def pdf_to_page_images(
    pdf_path: str,
    dpi: int = 400,
    poppler_path: Optional[str] = None,
) -> List[Image.Image]:
    pages = convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)
    return pages


def run_ocr_for_pdf(
    pdf_path: str,
    out_dir: Path,
    dpi: int = 400,
    poppler_path: Optional[str] = None,
    gpu: bool = False,
    recog_network: Optional[str] = None,
) -> Dict:
    pdf_path = str(pdf_path)
    pdf_stem = Path(pdf_path).stem
    work_dir = out_dir / sanitize_for_filename(pdf_stem)
    work_dir.mkdir(parents=True, exist_ok=True)

    pages = pdf_to_page_images(pdf_path, dpi=dpi, poppler_path=poppler_path)

    page_texts: List[str] = []
    for i, pil_img in enumerate(tqdm(pages, desc=f"OCR {pdf_stem}", unit="page")):
        try:
            res = easyocr_on_pil(pil_img, gpu=gpu)
            txt = res.get("text", "")
            page_texts.append(txt)

            # 페이지별 저장
            with open(work_dir / f"page_{i+1:03d}.txt", "w", encoding="utf-8") as f:
                f.write(txt)
            with open(work_dir / f"page_{i+1:03d}.json", "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)
        except Exception as e:
            # 개별 페이지 에러 시 빈 텍스트로 처리
            page_texts.append("")
            error_res = {"engine": "easyocr", "lines": [], "text": "", "error": str(e)}
            with open(work_dir / f"page_{i+1:03d}.txt", "w", encoding="utf-8") as f:
                f.write("")
            with open(work_dir / f"page_{i+1:03d}.json", "w", encoding="utf-8") as f:
                json.dump(error_res, f, ensure_ascii=False, indent=2)

    # 섹션 분류(룰 기반)
    sections: Dict[str, List[str]] = {k: [] for k in SECTION_KEYWORDS.keys()}
    for i, txt in enumerate(page_texts, start=1):
        lower = txt  # 한국어라 lowercase 영향 적음
        matched = None
        for sec, keys in SECTION_KEYWORDS.items():
            if any(k in lower for k in keys):
                matched = sec
                break
        block = PAGE_BREAK.format(page=i) + "\n" + txt
        if matched is None:
            # 미분류는 아무 섹션에도 넣지 않음(필요시 확장 가능)
            pass
        else:
            sections[matched].append(block)

    # 섹션 파일 저장
    for sec, chunks in sections.items():
        if chunks:
            sec_file = work_dir / f"section_{sanitize_for_filename(sec)}.txt"
            with open(sec_file, "w", encoding="utf-8") as f:
                f.write("\n\n".join(chunks))

    summary = {
        "file": Path(pdf_path).name,
        "page_count": len(pages),
        "section_counts": {sec: len(chunks) for sec, chunks in sections.items()},
        "engine_used": "easyocr",
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
    gpu: bool = False,
    recog_network: Optional[str] = None,
) -> List[Dict]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    pdf_dir = Path(pdf_dir)
    pdf_files = sorted([p for p in pdf_dir.iterdir() if p.suffix.lower() == ".pdf"])
    results = []
    for pdf in pdf_files:
        try:
            res = run_ocr_for_pdf(str(pdf), out_path, dpi=dpi, poppler_path=poppler_path, gpu=gpu, recog_network=recog_network)
            results.append(res)
        except Exception as e:
            res = {
                "file": pdf.name,
                "page_count": 0,
                "section_counts": {},
                "engine_used": "easyocr",
                "error": str(e),
            }
            results.append(res)
    # summary.csv 저장
    csv_path = out_path / "summary.csv"
    write_summary_csv(csv_path, results)
    return results


def write_summary_csv(csv_path: Path, rows: List[Dict]):
    import csv
    fields = ["file", "page_count", "section_counts", "engine_used", "error"]
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
    ap = argparse.ArgumentParser(description="EasyOCR OCR PoC (KO+EN)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pdf_dir", type=str, help="처리할 PDF 폴더")
    g.add_argument("--pdf_file", type=str, help="단일 PDF 파일 경로")
    ap.add_argument("--out_dir", type=str, required=True, help="출력 폴더")
    ap.add_argument("--dpi", type=int, default=400, help="PDF → 이미지 변환 DPI")
    ap.add_argument("--poppler_path", type=str, default=None, help="Windows에서 Poppler bin 경로 (예: C:\\poppler-xx\\Library\\bin)")
    ap.add_argument("--gpu", action="store_true", help="GPU 사용(가능한 경우)")
    ap.add_argument("--recog_network", type=str, default=None, help="EasyOCR recognizer 네트워크(선택). 지정 시 사용자 네트워크가 필요합니다.")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.pdf_dir:
        results = process_folder(args.pdf_dir, args.out_dir, dpi=args.dpi, poppler_path=args.poppler_path, gpu=args.gpu, recog_network=args.recog_network)
        print(f"[DONE] processed: {len(results)} files")
    else:
        out_path = Path(args.out_dir); out_path.mkdir(parents=True, exist_ok=True)
        res = run_ocr_for_pdf(args.pdf_file, out_path, dpi=args.dpi, poppler_path=args.poppler_path, gpu=args.gpu, recog_network=args.recog_network)
        print("[DONE] single file:", res["file"], "pages:", res["page_count"], "sections:", res["section_counts"])


if __name__ == "__main__":
    main()


