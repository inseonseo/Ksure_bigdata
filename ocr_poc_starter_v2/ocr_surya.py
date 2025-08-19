#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Surya 기반 OCR PoC (한글+영문, 메모리 효율적)
- PDF(스캔) → 이미지 → Surya OCR → 섹션 추출(사고경위/심사의견 등) → 결과 저장
- 폴더/단일 파일 모두 지원
- 메모리 효율적이고 빠른 처리 (GPU/CPU 모두 지원)
- 90+ 언어 지원, 레이아웃 분석 포함
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
    from surya.detection import DetectionPredictor
    from surya.foundation import FoundationPredictor
    from surya.recognition import RecognitionPredictor
    from surya.layout import LayoutPredictor
    SURYA_AVAILABLE = True
except ImportError:
    SURYA_AVAILABLE = False


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
# Surya 초기화/실행
# -------------------------
_det_model = None
_foundation_model = None
_rec_model = None
_order_model = None


def get_surya_models():
    global _det_model, _foundation_model, _rec_model, _order_model
    if not SURYA_AVAILABLE:
        raise RuntimeError("surya가 설치되어 있지 않습니다. pip install surya-ocr")
    
    if _det_model is None:
        _det_model = DetectionPredictor()
    if _foundation_model is None:
        _foundation_model = FoundationPredictor()
    if _rec_model is None:
        _rec_model = RecognitionPredictor(_foundation_model)
    if _order_model is None:
        _order_model = LayoutPredictor()
    
    return _det_model, _foundation_model, _rec_model, _order_model


def surya_ocr_on_pil(pil_img: Image.Image, languages: List[str] = ["ko", "en"]) -> Dict[str, Any]:
    """
    PIL 이미지 1장에 대해 Surya로 완전한 텍스트 추출 (Detection + Recognition + Layout)
    """
    try:
        det_model, foundation_model, rec_model, order_model = get_surya_models()
        
        # 1. Detection (텍스트 영역 자동 탐지)
        det_result = det_model([pil_img])
        
        # 2. Recognition (탐지된 영역에서 텍스트 읽기)
        if det_result and len(det_result) > 0 and det_result[0].bboxes:
            # Detection 결과에서 bbox 정보 추출
            bboxes = det_result[0].bboxes
            scores = [getattr(bbox, 'confidence', 0.0) for bbox in bboxes]
            polygons = [getattr(bbox, 'polygon', []) for bbox in bboxes]
            
            # Recognition 실행
            rec_result = rec_model.get_bboxes_text([pil_img], scores, polygons)
        else:
            rec_result = []
        
        # 3. Layout (읽기 순서 결정)
        layout_result = order_model.batch_layout_detection([pil_img])
        
        # 결과 파싱
        lines = []
        texts_only: List[str] = []
        
        # Recognition 결과에서 텍스트 추출
        if rec_result and len(rec_result) > 0:
            for text_result in rec_result:
                if hasattr(text_result, 'text') and text_result.text.strip():
                    text = text_result.text.strip()
                    bbox = getattr(text_result, 'bbox', [])
                    conf = getattr(text_result, 'confidence', 0.0)
                    
                    lines.append({
                        "bbox": bbox,
                        "text": text,
                        "score": conf
                    })
                    texts_only.append(text)
        
        return {
            "engine": "surya", 
            "lines": lines, 
            "text": "\n".join(texts_only),
            "detection_count": len(det_result[0].bboxes) if det_result and len(det_result) > 0 else 0
        }
        
    except Exception as e:
        # 에러 발생 시 빈 결과 반환
        return {
            "engine": "surya", 
            "lines": [], 
            "text": "", 
            "error": str(e)
        }


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
    languages: List[str] = ["ko", "en"],
) -> Dict:
    pdf_path = str(pdf_path)
    pdf_stem = Path(pdf_path).stem
    work_dir = out_dir / sanitize_for_filename(pdf_stem)
    work_dir.mkdir(parents=True, exist_ok=True)

    pages = pdf_to_page_images(pdf_path, dpi=dpi, poppler_path=poppler_path)

    page_texts: List[str] = []
    for i, pil_img in enumerate(tqdm(pages, desc=f"OCR {pdf_stem}", unit="page")):
        try:
            res = surya_ocr_on_pil(pil_img, languages=languages)
            txt = res.get("text", "")
            page_texts.append(txt)

            # 페이지별 저장
            with open(work_dir / f"page_{i+1:03d}.txt", "w", encoding="utf-8") as f:
                f.write(txt)
            with open(work_dir / f"page_{i+1:03d}.json", "w", encoding="utf-8") as f:
                # raw_result는 너무 크므로 제외
                save_res = {k: v for k, v in res.items() if k != "raw_result"}
                json.dump(save_res, f, ensure_ascii=False, indent=2)
        except Exception as e:
            # 개별 페이지 에러 시 빈 텍스트로 처리
            page_texts.append("")
            error_res = {"engine": "surya", "lines": [], "text": "", "error": str(e)}
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
        "engine_used": "surya",
        "languages": languages,
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
    languages: List[str] = ["ko", "en"],
) -> List[Dict]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    pdf_dir = Path(pdf_dir)
    pdf_files = sorted([p for p in pdf_dir.iterdir() if p.suffix.lower() == ".pdf"])
    results = []
    for pdf in pdf_files:
        try:
            res = run_ocr_for_pdf(
                str(pdf), 
                out_path, 
                dpi=dpi, 
                poppler_path=poppler_path, 
                languages=languages
            )
            results.append(res)
        except Exception as e:
            res = {
                "file": pdf.name,
                "page_count": 0,
                "section_counts": {},
                "engine_used": "surya",
                "languages": languages,
                "error": str(e),
            }
            results.append(res)
    # summary.csv 저장
    csv_path = out_path / "summary.csv"
    write_summary_csv(csv_path, results)
    return results


def write_summary_csv(csv_path: Path, rows: List[Dict]):
    import csv
    fields = ["file", "page_count", "section_counts", "engine_used", "languages", "error"]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            r2 = dict(r)
            r2["section_counts"] = json.dumps(r2.get("section_counts", {}), ensure_ascii=False)
            r2["languages"] = json.dumps(r2.get("languages", []), ensure_ascii=False)
            w.writerow(r2)


# -------------------------
# CLI
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Surya OCR PoC (KO+EN, 메모리 효율적)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pdf_dir", type=str, help="처리할 PDF 폴더")
    g.add_argument("--pdf_file", type=str, help="단일 PDF 파일 경로")
    ap.add_argument("--out_dir", type=str, required=True, help="출력 폴더")
    ap.add_argument("--dpi", type=int, default=400, help="PDF → 이미지 변환 DPI")
    ap.add_argument("--poppler_path", type=str, default=None, help="Windows에서 Poppler bin 경로 (예: C:\\poppler-xx\\Library\\bin)")
    ap.add_argument("--languages", type=str, default="ko,en", help="인식할 언어 (쉼표로 구분, 기본: ko,en)")
    return ap.parse_args()


def main():
    args = parse_args()
    languages = [lang.strip() for lang in args.languages.split(",")]
    
    if args.pdf_dir:
        results = process_folder(
            args.pdf_dir, 
            args.out_dir, 
            dpi=args.dpi, 
            poppler_path=args.poppler_path, 
            languages=languages
        )
        print(f"[DONE] processed: {len(results)} files")
    else:
        out_path = Path(args.out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        res = run_ocr_for_pdf(
            args.pdf_file, 
            out_path, 
            dpi=args.dpi, 
            poppler_path=args.poppler_path, 
            languages=languages
        )
        print("[DONE] single file:", res["file"], "pages:", res["page_count"], "sections:", res["section_counts"])


if __name__ == "__main__":
    main()

