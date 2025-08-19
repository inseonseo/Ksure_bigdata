#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EasyOCR 기반 OCR PoC (메모리 효율적 버전)
- PDF(스캔) → 이미지 → EasyOCR → 섹션 추출(사고경위/심사의견 등) → 결과 저장
- 메모리 사용량 최적화: 낮은 DPI, 이미지 리사이즈, 개별 처리
- 폴더/단일 파일 모두 지원
"""

import os
import re
import json
import argparse
import gc
from pathlib import Path
from typing import List, Dict, Optional, Any

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from pdf2image import convert_from_path

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False


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
# EasyOCR 초기화/실행 (메모리 효율적)
# -------------------------
_easyocr_reader = None


def get_easyocr_reader(gpu: bool = False):
    global _easyocr_reader
    if not EASYOCR_AVAILABLE:
        raise RuntimeError("easyocr가 설치되어 있지 않습니다. pip install easyocr")
    
    if _easyocr_reader is None:
        # 한국어+영어 모델만 로드 (메모리 절약)
        _easyocr_reader = easyocr.Reader(['ko', 'en'], gpu=gpu, verbose=False)
    
    return _easyocr_reader


def resize_image_for_memory(pil_img: Image.Image, max_size: int = 2048) -> Image.Image:
    """
    메모리 절약을 위해 이미지 크기 조정
    """
    width, height = pil_img.size
    
    # 최대 크기 제한
    if width > max_size or height > max_size:
        ratio = min(max_size / width, max_size / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        pil_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return pil_img


def easyocr_on_pil_lowmem(pil_img: Image.Image, gpu: bool = False, max_size: int = 2048) -> Dict[str, Any]:
    """
    PIL 이미지 1장에 대해 EasyOCR로 텍스트 추출 (메모리 효율적)
    """
    try:
        # 이미지 크기 조정
        pil_img = resize_image_for_memory(pil_img, max_size)
        
        reader = get_easyocr_reader(gpu)
        
        # OCR 실행
        results = reader.readtext(np.array(pil_img))
        
        # 결과 파싱
        lines = []
        texts_only: List[str] = []
        
        for item in results:
            if len(item) >= 3:
                bbox, text, conf = item[0], str(item[1]), float(item[2])
                
                # JSON 직렬화를 위한 타입 변환
                bbox_list = bbox.tolist() if hasattr(bbox, 'tolist') else list(bbox)
                score_float = float(conf) if hasattr(conf, 'item') else conf
                
                lines.append({
                    "bbox": bbox_list,
                    "text": text,
                    "score": score_float
                })
                if text:
                    texts_only.append(text)
        
        # 메모리 정리
        del results
        gc.collect()
        
        return {
            "engine": "easyocr_lowmem", 
            "lines": lines, 
            "text": "\n".join(texts_only)
        }
        
    except Exception as e:
        # 에러 발생 시 빈 결과 반환
        return {
            "engine": "easyocr_lowmem", 
            "lines": [], 
            "text": "", 
            "error": str(e)
        }


# -------------------------
# OCR 파이프라인
# -------------------------
def pdf_to_page_images_lowmem(
    pdf_path: str,
    dpi: int = 300,  # 낮은 DPI로 메모리 절약
    poppler_path: Optional[str] = None,
) -> List[Image.Image]:
    pages = convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)
    return pages


def run_ocr_for_pdf_lowmem(
    pdf_path: str,
    out_dir: Path,
    dpi: int = 300,  # 낮은 DPI
    poppler_path: Optional[str] = None,
    gpu: bool = False,
    max_image_size: int = 2048,  # 이미지 최대 크기 제한
) -> Dict:
    pdf_path = str(pdf_path)
    pdf_stem = Path(pdf_path).stem
    work_dir = out_dir / sanitize_for_filename(pdf_stem)
    work_dir.mkdir(parents=True, exist_ok=True)

    pages = pdf_to_page_images_lowmem(pdf_path, dpi=dpi, poppler_path=poppler_path)

    page_texts: List[str] = []
    for i, pil_img in enumerate(tqdm(pages, desc=f"OCR {pdf_stem}", unit="page")):
        try:
            res = easyocr_on_pil_lowmem(pil_img, gpu=gpu, max_size=max_image_size)
            txt = res.get("text", "")
            page_texts.append(txt)

            # 페이지별 저장
            with open(work_dir / f"page_{i+1:03d}.txt", "w", encoding="utf-8") as f:
                f.write(txt)
            with open(work_dir / f"page_{i+1:03d}.json", "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)
                
            # 메모리 정리
            del pil_img
            gc.collect()
            
        except Exception as e:
            # 개별 페이지 에러 시 빈 텍스트로 처리
            page_texts.append("")
            error_res = {"engine": "easyocr_lowmem", "lines": [], "text": "", "error": str(e)}
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
        "engine_used": "easyocr_lowmem",
        "dpi_used": dpi,
        "max_image_size": max_image_size,
        "error": "",
    }
    with open(work_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def process_folder_lowmem(
    pdf_dir: str,
    out_dir: str,
    dpi: int = 300,
    poppler_path: Optional[str] = None,
    gpu: bool = False,
    max_image_size: int = 2048,
) -> List[Dict]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    pdf_dir = Path(pdf_dir)
    pdf_files = sorted([p for p in pdf_dir.iterdir() if p.suffix.lower() == ".pdf"])
    results = []
    for pdf in pdf_files:
        try:
            res = run_ocr_for_pdf_lowmem(
                str(pdf), 
                out_path, 
                dpi=dpi, 
                poppler_path=poppler_path, 
                gpu=gpu,
                max_image_size=max_image_size
            )
            results.append(res)
        except Exception as e:
            res = {
                "file": pdf.name,
                "page_count": 0,
                "section_counts": {},
                "engine_used": "easyocr_lowmem",
                "dpi_used": dpi,
                "max_image_size": max_image_size,
                "error": str(e),
            }
            results.append(res)
    # summary.csv 저장
    csv_path = out_path / "summary.csv"
    write_summary_csv(csv_path, results)
    return results


def write_summary_csv(csv_path: Path, rows: List[Dict]):
    import csv
    fields = ["file", "page_count", "section_counts", "engine_used", "dpi_used", "max_image_size", "error"]
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
    ap = argparse.ArgumentParser(description="EasyOCR PoC (메모리 효율적 버전)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pdf_dir", type=str, help="처리할 PDF 폴더")
    g.add_argument("--pdf_file", type=str, help="단일 PDF 파일 경로")
    ap.add_argument("--out_dir", type=str, required=True, help="출력 폴더")
    ap.add_argument("--dpi", type=int, default=300, help="PDF → 이미지 변환 DPI (기본: 300, 메모리 절약)")
    ap.add_argument("--poppler_path", type=str, default=None, help="Windows에서 Poppler bin 경로")
    ap.add_argument("--gpu", action="store_true", help="GPU 사용 (가능한 경우)")
    ap.add_argument("--max_image_size", type=int, default=2048, help="이미지 최대 크기 (기본: 2048, 메모리 절약)")
    return ap.parse_args()


def main():
    args = parse_args()
    
    if args.pdf_dir:
        results = process_folder_lowmem(
            args.pdf_dir, 
            args.out_dir, 
            dpi=args.dpi, 
            poppler_path=args.poppler_path, 
            gpu=args.gpu,
            max_image_size=args.max_image_size
        )
        print(f"[DONE] processed: {len(results)} files")
    else:
        out_path = Path(args.out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        res = run_ocr_for_pdf_lowmem(
            args.pdf_file, 
            out_path, 
            dpi=args.dpi, 
            poppler_path=args.poppler_path, 
            gpu=args.gpu,
            max_image_size=args.max_image_size
        )
        print("[DONE] single file:", res["file"], "pages:", res["page_count"], "sections:", res["section_counts"])


if __name__ == "__main__":
    main()

