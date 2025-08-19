#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
심사보고서 OCR PoC:
- PDF -> 이미지(DPI)
- (선택) 전처리
- PaddleOCR(or Tesseract) 실행
- 섹션 추정 + 핵심 필드 추출
- 결과 저장(JSON/TXT/CSV)

사용 예시:
python ocr_poc.py --pdf_dir ./pdfs --out_dir ./out --dpi 400 --engine both --lang korean
"""
import os, re, json, argparse, pathlib, sys
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional
from pdf2image import convert_from_path
from PIL import Image
import numpy as np
import cv2
import pandas as pd

def sanitize_for_filename(name: str) -> str:
    # Replace path separators and reserved characters
    return re.sub(r'[\\/\:*?"<>|\s]+', '_', name)


# -----------------------------
# PaddleOCR (필수) / Tesseract(옵션)
# -----------------------------
try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

# PaddleOCR 인스턴스 캐시(언어별 재사용)
_PADDLE_OCR_CACHE: Dict[str, Any] = {}
_TESSERACT_CMD: Optional[str] = None

def _get_paddle_ocr(lang: str = "korean"):
    if PaddleOCR is None:
        raise RuntimeError("PaddleOCR가 설치되어 있지 않습니다. pip install paddleocr")
    inst = _PADDLE_OCR_CACHE.get(lang)
    if inst is None:
        inst = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
        _PADDLE_OCR_CACHE[lang] = inst
    return inst

# -----------------------------
# 섹션 키워드 사전
# -----------------------------
SECTION_KEYWORDS = {
    "요약/결재": ["내부결재문서", "요약", "심사결과", "위반하지", "지급한다", "지급 예정", "결재"],
    "보상판정/계약개요": ["보상판정", "개요", "보험계약", "사고건", "수출 이행내역", "사고발생 통지", "보험금 청구일"],
    "사고조사/경위": ["사고조사내용", "거래", "사고경위", "경위"],
    "심사의견": ["심사의견", "종합의견", "지급하고자", "변제충당", "연속 수출", "검토", "면책"],
    "결재서류": ["결재서류", "결재"]
}

DECISION_KEYWORDS = {
    "면책": ["면책", "반려", "지급 거절", "지급하지 않"],
    "가지급": ["가지급"],
    "지급": ["지급", "지급 예정", "지급하고자", "지급한다"]
}

DATE_PATTERNS = [
    r"(\d{4})[.\-\/](\d{1,2})[.\-\/](\d{1,2})",
    r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일"
]

AMOUNT_PATTERNS = [
    r"([\d,]+)\s*(원|KRW)",
    r"(USD)\s*([\d,]+)"
]

# -----------------------------
# 유틸 함수
# -----------------------------
def ensure_dir(p: str):
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def preprocess_for_tesseract(img_cv: np.ndarray) -> np.ndarray:
    """Tesseract에 유리한 간단 전처리: 그레이스케일 -> 잡음제거 -> 이진화 -> 데스큐(근사)"""
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    # OTSU 이진화
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # deskew
    coords = np.column_stack(np.where(thresh < 255))
    angle = 0.0
    if coords.size > 0:
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = thresh.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        thresh = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return thresh

def ocr_with_paddle(img_cv: np.ndarray, lang: str = "korean") -> Dict[str, Any]:
    # PaddleOCR는 BGR 순서도 입력 가능
    ocr = _get_paddle_ocr(lang)
    res = ocr.ocr(img_cv, cls=True)
    # 결과 정리
    lines = []
    for page in res:
        if page is None:
            continue
        for line in page:
            (x1, y1), (x2, y2), (x3, y3), (x4, y4) = line[0]
            text = line[1][0]
            score = float(line[1][1])
            lines.append({
                "bbox": [x1, y1, x2, y2, x3, y3, x4, y4],
                "text": text,
                "score": score
            })
    return {"engine": "paddle", "lines": lines, "text": "\n".join([l["text"] for l in lines])}

def ocr_with_tesseract(img_cv: np.ndarray) -> Dict[str, Any]:
    if pytesseract is None:
        raise RuntimeError("pytesseract가 설치되어 있지 않습니다. pip install pytesseract")
    # Windows에서 경로 직접 지정 옵션 지원
    global _TESSERACT_CMD
    if _TESSERACT_CMD:
        try:
            pytesseract.pytesseract.tesseract_cmd = _TESSERACT_CMD
        except Exception:
            pass
    pre = preprocess_for_tesseract(img_cv)
    config = "--oem 1 --psm 6 -l kor+eng"
    text = pytesseract.image_to_string(pre, config=config)
    # bbox/score는 생략(추출 어려움)
    return {"engine": "tesseract", "lines": [], "text": text}

def score_sections(text: str) -> Tuple[str, Dict[str, int]]:
    text_norm = text.replace(" ", "")
    scores = {}
    for sec, keys in SECTION_KEYWORDS.items():
        s = sum(text_norm.count(k) for k in keys)
        scores[sec] = s
    best = max(scores, key=scores.get) if scores else "미분류"
    return best, scores

def find_first(patterns: List[str], text: str, flags=0) -> Optional[str]:
    for p in patterns:
        m = re.search(p, text, flags)
        if m:
            return m.group(0)
    return None

def extract_dates(text: str) -> Dict[str, Optional[str]]:
    # 간단: 텍스트에서 날짜 형태를 모두 찾아 앞쪽 일부만 사용
    dates = []
    for p in DATE_PATTERNS:
        for m in re.finditer(p, text):
            try:
                if "년" in p:
                    y, mo, d = m.group(1), m.group(2), m.group(3)
                    dates.append(f"{int(y):04d}-{int(mo):02d}-{int(d):02d}")
                else:
                    y, mo, d = m.group(1), m.group(2), m.group(3)
                    dates.append(f"{int(y):04d}-{int(mo):02d}-{int(d):02d}")
            except:
                continue
    # 우선순위 라벨링(사고발생/통지/청구일 등은 키워드 근처 탐색이 이상적이지만 PoC에선 전체에서 앞쪽 일부만)
    uniq = []
    for d in dates:
        if d not in uniq:
            uniq.append(d)
    out = {
        "date_1": uniq[0] if len(uniq) > 0 else None,
        "date_2": uniq[1] if len(uniq) > 1 else None,
        "date_3": uniq[2] if len(uniq) > 2 else None,
    }
    return out

def extract_amounts(text: str) -> List[str]:
    found = []
    for p in AMOUNT_PATTERNS:
        for m in re.finditer(p, text):
            found.append(m.group(0))
    # 중복 제거
    seen = set()
    out = []
    for f in found:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out

def extract_decision(text: str) -> Optional[str]:
    # 면책 키워드 존재시 우선
    for kw in DECISION_KEYWORDS["면책"]:
        if kw in text:
            return "면책"
    # 가지급
    for kw in DECISION_KEYWORDS["가지급"]:
        if kw in text:
            return "가지급"
    # 지급(일반)
    for kw in DECISION_KEYWORDS["지급"]:
        if kw in text:
            return "지급"
    return None

@dataclass
class PageResult:
    page_index: int
    section_guess: str
    section_scores: Dict[str, int]
    engine: str
    text_path: str
    json_path: str

@dataclass
class FileSummary:
    file: str
    engine_used: str
    decision: Optional[str]
    amount_candidates: List[str]
    date_1: Optional[str]
    date_2: Optional[str]
    date_3: Optional[str]
    section_counts: Dict[str, int]

def process_pdf(pdf_path: str, out_dir: str, dpi: int = 400, engine: str = "paddle", lang: str = "korean", poppler_path: Optional[str] = None) -> FileSummary:
    file_stem = pathlib.Path(pdf_path).stem
    work_dir = os.path.join(out_dir, file_stem)
    ensure_dir(work_dir)

    # 1) PDF -> PIL images
    _convert_kwargs: Dict[str, Any] = {"dpi": dpi}
    if poppler_path:
        _convert_kwargs["poppler_path"] = poppler_path
    pages = convert_from_path(pdf_path, **_convert_kwargs)
    page_results: List[PageResult] = []
    section_agg = {k: 0 for k in SECTION_KEYWORDS.keys()}

    engine_used = []
    for i, pil_img in enumerate(pages, start=1):
        img_cv = pil_to_cv(pil_img)
        # 2) OCR
        res = None
        if engine in ("paddle", "both"):
            res = ocr_with_paddle(img_cv, lang=lang)
            engine_used.append("paddle")
        elif engine == "tesseract":
            res = ocr_with_tesseract(img_cv)
            engine_used.append("tesseract")
        elif engine == "both":
            # (위에서 처리)
            pass

        # both 모드: paddle 실패 시 tesseract 폴백
        if engine == "both" and (res is None or not res.get("text")):
            try:
                res = ocr_with_tesseract(img_cv)
                engine_used.append("tesseract")
            except Exception:
                res = {"engine": "none", "lines": [], "text": ""}

        text = res.get("text", "") if res else ""
        # 3) 섹션 추정
        section, scores = score_sections(text)
        section_agg[section] = section_agg.get(section, 0) + 1

        # 4) 저장
        txt_path = os.path.join(work_dir, f"page_{i:03d}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

        json_path = os.path.join(work_dir, f"page_{i:03d}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)

        page_results.append(PageResult(
            page_index=i, section_guess=section, section_scores=scores,
            engine=res.get("engine", "unknown"),
            text_path=txt_path, json_path=json_path
        ))

        # 섹션별 텍스트도 별도로 저장(모아서)
        sec_file = os.path.join(work_dir, f"section_{sanitize_for_filename(section)}.txt")
        with open(sec_file, "a", encoding="utf-8") as f:
            f.write(f"\n\n---- page {i:03d} ----\n{text}\n")

    # 5) 파일 단위 요약(간단 규칙)
    # 심사의견 섹션이 있으면 그쪽 텍스트 우선으로 판정/금액/일자 탐색
    decision = None
    amounts: List[str] = []
    dates = {"date_1": None, "date_2": None, "date_3": None}

    def read_if_exists(path):
        return open(path, "r", encoding="utf-8").read() if os.path.exists(path) else ""

    opinion_text = read_if_exists(os.path.join(work_dir, f"section_{sanitize_for_filename('심사의견')}.txt"))
    summary_text = read_if_exists(os.path.join(work_dir, f"section_{sanitize_for_filename('요약/결재')}.txt"))

    search_order = [opinion_text, summary_text]
    if not any(search_order):
        # 폴백: 전체 페이지를 순회하며 일부 텍스트 사용
        for pr in page_results:
            search_order.append(read_if_exists(pr.text_path))

    for t in search_order:
        if t and decision is None:
            decision = extract_decision(t)
        if t and not amounts:
            amounts = extract_amounts(t)
        if t and all(v is None for v in dates.values()):
            dates = extract_dates(t)

    # 6) 요약 저장 + 반환
    engine_used_str = ",".join(sorted(set(engine_used))) if engine_used else engine
    summary = FileSummary(
        file=os.path.basename(pdf_path),
        engine_used=engine_used_str,
        decision=decision,
        amount_candidates=amounts,
        date_1=dates.get("date_1"),
        date_2=dates.get("date_2"),
        date_3=dates.get("date_3"),
        section_counts=section_agg
    )
    with open(os.path.join(work_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, ensure_ascii=False, indent=2)

    return summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_dir", required=True, help="PDF 폴더 경로")
    ap.add_argument("--out_dir", required=True, help="출력 폴더 경로")
    ap.add_argument("--dpi", type=int, default=400, help="PDF->이미지 DPI(기본 400)")
    ap.add_argument("--engine", choices=["paddle", "tesseract", "both"], default="paddle")
    ap.add_argument("--lang", default="korean", help="PaddleOCR 언어 (기본 korean)")
    ap.add_argument("--poppler_path", default=None, help="(선택) Windows 등에서 poppler 경로 지정")
    ap.add_argument("--tesseract_cmd", default=None, help="(선택) Windows에서 Tesseract 실행 파일 경로")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    # 글로벌 설정(Windows용)
    global _TESSERACT_CMD
    _TESSERACT_CMD = args.tesseract_cmd

    rows = []
    pdfs = [p for p in pathlib.Path(args.pdf_dir).glob("*.pdf")]
    if not pdfs:
        print(f"[경고] {args.pdf_dir}에 PDF가 없습니다.")
        sys.exit(0)

    for pdf in pdfs:
        print(f"[INFO] Processing: {pdf.name}")
        try:
            summary = process_pdf(str(pdf), args.out_dir, dpi=args.dpi, engine=args.engine, lang=args.lang, poppler_path=args.poppler_path)
            row = asdict(summary)
            # JSON 직렬화 가능한 형태로 변환
            row["section_counts"] = json.dumps(row["section_counts"], ensure_ascii=False)
            row["amount_candidates"] = json.dumps(row["amount_candidates"], ensure_ascii=False)
            rows.append(row)
        except Exception as e:
            print(f"[ERROR] {pdf.name}: {e}")
            rows.append({
                "file": pdf.name, "engine_used": args.engine, "decision": None,
                "amount_candidates": "[]", "date_1": None, "date_2": None, "date_3": None,
                "section_counts": "{}", "error": str(e)
            })

    if rows:
        df = pd.DataFrame(rows)
        csv_path = os.path.join(args.out_dir, "summary.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"[DONE] Summary saved to: {csv_path}")

if __name__ == "__main__":
    main()
