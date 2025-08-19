#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OCR 결과 폴더(out/<파일명> 내 section_*.txt)를 모아
- 사고경위 텍스트
- 심사의견 텍스트
- 전체 합본 텍스트(사고설명 + 사고경위 + 심사의견)를
CSV로 정리합니다.
"""
import os, argparse, pathlib, pandas as pd

def read_text(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr_out_dir", required=True, help="ocr_poc.py 출력 폴더 (out)")
    ap.add_argument("--base_text_csv", required=False, help="원본 텍스트 CSV(사고설명 등)")
    ap.add_argument("--text_col", default="사고설명", help="원본 텍스트 컬럼명")
    ap.add_argument("--pdf_col", default="pdf_file", help="PDF 파일명 컬럼명(확장자 포함)")
    ap.add_argument("--out_csv", required=True, help="저장할 CSV 경로")
    ap.add_argument("--base_encoding", default="utf-8", help="원본 텍스트 CSV 인코딩 (예: cp949)")
    args = ap.parse_args()

    rows = []
    for pdf_dir in pathlib.Path(args.ocr_out_dir).iterdir():
        if not pdf_dir.is_dir():
            continue
        # 디렉토리명이 PDF stem이어야 함(ocr_poc가 그렇게 생성)
        pdf_stem = pdf_dir.name
        # 후보 섹션 파일명
        sec_opinion = pdf_dir / "section_심사의견.txt"
        sec_invest = pdf_dir / "section_사고조사_경위.txt"  # sanitize로 생성되는 이름 예시
        # 안전하게 다양한 패턴 시도
        candidates = [
            ("심사의견", ["section_심사의견.txt", "section_종합의견.txt", "section_심사의견_종합의견.txt"]),
            ("사고경위", ["section_사고조사_경위.txt", "section_사고조사_내용.txt", "section_사고경위.txt"]),
        ]
        data = {"pdf_stem": pdf_stem, "심사의견": "", "사고경위": ""}
        for key, names in candidates:
            for nm in names:
                p = pdf_dir / nm
                if p.exists():
                    data[key] = read_text(p)
                    break
        rows.append(data)

    df = pd.DataFrame(rows)

    # 원본 텍스트 CSV가 있으면 조인(pdf_file 또는 stem 기준)
    if args.base_text_csv and os.path.exists(args.base_text_csv):
        base = pd.read_csv(args.base_text_csv, encoding=args.base_encoding)
        # stem 생성
        base["pdf_stem"] = base[args.pdf_col].astype(str).str.replace(r"\.pdf$", "", regex=True)
        df = df.merge(base[[args.text_col, "pdf_stem"]], on="pdf_stem", how="left")
    else:
        df[args.text_col] = ""

    # 합본 텍스트(사고설명 + 사고경위 + 심사의견)
    def build_aug(row):
        parts = []
        if row[args.text_col]:
            parts.append(str(row[args.text_col]))
        if row["사고경위"]:
            parts.append("[사고경위]\n" + str(row["사고경위"]))
        if row["심사의견"]:
            parts.append("[심사의견]\n" + str(row["심사의견"]))
        return "\n\n".join(parts).strip()

    df["aug_text"] = df.apply(build_aug, axis=1)
    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(f"[DONE] Saved: {args.out_csv} (rows={len(df)})")

if __name__ == "__main__":
    main()
