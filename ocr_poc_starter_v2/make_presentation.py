#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OCR 산출물(out/<pdf_stem>/)을 모아 발표/공유용 요약 Markdown을 생성합니다.

생성물(기본): out/presentation/<prefix>_<pdf_stem>.md
 - prefix: 추정 판정(면책/지급/가지급/불명)에 따라 파일명 접두어 지정
 - 내용: 파일명, 엔진, 판정 후보, 금액/일자 후보, 섹션(사고경위/심사의견) 발췌

사용 예시:
python make_presentation.py --ocr_out_dir ./out --dst_dir ./out/presentation --max_section_chars 1200
"""
import os
import json
import argparse
import pathlib


def read_text(path: pathlib.Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


def pick_section_text(pdf_dir: pathlib.Path) -> dict:
    # collect_ocr_sections.py와 동일한 후보 집합을 사용하여 안전하게 탐색
    candidates = {
        "심사의견": [
            "section_심사의견.txt",
            "section_종합의견.txt",
            "section_심사의견_종합의견.txt",
        ],
        "사고경위": [
            "section_사고조사_경위.txt",
            "section_사고조사_내용.txt",
            "section_사고경위.txt",
        ],
    }
    data = {"심사의견": "", "사고경위": ""}
    for key, names in candidates.items():
        for nm in names:
            p = pdf_dir / nm
            if p.exists():
                data[key] = read_text(p)
                break
    return data


def load_summary(pdf_dir: pathlib.Path) -> dict:
    p = pdf_dir / "summary.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def decision_prefix(decision: str) -> str:
    if not decision:
        return "불명"
    if decision in ("면책", "지급", "가지급"):
        return decision
    return "불명"


def truncate(text: str, max_chars: int) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... (중략) ..."


def build_markdown(stem: str, summary: dict, sections: dict, max_section_chars: int) -> str:
    decision = summary.get("decision")
    engine_used = summary.get("engine_used", "")
    amount_candidates = summary.get("amount_candidates") or []
    if isinstance(amount_candidates, str):
        # CSV에서 읽은 문자열이 들어오는 경우 대비
        try:
            amount_candidates = json.loads(amount_candidates)
        except Exception:
            amount_candidates = [amount_candidates]

    dates = [summary.get("date_1"), summary.get("date_2"), summary.get("date_3")]
    dates = [d for d in dates if d]

    md = []
    md.append(f"# {stem}")
    md.append("")
    md.append(f"- 파일명: {stem}.pdf")
    md.append(f"- OCR 엔진: {engine_used}")
    md.append(f"- 판정 후보: {decision or '불명'}")
    md.append(f"- 금액 후보: {', '.join(amount_candidates) if amount_candidates else '-'}")
    md.append(f"- 일자 후보: {', '.join(dates) if dates else '-'}")
    md.append("")
    if sections.get("사고경위"):
        md.append("## 사고경위")
        md.append("")
        md.append(truncate(sections["사고경위"], max_section_chars))
        md.append("")
    if sections.get("심사의견"):
        md.append("## 심사의견")
        md.append("")
        md.append(truncate(sections["심사의견"], max_section_chars))
        md.append("")

    # 고정 가드레일 문구
    md.append(
        "> 참고: 본 결과는 유사사례 비교 지표 제공 목적이며, 지급/면책 판단을 대체하지 않습니다."
    )
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr_out_dir", required=True, help="ocr_poc.py 출력 폴더 (out)")
    ap.add_argument("--dst_dir", required=True, help="발표/공유용 Markdown 저장 폴더")
    ap.add_argument("--max_section_chars", type=int, default=1200, help="섹션 발췌 최대 길이")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.ocr_out_dir)
    dst_dir = pathlib.Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    made = 0
    for pdf_dir in out_dir.iterdir():
        if not pdf_dir.is_dir():
            continue
        stem = pdf_dir.name
        summary = load_summary(pdf_dir)
        sections = pick_section_text(pdf_dir)
        decision = (summary.get("decision") or "").strip()
        prefix = decision_prefix(decision)
        file_name = f"{prefix}_{stem}.md"
        md = build_markdown(stem, summary, sections, args.max_section_chars)
        (dst_dir / file_name).write_text(md, encoding="utf-8")
        made += 1

    print(f"[DONE] Created {made} markdown files in: {str(dst_dir)}")


if __name__ == "__main__":
    main()


