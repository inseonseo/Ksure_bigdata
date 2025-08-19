#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OCR로 확장된 텍스트(aug_text)와 기존 design.csv의 사고설명 간 유사도 비교 데모

출력:
- 콘솔: 쿼리(문서)별 Top-5 매칭을 사람이 읽기 좋은 형태로 출력
- CSV: 쿼리(문서)별 Top-10 매칭을 저장 (컬럼 고정)

예시 실행:
python compare_similarity.py \
  --ocr_aug_csv .\out\ocr_aug.csv \
  --design_csv .\data\design.csv \
  --design_text_col 사고설명 \
  --design_label_col 판정구분 \
  --encoding_design cp949 \
  --out_dir .\out\similarity \
  --print_topk 5 --save_topk 10

비고:
- 벡터화는 한국어에 적합한 char_wb n-gram TF-IDF(2~5그램)를 사용해 추가 의존성 없이 간단히 구현
- design_id_col이 없으면 자동으로 row_index를 base_id로 사용
"""
import os
import argparse
import pathlib
import pandas as pd
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_ocr_aug(path: str, encoding: str = "utf-8") -> pd.DataFrame:
    df = pd.read_csv(path, encoding=encoding)
    # 필수 컬럼 존재 여부 확인
    if "pdf_stem" not in df.columns:
        # collect_ocr_sections.py 생성물은 pdf_stem을 가짐
        # 없으면 추정 불가 → 에러 처리
        raise ValueError("ocr_aug.csv에 'pdf_stem' 컬럼이 필요합니다.")
    if "aug_text" not in df.columns:
        raise ValueError("ocr_aug.csv에 'aug_text' 컬럼이 필요합니다.")
    df["aug_text"] = df["aug_text"].fillna("").astype(str)
    # 쿼리는 텍스트가 비어있지 않은 행으로 한정
    df = df[df["aug_text"].str.strip() != ""].reset_index(drop=True)
    return df


def load_design(
    path: str,
    text_col: str,
    id_col: str = None,
    label_col: str = None,
    encoding: str = "cp949",
) -> pd.DataFrame:
    base = pd.read_csv(path, encoding=encoding)
    if text_col not in base.columns:
        raise ValueError(f"design.csv에 '{text_col}' 컬럼이 필요합니다.")
    base_text = base[text_col].fillna("").astype(str)
    # base_id
    if id_col and id_col in base.columns:
        base_id = base[id_col].astype(str)
    else:
        base_id = pd.Series(range(len(base)), name="row_index").astype(str)
        id_col = "row_index"
    # base_label
    if label_col and label_col in base.columns:
        base_label = base[label_col].astype(str)
    else:
        base_label = pd.Series([""] * len(base))
        label_col = None
    out = pd.DataFrame({
        "base_id": base_id,
        "base_text": base_text,
        "base_label": base_label,
    })
    return out


def build_vectorizer():
    # 한국어/짧은 문장 대응을 위해 char_wb 2~5-gram 권장
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1)


def compute_similarity(
    queries: List[str],
    base_texts: List[str],
    print_topk: int,
    save_topk: int,
    base_ids: List[str],
    base_labels: List[str],
    query_names: List[str],
    out_csv_path: str,
):
    vectorizer = build_vectorizer()
    X_base = vectorizer.fit_transform(base_texts)
    X_query = vectorizer.transform(queries)

    sim_mat = cosine_similarity(X_query, X_base)
    rows = []

    for qi, qname in enumerate(query_names):
        scores = sim_mat[qi]
        # 상위 save_topk 인덱스(내림차순)
        top_idx = scores.argsort()[::-1][:save_topk]

        # 콘솔 Top-5 출력
        print(f"\n[Query] {qname}.pdf")
        print("-" * 54)
        for r, bi in enumerate(top_idx[:print_topk], start=1):
            label = base_labels[bi] if base_labels[bi] else "-"
            preview = base_texts[bi].replace("\n", " ")
            if len(preview) > 60:
                preview = preview[:60] + "..."
            print(f"#{r} ({scores[bi]:.3f}) [{label}]   {preview}")

        # CSV 저장용 행 구성(Top-10)
        for r, bi in enumerate(top_idx, start=1):
            rows.append({
                "query_pdf_stem": qname,
                "query_text": queries[qi],
                "rank": r,
                "base_id": base_ids[bi],
                "base_text": base_texts[bi],
                "similarity": float(scores[bi]),
                "base_label": base_labels[bi],
            })

    pd.DataFrame(rows).to_csv(out_csv_path, index=False, encoding="utf-8-sig")
    print(f"[DONE] Saved CSV: {out_csv_path} (rows={len(rows)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr_aug_csv", required=True, help="collect_ocr_sections.py의 출력 CSV")
    ap.add_argument("--design_csv", required=True, help="기존 사례 CSV (사고설명 포함)")
    ap.add_argument("--design_text_col", default="사고설명", help="기존 사례 텍스트 컬럼명")
    ap.add_argument("--design_id_col", default=None, help="기존 사례 ID 컬럼명 (없으면 row_index 사용)")
    ap.add_argument("--design_label_col", default="판정구분", help="기존 사례 라벨 컬럼명(지급/면책 등)")
    ap.add_argument("--encoding_design", default="cp949", help="기존 사례 CSV 인코딩")
    ap.add_argument("--encoding_ocr", default="utf-8", help="ocr_aug.csv 인코딩")
    ap.add_argument("--out_dir", required=True, help="결과 CSV 저장 폴더")
    ap.add_argument("--print_topk", type=int, default=5)
    ap.add_argument("--save_topk", type=int, default=10)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ocr_df = load_ocr_aug(args.ocr_aug_csv, encoding=args.encoding_ocr)
    base_df = load_design(
        args.design_csv,
        text_col=args.design_text_col,
        id_col=args.design_id_col,
        label_col=args.design_label_col,
        encoding=args.encoding_design,
    )

    # 1) OCR 확장 텍스트(aug_text)를 쿼리로 사용
    queries_aug = ocr_df["aug_text"].tolist()
    query_names = ocr_df["pdf_stem"].astype(str).tolist()
    base_texts = base_df["base_text"].tolist()
    base_ids = base_df["base_id"].astype(str).tolist()
    base_labels = base_df["base_label"].astype(str).tolist()
    out_csv_aug = str(out_dir / "similarity_aug.csv")
    print("\n=== OCR 확장 텍스트 기반 (aug_text) → Top-5 프린트 / Top-10 CSV 저장 ===")
    compute_similarity(
        queries_aug,
        base_texts,
        args.print_topk,
        args.save_topk,
        base_ids,
        base_labels,
        query_names,
        out_csv_aug,
    )

    # 2) 기존 사고설명만을 쿼리로 사용(비교용)
    # collect_ocr_sections의 병합으로 동일 pdf_stem에 기존 사고설명이 포함되어 있을 수 있음
    if "사고설명" in ocr_df.columns:
        desc_series = ocr_df["사고설명"].fillna("").astype(str)
        # 빈 텍스트 제외
        mask = desc_series.str.strip() != ""
        if mask.any():
            queries_desc = desc_series[mask].tolist()
            query_names_desc = ocr_df.loc[mask, "pdf_stem"].astype(str).tolist()
            out_csv_desc = str(out_dir / "similarity_desc.csv")
            print("\n=== 기존 사고설명만 기반 → Top-5 프린트 / Top-10 CSV 저장 ===")
            compute_similarity(
                queries_desc,
                base_texts,
                args.print_topk,
                args.save_topk,
                base_ids,
                base_labels,
                query_names_desc,
                out_csv_desc,
            )
        else:
            print("[INFO] OCR 합본 내 '사고설명' 컬럼이 비어 있어 설명-only 비교는 건너뜁니다.")
    else:
        print("[INFO] OCR 합본에 '사고설명' 컬럼이 없어 설명-only 비교는 건너뜁니다.")


if __name__ == "__main__":
    main()


