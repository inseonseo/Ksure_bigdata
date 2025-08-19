#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
의존성 최소 버전: 표준 라이브러리만 사용하여 문자 n-그램 TF-IDF 코사인 유사도 계산

입력:
- --ocr_aug_csv: collect_ocr_sections.py 생성 CSV(utf-8 기본)
- --design_csv: 기존 사례 CSV(cp949 기본)

출력:
- 콘솔: 쿼리(문서)별 Top-5 매칭
- CSV: out_dir/similarity_aug_nolib.csv 에 Top-10 저장

예시 실행:
python compare_similarity_nolib.py \
  --ocr_aug_csv .\out\ocr_aug.csv \
  --design_csv .\data\design.csv \
  --design_text_col 사고설명 \
  --design_label_col 판정구분 \
  --encoding_design cp949 \
  --out_dir .\out\similarity \
  --print_topk 5 --save_topk 10
"""
import csv
import os
import math
import argparse
import pathlib
from collections import defaultdict, Counter


def read_csv_rows(path, encoding='utf-8'):
    with open(path, 'r', encoding=encoding, newline='') as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_csv_rows(path, rows, fieldnames, encoding='utf-8-sig'):
    with open(path, 'w', encoding=encoding, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def char_ngrams(text: str, nmin=2, nmax=5):
    text = text.replace('\n', ' ').strip()
    grams = []
    L = len(text)
    for n in range(nmin, nmax + 1):
        if L < n:
            continue
        for i in range(L - n + 1):
            grams.append(text[i:i+n])
    return grams


def build_corpus_index(base_texts, nmin=2, nmax=5):
    # DF 계산
    df_counts = Counter()
    doc_tokens = []  # list[Counter]
    for txt in base_texts:
        grams = char_ngrams(txt, nmin, nmax)
        c = Counter(grams)
        doc_tokens.append(c)
        df_counts.update(c.keys())

    N = len(base_texts)
    idf = {}
    for tok, df in df_counts.items():
        # IDF with smoothing
        idf[tok] = math.log((1 + N) / (1 + df)) + 1.0

    # TF-IDF 가중치 및 정규화(norm)
    base_vectors = []  # list[dict[token, weight]]
    base_norms = []
    for c in doc_tokens:
        vec = {}
        for tok, tf in c.items():
            vec[tok] = (1.0 + math.log(tf)) * idf.get(tok, 0.0)
        norm = math.sqrt(sum(w*w for w in vec.values())) or 1.0
        base_vectors.append(vec)
        base_norms.append(norm)

    # Inverted index: token -> list[(doc_id, weight)]
    inv = defaultdict(list)
    for di, vec in enumerate(base_vectors):
        for tok, w in vec.items():
            inv[tok].append((di, w))

    return idf, base_vectors, base_norms, inv


def vectorize_query(text, idf, nmin=2, nmax=5):
    grams = char_ngrams(text, nmin, nmax)
    c = Counter(grams)
    vec = {}
    for tok, tf in c.items():
        vec[tok] = (1.0 + math.log(tf)) * idf.get(tok, 0.0)
    norm = math.sqrt(sum(w*w for w in vec.values())) or 1.0
    return vec, norm


def topk_similarity(query_vec, query_norm, base_norms, inv, k=10):
    # 점수 누적(쿼리 토큰과 겹치는 문서만)
    scores = defaultdict(float)
    for tok, qw in query_vec.items():
        for di, bw in inv.get(tok, ()):  # list[(doc_id, weight)]
            scores[di] += qw * bw
    # 코사인 정규화
    for di in list(scores.keys()):
        scores[di] = scores[di] / (query_norm * base_norms[di])
    # 정렬 후 상위 k
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ocr_aug_csv', required=True)
    ap.add_argument('--design_csv', required=True)
    ap.add_argument('--design_text_col', default='사고설명')
    ap.add_argument('--design_id_col', default=None)
    ap.add_argument('--design_label_col', default='판정구분')
    ap.add_argument('--encoding_design', default='cp949')
    ap.add_argument('--encoding_ocr', default='utf-8')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--print_topk', type=int, default=5)
    ap.add_argument('--save_topk', type=int, default=10)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load OCR
    ocr_rows = read_csv_rows(args.ocr_aug_csv, encoding=args.encoding_ocr) 
    # 필요한 컬럼 확인
    for col in ('pdf_stem', 'aug_text'):
        if len(ocr_rows) == 0 or col not in ocr_rows[0]:
            raise SystemExit(f"[ERROR] {args.ocr_aug_csv}에 '{col}' 컬럼이 필요합니다.")
    ocr_rows = [r for r in ocr_rows if (r.get('aug_text') or '').strip()]
    if not ocr_rows:
        raise SystemExit('[ERROR] aug_text가 비어 있어 비교할 수 없습니다.')

    # Load design
    base_rows = read_csv_rows(args.design_csv, encoding=args.encoding_design)
    if not base_rows or args.design_text_col not in base_rows[0]:
        raise SystemExit(f"[ERROR] {args.design_csv}에 '{args.design_text_col}' 컬럼이 필요합니다.")
    base_texts = [(r.get(args.design_text_col) or '').strip() for r in base_rows]
    base_ids = [str(r.get(args.design_id_col)) if args.design_id_col else str(i) for i, r in enumerate(base_rows)]
    base_labels = [str(r.get(args.design_label_col) or '') for r in base_rows]

    # Build index
    idf, base_vectors, base_norms, inv = build_corpus_index(base_texts, nmin=2, nmax=5)

    # Compute
    results = []
    for r in ocr_rows:
        qname = r['pdf_stem']
        qtext = r['aug_text']
        qvec, qnorm = vectorize_query(qtext, idf, nmin=2, nmax=5)
        top = topk_similarity(qvec, qnorm, base_norms, inv, k=args.save_topk)

        print(f"\n[Query] {qname}.pdf")
        print('-' * 54)
        for rank, (bi, score) in enumerate(top[:args.print_topk], start=1):
            label = base_labels[bi] if base_labels[bi] else '-'
            preview = base_texts[bi].replace('\n', ' ')
            if len(preview) > 60:
                preview = preview[:60] + '...'
            print(f"#{rank} ({score:.3f}) [{label}]   {preview}")

        for rank, (bi, score) in enumerate(top, start=1):
            results.append({
                'query_pdf_stem': qname,
                'query_text': qtext,
                'rank': rank,
                'base_id': base_ids[bi],
                'base_text': base_texts[bi],
                'similarity': f"{score:.6f}",
                'base_label': base_labels[bi],
            })

    out_csv = str(out_dir / 'similarity_aug_nolib.csv')
    write_csv_rows(out_csv, results, fieldnames=['query_pdf_stem','query_text','rank','base_id','base_text','similarity','base_label'])
    print(f"[DONE] Saved CSV: {out_csv} (rows={len(results)})")


if __name__ == '__main__':
    main()


