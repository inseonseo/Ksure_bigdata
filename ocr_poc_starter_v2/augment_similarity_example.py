#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OCR로 확대된 텍스트(aug_text)를 KoSimCSE 임베딩에 포함하여
간단한 유사사례 검색 데모를 수행합니다.

전제:
- ocr_poc.py 실행으로 out/<pdf_stem>/section_*.txt 생성
- collect_ocr_sections.py 로 aug_text CSV 생성

사용 예시:
python augment_similarity_example.py --aug_csv ./out/ocr_aug.csv --topk 5 \
  --improved_path C:\Users\wq240\Project\case\KoSimCSE\new\improved_insurance_system.py
"""
import argparse, os, sys
import pandas as pd
import numpy as np
from importlib.machinery import SourceFileLoader
from sklearn.metrics.pairwise import cosine_similarity

def load_simulator(improved_path: str):
    mod = SourceFileLoader('improved', improved_path).load_module()
    sim = mod.ImprovedInsuranceSystem()
    return sim

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug_csv", required=True, help="collect_ocr_sections.py의 출력 CSV")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--improved_path", required=True, help="improved_insurance_system.py 경로")
    args = ap.parse_args()

    df = pd.read_csv(args.aug_csv)
    # 텍스트가 빈 값인 샘플은 제외
    df = df[df["aug_text"].fillna("").str.strip() != ""]
    df = df.reset_index(drop=True)
    if df.empty:
        print("[WARN] aug_text가 비어 있습니다. OCR/수집 과정을 확인하세요.")
        sys.exit(0)

    sim = load_simulator(args.improved_path)

    # 전체 텍스트 임베딩
    texts = df["aug_text"].tolist()
    embs = sim.get_text_embeddings(texts, batch_size=8)  # 사용 환경에 맞게 배치 조정
    if embs is None:
        print("[INFO] get_text_embeddings가 None을 반환: 폴백(단어 빈도) 사용 예시는 생략합니다.")
        sys.exit(0)

    # 데모용: 상위 몇 개를 쿼리로 삼아 유사문서 검색
    queries = texts[:3]
    q_embs = embs[:3]
    sim_mat = cosine_similarity(q_embs, embs)

    for qi, q in enumerate(queries):
        scores = sim_mat[qi]
        top_idx = np.argsort(-scores)[:args.topk+1]  # 자기 자신 포함
        print("\n" + "="*60)
        print(f"[Query #{qi+1}]")
        print(q[:200].replace("\n"," ") + ("..." if len(q) > 200 else ""))
        print("- Top matches -")
        rank = 1
        for idx in top_idx:
            if idx == qi:  # 자기 자신은 건너뛰기
                continue
            print(f"{rank:>2}. idx={idx} score={scores[idx]:.3f}")
            print(df.loc[idx, "aug_text"][:200].replace("\n", " ") + ("..." if len(df.loc[idx, "aug_text"]) > 200 else ""))
            rank += 1
            if rank > args.topk:
                break

if __name__ == "__main__":
    main()
