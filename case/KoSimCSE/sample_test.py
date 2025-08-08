#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
샘플 데이터로 빠른 테스트
전체 데이터의 일부만 사용하여 KoSimCSE 테스트
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from transformers import AutoTokenizer, AutoModel
import time

class SampleCaseSearch:
    def __init__(self, data_path='data/testy.csv', sample_size=1000):
        """
        샘플 데이터로 빠른 테스트
        
        Args:
            data_path: 데이터 파일 경로
            sample_size: 샘플 크기 (기본 1000개)
        """
        self.data_path = data_path
        self.sample_size = sample_size
        self.data = None
        self.case_summary = None
        self.kosimcse_model = None
        self.kosimcse_tokenizer = None
        self.text_embeddings = None
        
        print(f"🚀 샘플 테스트 시작 (데이터 {sample_size}개)")
        self.load_and_preprocess_data()
    
    def load_kosimcse_model(self):
        """KoSimCSE 모델 로드"""
        print("🤖 KoSimCSE 모델 로드 중...")
        try:
            model_path = 'BM-K/KoSimCSE-roberta'
            self.kosimcse_tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.kosimcse_model = AutoModel.from_pretrained(model_path)
            print("✅ KoSimCSE 모델 로드 완료")
        except Exception as e:
            print(f"⚠️ KoSimCSE 모델 로드 실패: {str(e)}")
            self.kosimcse_model = None
            self.kosimcse_tokenizer = None
    
    def load_and_preprocess_data(self):
        """샘플 데이터 로드 및 전처리"""
        print("📊 샘플 데이터 로드 중...")
        
        # 전체 데이터 로드
        full_data = pd.read_csv(self.data_path, encoding='cp949')
        print(f"   전체 데이터: {len(full_data)}개")
        
        # 샘플링 (랜덤)
        self.data = full_data.sample(n=min(self.sample_size, len(full_data)), random_state=42)
        print(f"   샘플 데이터: {len(self.data)}개")
        
        # KoSimCSE 모델 로드
        self.load_kosimcse_model()
        
        # 사건별 요약 생성
        self.create_case_summary()
        
        # 텍스트 임베딩 생성
        self.create_text_embeddings()
        
        print(f"✅ 샘플 데이터 처리 완료: {len(self.case_summary)}개 사건")
    
    def create_case_summary(self):
        """사건별 요약 정보 생성"""
        print("🔧 사건별 요약 정보 생성 중...")
        
        if '보상파일번호' in self.data.columns and '사고번호' in self.data.columns:
            # 사건별 그룹핑
            case_groups = self.data.groupby(['보상파일번호', '사고번호']).agg({
                '사고접수일자': 'first' if '사고접수일자' in self.data.columns else None,
                '사고금액': 'first' if '사고금액' in self.data.columns else None,
                '사고유형명': 'first' if '사고유형명' in self.data.columns else None,
                '수입국': 'first' if '수입국' in self.data.columns else None,
                '사고설명': 'first' if '사고설명' in self.data.columns else None,
                '수출자': 'first' if '수출자' in self.data.columns else None,
                '보험종목': 'first' if '보험종목' in self.data.columns else None,
                '판정구분': list if '판정구분' in self.data.columns else None,
                '판정사유': list if '판정사유' in self.data.columns else None,
            }).reset_index()
            
            case_groups = case_groups.dropna(axis=1, how='all')
        else:
            case_groups = self.data.copy()
            case_groups['보상파일번호'] = 'default'
            case_groups['사고번호'] = range(len(case_groups))
        
        self.case_summary = case_groups
        print(f"✅ 사건별 요약 완료: {len(self.case_summary)}개 사건")
    
    def create_text_embeddings(self):
        """KoSimCSE 텍스트 임베딩 생성"""
        if self.kosimcse_model is None:
            print("⚠️ KoSimCSE 모델이 없어 텍스트 임베딩을 건너뜁니다.")
            return
        
        print("🤖 KoSimCSE 임베딩 생성 중...")
        
        # 텍스트 결합
        text_columns = []
        for col in ['사고설명', '수출자']:
            if col in self.case_summary.columns:
                text_columns.append(col)
        
        if text_columns:
            self.case_summary['combined_text'] = self.case_summary[text_columns].fillna('').agg(' '.join, axis=1)
        else:
            self.case_summary['combined_text'] = ''
        
        # 임베딩 생성
        texts = self.case_summary['combined_text'].fillna('').tolist()
        self.text_embeddings = self.encode_text_with_kosimcse(texts)
        
        if self.text_embeddings is not None:
            print(f"✅ KoSimCSE 임베딩 생성 완료: {self.text_embeddings.shape}")
        else:
            print("⚠️ KoSimCSE 임베딩 생성 실패")
    
    def encode_text_with_kosimcse(self, texts):
        """KoSimCSE를 사용하여 텍스트 임베딩 생성"""
        if self.kosimcse_model is None or self.kosimcse_tokenizer is None:
            return None
        
        try:
            batch_size = 16  # 샘플용으로 작은 배치
            embeddings = []
            
            print(f"   배치 크기: {batch_size}, 총 텍스트: {len(texts)}개")
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # 진행률 표시
                if i % (batch_size * 5) == 0:
                    progress = (i / len(texts)) * 100
                    print(f"   진행률: {progress:.1f}% ({i}/{len(texts)})")
                
                # 토크나이징
                inputs = self.kosimcse_tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=128,  # 샘플용으로 더 짧게
                    return_tensors="pt"
                )
                
                # 임베딩 생성
                with torch.no_grad():
                    outputs = self.kosimcse_model(**inputs)
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    embeddings.extend(batch_embeddings)
            
            print("✅ KoSimCSE 임베딩 생성 완료!")
            return np.array(embeddings)
            
        except Exception as e:
            print(f"⚠️ KoSimCSE 임베딩 생성 실패: {str(e)}")
            return None
    
    def search_similar_cases(self, query, top_k=3):
        """유사한 사고 사례 검색"""
        print(f"🔍 유사 사례 검색 중... (상위 {top_k}개)")
        
        # 쿼리 텍스트 생성
        query_text = ' '.join([str(v) for v in query.values() if v])
        
        # 쿼리 임베딩 생성
        if self.kosimcse_model is not None:
            query_embedding = self.encode_text_with_kosimcse([query_text])
            if query_embedding is not None:
                query_embedding = query_embedding[0]
            else:
                query_embedding = None
        else:
            query_embedding = None
        
        # 유사도 계산
        similarities = []
        for idx, case in self.case_summary.iterrows():
            similarity = 0
            
            # 텍스트 유사도
            if query_embedding is not None and self.text_embeddings is not None:
                case_embedding = self.text_embeddings[idx]
                similarity = cosine_similarity([query_embedding], [case_embedding])[0][0]
                similarity = max(0, similarity)
            
            # 범주형 유사도 추가
            categorical_similarity = 0
            if '사고유형명' in query and '사고유형명' in case:
                if query['사고유형명'] == case['사고유형명']:
                    categorical_similarity += 0.5
            
            if '수입국' in query and '수입국' in case:
                if query['수입국'] == case['수입국']:
                    categorical_similarity += 0.3
            
            if '보험종목' in query and '보험종목' in case:
                if query['보험종목'] == case['보험종목']:
                    categorical_similarity += 0.2
            
            # 최종 유사도 (텍스트 70%, 범주형 30%)
            final_similarity = 0.7 * similarity + 0.3 * categorical_similarity
            similarities.append(final_similarity)
        
        # 상위 k개 선택
        similarities = np.array(similarities)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for i, idx in enumerate(top_indices):
            case = self.case_summary.iloc[idx]
            similarity = similarities[idx]
            
            result = {
                'rank': i + 1,
                'similarity': float(similarity),
                'case_id': f"{case['보상파일번호']}_{case['사고번호']}",
                'case_info': {
                    '사고유형명': case.get('사고유형명', 'N/A'),
                    '수입국': case.get('수입국', 'N/A'),
                    '보험종목': case.get('보험종목', 'N/A'),
                    '사고설명': case.get('사고설명', 'N/A')[:100] + '...' if len(str(case.get('사고설명', ''))) > 100 else case.get('사고설명', 'N/A')
                },
                'predicted_results': {
                    '판정구분': case.get('판정구분', ['N/A'])[-1] if isinstance(case.get('판정구분'), list) and case.get('판정구분') else 'N/A',
                    '판정사유': case.get('판정사유', ['N/A'])[-1] if isinstance(case.get('판정사유'), list) and case.get('판정사유') else 'N/A'
                }
            }
            results.append(result)
        
        return results

def sample_test():
    print("🚀 샘플 데이터로 KoSimCSE 테스트")
    start_time = time.time()
    
    # 샘플 검색기 초기화 (1000개 데이터)
    search_engine = SampleCaseSearch(sample_size=1000)
    
    init_time = time.time() - start_time
    print(f"✅ 초기화 완료: {init_time:.2f}초")
    
    # 검색 테스트
    test_query = {
        '사고유형명': '지급거절',
        '수입국': '미국',
        '보험종목': '단기수출보험',
        '사고설명': '수입자가 품질 문제를 이유로 지급을 거절함. 신생 업체의 첫 수출 건에서 발생한 문제입니다.',
        '수출자': '신생 전자제품 수출업체'
    }
    
    print(f"\n🔍 검색 쿼리: {test_query}")
    
    search_start = time.time()
    results = search_engine.search_similar_cases(test_query, top_k=3)
    search_time = time.time() - search_start
    
    print(f"✅ 검색 완료: {search_time:.2f}초")
    
    # 결과 출력
    for i, result in enumerate(results):
        print(f"\n🏆 {i+1}위 (유사도: {result['similarity']:.3f})")
        print(f"   사건 ID: {result['case_id']}")
        print(f"   사고유형: {result['case_info']['사고유형명']}")
        print(f"   수입국: {result['case_info']['수입국']}")
        print(f"   예측 판정구분: {result['predicted_results']['판정구분']}")
        print(f"   예측 판정사유: {result['predicted_results']['판정사유']}")
    
    total_time = time.time() - start_time
    print(f"\n✅ 전체 테스트 완료: {total_time:.2f}초")
    print("💡 이제 전체 데이터로 실행해보세요!")

if __name__ == "__main__":
    sample_test() 