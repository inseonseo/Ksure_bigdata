import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder, StandardScaler
import re
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModel
import pickle
import os

class CaseSimilaritySearch:
    def __init__(self, data_path='data/testy.csv', fast_mode=False):
        """
        사고 유사도 검색기 초기화
        
        Args:
            data_path: 테스트 데이터 파일 경로
            fast_mode: 빠른 모드 (KoSimCSE 임베딩 생략)
        """
        self.data_path = data_path
        self.fast_mode = fast_mode
        self.data = None
        self.case_summary = None
        self.label_encoders = {}
        self.scaler = None
        self.kosimcse_model = None
        self.kosimcse_tokenizer = None
        self.text_embeddings = None
        
        # 임베딩 캐시 파일 경로 (현재 파일과 같은 디렉토리)
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.embedding_cache_path = os.path.join(current_dir, 'kosimcse_embeddings_cache.pkl')
        
        # 입력 필드 정의
        self.input_fields = {
            '신청대상구분': 'dropdown',
            '신청자': 'text', 
            '보험종목': 'dropdown',
            '사고유형명': 'dropdown',
            '수출자': 'text',
            '수입국': 'dropdown'
        }
        
        self.optional_fields = {
            '사고경위': 'textarea',
            '사고설명': 'textarea'
        }
        
        # 데이터 로드 및 전처리
        self.load_and_preprocess_data()
    
    def load_kosimcse_model(self):
        """KoSimCSE 모델 로드"""
        print("🤖 KoSimCSE 모델 로드 중...")
        try:
            # KoSimCSE 모델 경로 (Hugging Face 모델 사용)
            model_path = 'BM-K/KoSimCSE-roberta'
            
            self.kosimcse_tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.kosimcse_model = AutoModel.from_pretrained(model_path)
            print("✅ KoSimCSE 모델 로드 완료")
                
        except Exception as e:
            print(f"⚠️ KoSimCSE 모델 로드 실패: {str(e)}")
            print("⚠️ KoSimCSE 모델을 찾을 수 없습니다. 텍스트 유사도만 사용합니다.")
            self.kosimcse_model = None
            self.kosimcse_tokenizer = None
    
    def encode_text_with_kosimcse(self, texts):
        """KoSimCSE를 사용하여 텍스트 임베딩 생성"""
        if self.kosimcse_model is None or self.kosimcse_tokenizer is None:
            return None
        
        try:
            # 배치 크기 증가 (8 -> 32)
            batch_size = 32
            embeddings = []
            
            print(f"🤖 KoSimCSE 임베딩 생성 중... (총 {len(texts)}개, 배치 크기: {batch_size})")
            
            import time
            start_time = time.time()
            last_reported_progress = -1  # 마지막으로 보고된 진행률 추적
            
            for i in range(0, len(texts), batch_size):
                batch_start_time = time.time()
                batch_texts = texts[i:i+batch_size]
                
                # 진행률 및 남은 시간 표시 (10%마다 한 번만 표시)
                progress = (i / len(texts)) * 100
                progress_milestone = int(progress // 10) * 10  # 10의 배수로 반올림
                
                if (i == 0 or progress_milestone > last_reported_progress) and progress_milestone <= 100:
                    elapsed_time = time.time() - start_time
                    
                    if progress > 0:
                        # 남은 시간 계산
                        estimated_total_time = elapsed_time / (progress / 100)
                        remaining_time = estimated_total_time - elapsed_time
                        
                        # 시간 포맷팅
                        if remaining_time > 3600:  # 1시간 이상
                            remaining_str = f"{remaining_time/3600:.1f}시간"
                        elif remaining_time > 60:  # 1분 이상
                            remaining_str = f"{remaining_time/60:.1f}분"
                        else:
                            remaining_str = f"{remaining_time:.0f}초"
                        
                        print(f"   진행률: {progress_milestone}% ({i}/{len(texts)}) - 남은 시간: 약 {remaining_str}")
                    else:
                        print(f"   진행률: {progress_milestone}% ({i}/{len(texts)})")
                    
                    last_reported_progress = progress_milestone
                
                # 토크나이징
                inputs = self.kosimcse_tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=256,  # 최대 길이 단축 (512 -> 256)
                    return_tensors="pt"
                )
                
                # 임베딩 생성
                with torch.no_grad():
                    outputs = self.kosimcse_model(**inputs)
                    # [CLS] 토큰의 임베딩 사용
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    embeddings.extend(batch_embeddings)
                
                # 배치별 처리 시간 표시 (디버깅용)
                batch_time = time.time() - batch_start_time
                if i % (batch_size * 20) == 0:  # 20배치마다
                    print(f"   배치 처리 시간: {batch_time:.2f}초")
            
            total_time = time.time() - start_time
            print(f"✅ KoSimCSE 임베딩 생성 완료! (총 소요시간: {total_time/60:.1f}분)")
            return np.array(embeddings)
            
        except Exception as e:
            print(f"⚠️ KoSimCSE 임베딩 생성 실패: {str(e)}")
            return None
    
    def load_and_preprocess_data(self):
        """데이터 로드 및 사건별 전처리"""
        print("📊 데이터 로드 중...")
        
        # 데이터 로드
        self.data = pd.read_csv(self.data_path, encoding='cp949')
        
        # KoSimCSE 모델 로드
        self.load_kosimcse_model()
        
        # 사건별로 그룹핑하여 요약 정보 생성
        self.create_case_summary()
        
        # 텍스트 및 범주형 변수 전처리
        self.preprocess_features()
        
        print(f"✅ 데이터 로드 완료: {len(self.case_summary)}개 사건")
    
    def create_case_summary(self):
        """사건별로 모든 판정을 묶어서 요약 정보 생성"""
        print("🔧 사건별 요약 정보 생성 중...")
        
        # testy.csv의 컬럼 구조에 맞게 수정
        # 실제 컬럼명을 확인하고 수정 필요
        if '보상파일번호' in self.data.columns and '사고번호' in self.data.columns:
            # 사건별 그룹핑
            case_groups = self.data.groupby(['보상파일번호', '사고번호']).agg({
                # 기본 사고 정보 (첫 번째 값 사용)
                '사고접수일자': 'first' if '사고접수일자' in self.data.columns else None,
                '사고금액': 'first' if '사고금액' in self.data.columns else None,
                '사고유형명': 'first' if '사고유형명' in self.data.columns else None,
                '수입국': 'first' if '수입국' in self.data.columns else None,
                '사고설명': 'first' if '사고설명' in self.data.columns else None,
                '수출자': 'first' if '수출자' in self.data.columns else None,
                '보험종목': 'first' if '보험종목' in self.data.columns else None,
                '결제금액': 'first' if '결제금액' in self.data.columns else None,
                '수입자명': 'first' if '수입자명' in self.data.columns else None,
                
                # 판정 과정 정보 (모든 판정 포함)
                '판정일': list if '판정일' in self.data.columns else None,
                '판정구분': list if '판정구분' in self.data.columns else None,
                '판정사유': list if '판정사유' in self.data.columns else None,
                '판정금액': list if '판정금액' in self.data.columns else None,
                '판정진행상태': list if '판정진행상태' in self.data.columns else None,
                '판정회차': list if '판정회차' in self.data.columns else None
            }).reset_index()
            
            # None 값 제거
            case_groups = case_groups.dropna(axis=1, how='all')
        else:
            # 컬럼이 없는 경우 전체 데이터를 하나의 사건으로 처리
            case_groups = self.data.copy()
            case_groups['보상파일번호'] = 'default'
            case_groups['사고번호'] = range(len(case_groups))
        
        # 판정 패턴 분석
        case_groups = self.analyze_decision_patterns(case_groups)
        
        self.case_summary = case_groups
        print(f"✅ 사건별 요약 완료: {len(self.case_summary)}개 사건")
    
    def analyze_decision_patterns(self, case_groups):
        """판정 과정의 패턴을 분석하여 분류"""
        print("🔍 판정 패턴 분석 중...")
        
        for idx, case in case_groups.iterrows():
            if '판정구분' in case and isinstance(case['판정구분'], list):
                판정구분_list = case['판정구분']
                판정금액_list = case.get('판정금액', []) if '판정금액' in case else []
                
                # 패턴 분류
                if len(판정구분_list) == 1:
                    pattern = "단일판정"
                    summary = f"단일 판정: {판정구분_list[0]}"
                elif len(set(판정구분_list)) == 1:
                    pattern = "동일판정_반복"
                    summary = f"동일 판정 반복: {판정구분_list[0]} ({len(판정구분_list)}회)"
                elif "면책" in 판정구분_list and "지급" in 판정구분_list:
                    pattern = "면책지급_혼재"
                    면책_횟수 = 판정구분_list.count("면책")
                    지급_횟수 = 판정구분_list.count("지급")
                    summary = f"혼재 판정: 면책 {면책_횟수}회, 지급 {지급_횟수}회"
                else:
                    pattern = "기타"
                    summary = f"복합 판정: {len(판정구분_list)}회 판정"
                
                case_groups.at[idx, '판정패턴'] = pattern
                case_groups.at[idx, '판정요약'] = summary
                case_groups.at[idx, '판정횟수'] = len(판정구분_list)
            else:
                # 판정 정보가 없는 경우
                case_groups.at[idx, '판정패턴'] = "정보없음"
                case_groups.at[idx, '판정요약'] = "판정 정보 없음"
                case_groups.at[idx, '판정횟수'] = 0
        
        return case_groups
    
    def preprocess_features(self):
        """텍스트 및 범주형 변수 전처리"""
        print("🔧 특성 전처리 중...")
        
        # 텍스트 필드 결합 (사고설명 중심으로 변경)
        text_columns = []
        
        # 사고설명을 우선적으로 포함
        if '사고설명' in self.case_summary.columns:
            text_columns.append('사고설명')
        
        # 수출자/수입자는 보조 정보로만 사용 (가중치 낮춤)
        # 실제로는 사고설명에 업종 정보가 포함되어 있음
        if '수출자' in self.case_summary.columns:
            text_columns.append('수출자')
        
        if text_columns:
            self.case_summary['combined_text'] = self.case_summary[text_columns].fillna('').agg(' '.join, axis=1)
        else:
            self.case_summary['combined_text'] = ''
        
        # KoSimCSE 임베딩 생성
        if not self.fast_mode and self.kosimcse_model is not None and 'combined_text' in self.case_summary.columns:
            print("🤖 KoSimCSE 임베딩 처리 중...")
            
            # 캐시에서 로드 시도
            if not self.load_embeddings_cache():
                # 캐시가 없거나 데이터가 변경된 경우 새로 생성
                print("🔄 새로운 KoSimCSE 임베딩 생성 중...")
                texts = self.case_summary['combined_text'].fillna('').tolist()
                self.text_embeddings = self.encode_text_with_kosimcse(texts)
                
                if self.text_embeddings is not None:
                    print(f"✅ KoSimCSE 임베딩 생성 완료: {self.text_embeddings.shape}")
                    # 캐시에 저장
                    self.save_embeddings_cache()
                else:
                    print("⚠️ KoSimCSE 임베딩 생성 실패")
            else:
                print("✅ 캐시된 KoSimCSE 임베딩 사용")
        
        # 범주형 변수 인코딩 (검색 입력 정보만)
        categorical_columns = ['사고유형명', '수입국', '보험종목', '결제방법', '결제방법설명', '결제조건', '향후결제전망']
        for col in categorical_columns:
            if col in self.case_summary.columns:
                le = LabelEncoder()
                self.case_summary[f'{col}_encoded'] = le.fit_transform(
                    self.case_summary[col].astype(str)
                )
                self.label_encoders[col] = le
        
        # 수치형 변수 스케일링 (검색 입력 정보만)
        numerical_columns = []
        for col in ['사고금액', '결제금액', '수출보증금액']:
            if col in self.case_summary.columns:
                numerical_columns.append(col)
        
        if numerical_columns:
            self.scaler = StandardScaler()
            scaled_values = self.scaler.fit_transform(
                self.case_summary[numerical_columns].fillna(0)
            )
            # 스케일링된 값을 별도 컬럼으로 저장
            for i, col in enumerate(numerical_columns):
                self.case_summary[f'{col}_scaled'] = scaled_values[:, i]
        
        print("✅ 특성 전처리 완료")
    
    def get_available_options(self):
        """드롭다운 옵션들 반환"""
        options = {}
        
        for field in self.input_fields:
            if self.input_fields[field] == 'dropdown':
                if field in self.case_summary.columns:
                    options[field] = sorted(self.case_summary[field].dropna().unique().tolist())
        
        return options
    
    def search_similar_cases(self, query, top_k=5, verbose=True):
        """
        유사한 사고 사례 검색
        
        Args:
            query: 검색 쿼리 (딕셔너리) - 검색 입력 정보만
            top_k: 반환할 상위 결과 수
            verbose: 상세 출력 여부
        
        Returns:
            유사한 사례들의 리스트 (판정사유, 판정구분은 예측 결과로 표시)
        """
        if verbose:
            print(f"🔍 유사 사례 검색 중... (상위 {top_k}개)")
            print(f"📝 검색 입력: {list(query.keys())}")
        
        # 쿼리 전처리
        processed_query = self.preprocess_query(query)
        
        # 유사도 계산 (검색 입력 정보만 사용)
        similarities = self.calculate_similarity(processed_query)
        
        # 상위 k개 선택
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for i, idx in enumerate(top_indices):
            case = self.case_summary.iloc[idx]
            similarity = similarities[idx]
            
            # 상세 판정 과정 구성
            decision_process = self.create_decision_process(case)
            
            result = {
                'rank': i + 1,
                'similarity': float(similarity),
                'case_id': f"{case['보상파일번호']}_{case['사고번호']}",
                '판정횟수': int(case.get('판정횟수', 0)),
                'decision_process': decision_process,
                'case_info': {
                    '사고접수일자': case.get('사고접수일자', 'N/A'),
                    '사고금액': case.get('사고금액', 0),
                    '사고유형명': case.get('사고유형명', 'N/A'),
                    '수입국': case.get('수입국', 'N/A'),
                    '수출자': case.get('수출자', 'N/A'),
                    '보험종목': case.get('보험종목', 'N/A'),
                    '결제금액': case.get('결제금액', 0),
                    '수입자명': case.get('수입자명', 'N/A'),
                    '사고설명': case.get('사고설명', 'N/A')
                },
                # 예측 결과 (찾고자 하는 정보)
                'predicted_results': {
                    '판정구분': self.get_prediction_from_similar_cases(case, '판정구분'),
                    '판정사유': self.get_prediction_from_similar_cases(case, '판정사유')
                }
            }
            
            results.append(result)
            
            if verbose:
                self.print_case_result(result)
        
        return results
    
    def preprocess_query(self, query):
        """검색 쿼리 전처리"""
        processed = {}
        
        # 텍스트 필드 결합 (사고설명 중심)
        text_parts = []
        
        # 사고설명을 우선적으로 포함
        if '사고설명' in query and query['사고설명']:
            text_parts.append(str(query['사고설명']))
        
        # 수출자는 보조 정보로만 사용
        if '수출자' in query and query['수출자']:
            text_parts.append(str(query['수출자']))
        
        processed['combined_text'] = ' '.join(text_parts)
        
        # KoSimCSE 임베딩 생성 (쿼리용)
        if not self.fast_mode and self.kosimcse_model is not None and processed['combined_text']:
            query_embedding = self.encode_text_with_kosimcse([processed['combined_text']])
            if query_embedding is not None:
                processed['text_embedding'] = query_embedding[0]
        
        # 범주형 변수 인코딩
        for field in ['사고유형명', '수입국', '보험종목']:
            if field in query and query[field]:
                if field in self.label_encoders:
                    try:
                        processed[f'{field}_encoded'] = self.label_encoders[field].transform([query[field]])[0]
                    except:
                        processed[f'{field}_encoded'] = -1  # 알 수 없는 값
                else:
                    processed[f'{field}_encoded'] = 0
        
        # 수치형 변수 스케일링
        numerical_values = []
        for field in ['사고금액', '결제금액']:
            if field in query and query[field]:
                numerical_values.append(float(query[field]))
            else:
                numerical_values.append(0.0)
        
        if self.scaler:
            processed['numerical_features'] = self.scaler.transform([numerical_values])[0]
        
        return processed
    
    def calculate_similarity(self, processed_query):
        """유사도 계산 (검색 입력 정보만 사용)"""
        similarities = []
        
        for idx, case in self.case_summary.iterrows():
            # 텍스트 유사도 (KoSimCSE 사용)
            text_similarity = 0
            if not self.fast_mode and 'text_embedding' in processed_query and self.text_embeddings is not None:
                query_embedding = processed_query['text_embedding']
                case_embedding = self.text_embeddings[idx]
                
                # 코사인 유사도 계산
                similarity = cosine_similarity([query_embedding], [case_embedding])[0][0]
                text_similarity = max(0, similarity)  # 음수 값 방지
            elif 'combined_text' in processed_query and processed_query['combined_text']:
                # KoSimCSE가 없는 경우 기본 텍스트 유사도
                query_text = processed_query['combined_text']
                case_text = case.get('combined_text', '')
                
                # 간단한 키워드 기반 유사도
                query_words = set(query_text.lower().split())
                case_words = set(case_text.lower().split())
                
                if query_words and case_words:
                    intersection = query_words.intersection(case_words)
                    union = query_words.union(case_words)
                    text_similarity = len(intersection) / len(union) if union else 0
            
            # 범주형 변수 유사도 (검색 입력 정보만)
            categorical_similarity = 0
            categorical_matches = 0
            total_categorical = 0
            
            # 검색 입력 정보만 사용
            search_fields = ['사고유형명', '수입국', '보험종목']
            for field in search_fields:
                if f'{field}_encoded' in processed_query:
                    total_categorical += 1
                    if processed_query[f'{field}_encoded'] == case.get(f'{field}_encoded', -1):
                        categorical_matches += 1
            
            if total_categorical > 0:
                categorical_similarity = categorical_matches / total_categorical
            
            # 수치형 변수 유사도 (검색 입력 정보만)
            numerical_similarity = 0
            if 'numerical_features' in processed_query:
                query_numerical = processed_query['numerical_features']
                case_numerical = []
                # 검색 입력 정보만 사용 - 스케일링된 값 사용
                for col in ['사고금액', '결제금액']:
                    if f'{col}_scaled' in case:
                        case_numerical.append(case[f'{col}_scaled'])
                    else:
                        case_numerical.append(0)  # 스케일링된 값이 없으면 0
                
                if len(case_numerical) == len(query_numerical):
                    # 유클리드 거리를 유사도로 변환
                    distance = np.sqrt(np.sum((query_numerical - case_numerical) ** 2))
                    numerical_similarity = 1 / (1 + distance)
                else:
                    numerical_similarity = 0
            
            # 가중 평균 (사고설명 중심으로 가중치 조정)
            if not self.fast_mode and self.text_embeddings is not None:
                final_similarity = (
                    0.7 * text_similarity +  # 사고설명 중심 텍스트 유사도 (가중치 증가)
                    0.25 * categorical_similarity +  # 범주형 유사도
                    0.05 * numerical_similarity  # 수치형 유사도 (가중치 감소)
                )
            else:
                final_similarity = (
                    0.5 * text_similarity +  # 기본 텍스트 유사도
                    0.4 * categorical_similarity +  # 범주형 유사도
                    0.1 * numerical_similarity  # 수치형 유사도
                )
            
            similarities.append(final_similarity)
        
        return np.array(similarities)
    
    def create_decision_process(self, case):
        """판정 과정 상세 정보 생성"""
        process = []
        
        # 판정 관련 컬럼들이 있는지 확인
        판정_컬럼들 = ['판정일', '판정구분', '판정사유', '판정금액', '판정진행상태', '판정회차']
        
        if all(col in case for col in 판정_컬럼들):
            판정일_list = case['판정일'] if isinstance(case['판정일'], list) else [case['판정일']]
            판정구분_list = case['판정구분'] if isinstance(case['판정구분'], list) else [case['판정구분']]
            판정사유_list = case['판정사유'] if isinstance(case['판정사유'], list) else [case['판정사유']]
            판정금액_list = case['판정금액'] if isinstance(case['판정금액'], list) else [case['판정금액']]
            판정진행상태_list = case['판정진행상태'] if isinstance(case['판정진행상태'], list) else [case['판정진행상태']]
            판정회차_list = case['판정회차'] if isinstance(case['판정회차'], list) else [case['판정회차']]
            
            for i in range(len(판정구분_list)):
                process.append({
                    '회차': 판정회차_list[i] if i < len(판정회차_list) else i+1,
                    '날짜': 판정일_list[i] if i < len(판정일_list) else 'N/A',
                    '판정구분': 판정구분_list[i] if i < len(판정구분_list) else 'N/A',
                    '판정금액': 판정금액_list[i] if i < len(판정금액_list) else 0,
                    '판정사유': 판정사유_list[i] if i < len(판정사유_list) else 'N/A',
                    '진행상태': 판정진행상태_list[i] if i < len(판정진행상태_list) else 'N/A'
                })
        else:
            # 판정 정보가 없는 경우
            process.append({
                '회차': 1,
                '날짜': 'N/A',
                '판정구분': '정보없음',
                '판정금액': 0,
                '판정사유': '정보없음',
                '진행상태': '정보없음'
            })
        
        return process
    
    def print_case_result(self, result):
        """사례 결과 출력"""
        print(f"\n{'='*60}")
        print(f"🏆 순위: {result['rank']} (유사도: {result['similarity']:.3f})")
        print(f"📋 사건 ID: {result['case_id']}")
        
        # 예측 결과 표시
        predicted = result.get('predicted_results', {})
        print(f"🎯 예측 결과:")
        print(f"   - 판정구분: {predicted.get('판정구분', 'N/A')}")
        print(f"   - 판정사유: {predicted.get('판정사유', 'N/A')}")
        
        print(f"🔄 판정 횟수: {result['판정횟수']}회")
        
        print(f"\n📋 사건 정보:")
        case_info = result['case_info']
        print(f"   - 사고접수일자: {case_info['사고접수일자']}")
        print(f"   - 사고유형: {case_info['사고유형명']}")
        print(f"   - 수입국: {case_info['수입국']}")
        print(f"   - 수출자: {case_info['수출자']}")
        print(f"   - 보험종목: {case_info['보험종목']}")
        print(f"   - 사고금액: {case_info['사고금액']:,.0f}")
        print(f"   - 결제금액: {case_info['결제금액']:,.0f}")
        
        print(f"\n📋 판정 과정:")
        for i, decision in enumerate(result['decision_process']):
            print(f"   {i+1}차 판정:")
            print(f"     - 날짜: {decision['날짜']}")
            print(f"     - 판정: {decision['판정구분']}")
            print(f"     - 금액: {decision['판정금액']:,.0f}")
            print(f"     - 사유: {decision['판정사유']}")
            print(f"     - 상태: {decision['진행상태']}")
        
        print(f"{'='*60}")

    def get_prediction_from_similar_cases(self, case, field):
        """유사한 사례들의 판정 정보를 기반으로 예측"""
        if field in case and isinstance(case[field], list):
            # 리스트인 경우 (여러 판정이 있는 경우)
            if len(case[field]) > 0:
                # 가장 최근 판정 또는 가장 빈도가 높은 판정 반환
                return case[field][-1] if len(case[field]) > 0 else 'N/A'
            else:
                return 'N/A'
        elif field in case:
            # 단일 값인 경우
            return case[field]
        else:
            return 'N/A'

    def save_embeddings_cache(self):
        """KoSimCSE 임베딩을 캐시 파일에 저장"""
        if self.text_embeddings is not None:
            try:
                cache_data = {
                    'embeddings': self.text_embeddings,
                    'data_hash': hash(str(self.case_summary['combined_text'].tolist()))
                }
                with open(self.embedding_cache_path, 'wb') as f:
                    pickle.dump(cache_data, f)
                print(f"💾 임베딩 캐시 저장 완료: {self.embedding_cache_path}")
            except Exception as e:
                print(f"⚠️ 임베딩 캐시 저장 실패: {str(e)}")
    
    def load_embeddings_cache(self):
        """KoSimCSE 임베딩 캐시 파일에서 로드"""
        try:
            if os.path.exists(self.embedding_cache_path):
                with open(self.embedding_cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                
                # 캐시 데이터 구조 확인
                if 'embeddings' in cache_data and 'data_hash' in cache_data:
                    # 데이터 해시 확인 (데이터가 변경되었는지 체크)
                    if 'combined_text' in self.case_summary.columns:
                        current_hash = hash(str(self.case_summary['combined_text'].tolist()))
                        if cache_data['data_hash'] == current_hash:
                            self.text_embeddings = cache_data['embeddings']
                            print(f"📂 임베딩 캐시 로드 완료: {self.text_embeddings.shape}")
                            return True
                        else:
                            print("⚠️ 데이터가 변경되어 캐시를 무시합니다.")
                            return False
                    else:
                        print("⚠️ combined_text 컬럼이 없어 캐시를 무시합니다.")
                        return False
                else:
                    print("⚠️ 캐시 파일 구조가 올바르지 않습니다.")
                    return False
        except Exception as e:
            print(f"⚠️ 임베딩 캐시 로드 실패: {str(e)}")
            return False

# 사용 예시
if __name__ == "__main__":
    # 검색기 초기화
    search_engine = CaseSimilaritySearch()
    
    # 사용 가능한 옵션 확인
    options = search_engine.get_available_options()
    print("📋 사용 가능한 옵션:")
    for field, values in options.items():
        print(f"   {field}: {len(values)}개 옵션")
    
    # 검색 예시
    query = {
        '사고유형명': '지급거절',
        '수입국': '미국',
        '보험종목': '단기수출보험',
        '사고설명': '수입자가 지급을 거절함'
    }
    
    results = search_engine.search_similar_cases(query, top_k=3) 