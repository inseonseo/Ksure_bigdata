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

class ComprehensiveSimilaritySearch:
    def __init__(self, data_path='data/testy.csv'):
        """
        종합 사고 유사도 검색기 초기화 (모든 특성 사용)
        
        Args:
            data_path: 테스트 데이터 파일 경로
        """
        self.data_path = data_path
        self.data = None
        self.case_summary = None
        self.label_encoders = {}
        self.scaler = None
        self.kosimcse_model = None
        self.kosimcse_tokenizer = None
        self.text_embeddings = None
        
        # 임베딩 캐시 파일 경로 (현재 파일과 같은 디렉토리)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.embedding_cache_path = os.path.join(current_dir, 'comprehensive_embeddings_cache.pkl')
        
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
            # 배치 크기 설정
            batch_size = 32
            embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # 토크나이징
                inputs = self.kosimcse_tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                
                # 임베딩 생성
                with torch.no_grad():
                    outputs = self.kosimcse_model(**inputs)
                    # [CLS] 토큰의 임베딩 사용
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    embeddings.extend(batch_embeddings)
            
            return np.array(embeddings)
            
        except Exception as e:
            print(f"⚠️ KoSimCSE 임베딩 생성 실패: {str(e)}")
            return None
    
    def load_and_preprocess_data(self):
        """데이터 로드 및 전처리"""
        print("📊 데이터 로드 중...")
        
        try:
            # testy.csv 파일 로드 (cp949 인코딩)
            self.data = pd.read_csv(self.data_path, encoding='cp949')
            print(f"✅ 데이터 로드 완료: {len(self.data)}개 레코드")
            
            # KoSimCSE 모델 로드
            self.load_kosimcse_model()
            
            # 사례 요약 생성
            self.create_case_summary()
            
            # 특성 전처리
            self.preprocess_features()
            
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {str(e)}")
            raise
    
    def create_case_summary(self):
        """사건별로 그룹화하여 사례 요약 생성"""
        print("📋 사례 요약 생성 중...")
        
        # 필요한 컬럼들이 있는지 확인
        required_columns = ['보상파일번호', '사고번호']
        available_columns = [col for col in required_columns if col in self.data.columns]
        
        if len(available_columns) < 2:
            print("⚠️ 보상파일번호 또는 사고번호 컬럼이 없습니다. 전체 데이터를 사용합니다.")
            self.case_summary = self.data.copy()
            return
        
        # 사건별 그룹화 (보상파일번호 + 사고번호)
        case_groups = self.data.groupby(['보상파일번호', '사고번호'])
        
        case_summaries = []
        for (보상파일번호, 사고번호), group in case_groups:
            case_summary = {
                '보상파일번호': 보상파일번호,
                '사고번호': 사고번호
            }
            
            # 단일 값 컬럼들 (첫 번째 값 사용)
            single_value_columns = [
                '사고접수일자', '사고유형명', '수입국', '수출자', '보험종목',
                '사고금액', '결제금액', '수출보증금액', '사고설명', '수입자명'
            ]
            
            for col in single_value_columns:
                if col in group.columns:
                    case_summary[col] = group[col].iloc[0] if not group[col].isna().all() else None
            
            # 리스트로 저장할 컬럼들 (모든 판정 정보)
            list_columns = [
                '판정일', '판정결재일', '판정진행상태', '사고진행상태',
                '판정구분', '판정금액', '판정사유', '판정회차',
                '결제방법', '결제방법설명', '결제조건', '향후결제전망',
                '수출보증금액통화', '결제금액통화', '사고금액통화', '판정금액통화'
            ]
            
            for col in list_columns:
                if col in group.columns:
                    values = group[col].dropna().unique().tolist()
                    case_summary[col] = values if values else []
            
            case_summaries.append(case_summary)

        self.case_summary = pd.DataFrame(case_summaries)
        print(f"✅ 사례 요약 생성 완료: {len(self.case_summary)}개 사례")
    
    def analyze_decision_patterns(self, case_group):
        """판정 패턴 분석"""
        if '판정구분' not in case_group.columns:
            return {'패턴': '정보없음', '요약': '판정 정보 없음'}
        
        판정구분들 = case_group['판정구분'].dropna().tolist()
        
        if not 판정구분들:
            return {'패턴': '정보없음', '요약': '판정 정보 없음'}
        
        # 패턴 분석
        unique_판정구분 = list(set(판정구분들))
        
        if len(unique_판정구분) == 1:
            패턴 = f"단일 판정: {unique_판정구분[0]}"
        else:
            패턴 = f"복합 판정: {' → '.join(unique_판정구분)}"
        
        # 요약 생성
        판정_빈도 = {}
        for 판정 in 판정구분들:
            판정_빈도[판정] = 판정_빈도.get(판정, 0) + 1
        
        요약 = ", ".join([f"{판정}({빈도}회)" for 판정, 빈도 in 판정_빈도.items()])
        
        return {'패턴': 패턴, '요약': 요약}
    
    def preprocess_features(self):
        """텍스트 및 범주형 변수 전처리 (모든 특성 사용)"""
        print("🔧 특성 전처리 중...")
        
        # 텍스트 필드 결합 (모든 텍스트 정보 포함)
        text_columns = []
        for col in ['사고설명']:
            if col in self.case_summary.columns:
                text_columns.append(col)
        
        if text_columns:
            self.case_summary['combined_text'] = self.case_summary[text_columns].fillna('').agg(' '.join, axis=1)
        else:
            self.case_summary['combined_text'] = ''
        
        # KoSimCSE 임베딩 생성
        if self.kosimcse_model is not None and 'combined_text' in self.case_summary.columns:
            print("🤖 KoSimCSE 임베딩 생성 중...")
            texts = self.case_summary['combined_text'].fillna('').tolist()
            self.text_embeddings = self.encode_text_with_kosimcse(texts)
            
            if self.text_embeddings is not None:
                print(f"✅ KoSimCSE 임베딩 생성 완료: {self.text_embeddings.shape}")
            else:
                print("⚠️ KoSimCSE 임베딩 생성 실패")
        
        # 범주형 변수 인코딩 (사용자가 요청한 모든 특성 포함)
        categorical_columns = [
            '판정진행상태', '사고진행상태', '수출자', '수입자', 
            '수출보증금액통화', '결제금액통화', '사고금액통화', 
            '사고유형명', '수입국', '보험종목', '결제방법', 
            '결제방법설명', '결제조건', '향후결제전망', '수입자명',
            '판정구분', '판정사유'  # 사용자가 요청한 추가 특성
        ]
        
        for col in categorical_columns:
            if col in self.case_summary.columns:
                # 리스트 형태의 컬럼은 첫 번째 값만 사용
                if self.case_summary[col].apply(lambda x: isinstance(x, list)).any():
                    # 리스트의 첫 번째 값 또는 빈 리스트인 경우 None
                    values = self.case_summary[col].apply(
                        lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None
                    )
                else:
                    values = self.case_summary[col]
                
                le = LabelEncoder()
                self.case_summary[f'{col}_encoded'] = le.fit_transform(
                    values.astype(str)
                )
                self.label_encoders[col] = le
        
        # 수치형 변수 스케일링 (모든 수치형 특성 포함)
        numerical_columns = []
        for col in ['사고금액', '결제금액', '수출보증금액', '판정금액']:
            if col in self.case_summary.columns:
                # 리스트 형태의 컬럼은 첫 번째 값만 사용
                if self.case_summary[col].apply(lambda x: isinstance(x, list)).any():
                    values = self.case_summary[col].apply(
                        lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 0
                    )
                else:
                    values = self.case_summary[col]
                
                self.case_summary[col] = values
                numerical_columns.append(col)
        
        if numerical_columns:
            self.scaler = StandardScaler()
            self.case_summary[numerical_columns] = self.scaler.fit_transform(
                self.case_summary[numerical_columns].fillna(0)
            )
        
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
        유사한 사고 사례 검색 (모든 특성 사용)
        
        Args:
            query: 검색 쿼리 (딕셔너리)
            top_k: 반환할 상위 결과 수
            verbose: 상세 출력 여부
        
        Returns:
            유사한 사례들의 리스트
        """
        if verbose:
            print(f"🔍 유사 사례 검색 중... (상위 {top_k}개)")
            print(f"📝 검색 입력: {list(query.keys())}")
        
        # 쿼리 전처리
        processed_query = self.preprocess_query(query)
        
        # 유사도 계산 (모든 특성 사용)
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
                'case_info': {
                    '사고접수일자': case.get('사고접수일자', 'N/A'),
                    '사고유형명': case.get('사고유형명', 'N/A'),
                    '수입국': case.get('수입국', 'N/A'),
                    '수출자': case.get('수출자', 'N/A'),
                    '보험종목': case.get('보험종목', 'N/A'),
                    '사고금액': case.get('사고금액', 0),
                    '결제금액': case.get('결제금액', 0),
                    '수출보증금액': case.get('수출보증금액', 0),
                    '판정금액': case.get('판정금액', 0)
                },
                'decision_process': decision_process
            }
            
            results.append(result)
        
        if verbose:
            print(f"✅ 검색 완료: {len(results)}개 결과")
        
        return results
    
    def preprocess_query(self, query):
        """검색 쿼리 전처리"""
        processed_query = {}
        
        # 텍스트 필드 처리
        text_parts = []
        for field in ['사고설명']:
            if field in query and query[field]:
                text_parts.append(str(query[field]))
        
        if text_parts:
            combined_text = ' '.join(text_parts)
            processed_query['combined_text'] = combined_text
            
            # KoSimCSE 임베딩 생성
            if self.kosimcse_model is not None:
                query_embedding = self.encode_text_with_kosimcse([combined_text])
                if query_embedding is not None:
                    processed_query['text_embedding'] = query_embedding[0]
        
        # 범주형 변수 인코딩
        categorical_columns = [
            '판정진행상태', '사고진행상태', '수출자', '수입자', 
            '수출보증금액통화', '결제금액통화', '사고금액통화', 
            '사고유형명', '수입국', '보험종목', '결제방법', 
            '결제방법설명', '결제조건', '향후결제전망', '수입자명',
            '판정구분', '판정사유'
        ]
        
        for col in categorical_columns:
            if col in query and query[col]:
                if col in self.label_encoders:
                    try:
                        encoded_value = self.label_encoders[col].transform([str(query[col])])[0]
                        processed_query[f'{col}_encoded'] = encoded_value
                    except:
                        processed_query[f'{col}_encoded'] = -1
        
        # 수치형 변수 처리
        numerical_features = []
        for col in ['사고금액', '결제금액', '수출보증금액', '판정금액']:
            if col in query and query[col]:
                try:
                    value = float(query[col])
                    numerical_features.append(value)
                except:
                    numerical_features.append(0)
            else:
                numerical_features.append(0)
        
        if self.scaler is not None:
            numerical_features = self.scaler.transform([numerical_features])[0]
        
        processed_query['numerical_features'] = np.array(numerical_features)
        
        return processed_query
    
    def calculate_similarity(self, processed_query):
        """유사도 계산 (모든 특성 사용)"""
        similarities = []
        
        for idx, case in self.case_summary.iterrows():
            # 텍스트 유사도 (KoSimCSE 사용)
            text_similarity = 0
            if 'text_embedding' in processed_query and self.text_embeddings is not None:
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
            
            # 범주형 변수 유사도 (모든 범주형 특성 사용)
            categorical_similarity = 0
            categorical_matches = 0
            total_categorical = 0
            
            categorical_columns = [
                '판정진행상태', '사고진행상태', '수출자', '수입자', 
                '수출보증금액통화', '결제금액통화', '사고금액통화', 
                '사고유형명', '수입국', '보험종목', '결제방법', 
                '결제방법설명', '결제조건', '향후결제전망', '수입자명',
                '판정구분', '판정사유'
            ]
            
            for field in categorical_columns:
                if f'{field}_encoded' in processed_query:
                    total_categorical += 1
                    if processed_query[f'{field}_encoded'] == case.get(f'{field}_encoded', -1):
                        categorical_matches += 1
            
            if total_categorical > 0:
                categorical_similarity = categorical_matches / total_categorical
            
            # 수치형 변수 유사도 (모든 수치형 특성 사용)
            numerical_similarity = 0
            if 'numerical_features' in processed_query:
                query_numerical = processed_query['numerical_features']
                case_numerical = []
                
                for col in ['사고금액', '결제금액', '수출보증금액', '판정금액']:
                    case_numerical.append(case.get(col, 0))
                
                # 유클리드 거리를 유사도로 변환
                distance = np.sqrt(np.sum((query_numerical - case_numerical) ** 2))
                numerical_similarity = 1 / (1 + distance)
            
            # 가중 평균 (KoSimCSE가 있으면 텍스트 가중치 증가)
            if self.text_embeddings is not None:
                final_similarity = (
                    0.6 * text_similarity +  # KoSimCSE 사용시 텍스트 가중치 증가
                    0.3 * categorical_similarity +
                    0.1 * numerical_similarity
                )
            else:
                final_similarity = (
                    0.4 * text_similarity +
                    0.4 * categorical_similarity +
                    0.2 * numerical_similarity
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
        
        print(f"📝 판정 요약: {result['판정요약']}")
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
        print(f"   - 수출보증금액: {case_info['수출보증금액']:,.0f}")
        print(f"   - 판정금액: {case_info['판정금액']:,.0f}")
        
        print(f"\n📋 판정 과정:")
        for i, decision in enumerate(result['decision_process']):
            print(f"   {i+1}차 판정:")
            print(f"     - 날짜: {decision['날짜']}")
            print(f"     - 판정: {decision['판정구분']}")
            print(f"     - 금액: {decision['판정금액']:,.0f}")
            print(f"     - 사유: {decision['판정사유']}")
            print(f"     - 상태: {decision['진행상태']}")
        
        print(f"{'='*60}")

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
    search_engine = ComprehensiveSimilaritySearch()
    
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
    
    # 결과 출력
    for result in results:
        search_engine.print_case_result(result) 