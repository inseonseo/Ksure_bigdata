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
import time
from typing import Dict, List, Tuple, Optional
import hashlib

class OptimizedComprehensiveSimilaritySearch:
    def __init__(self, data_path='data/testy.csv'):
        """
        최적화된 종합 사고 유사도 검색기 초기화
        
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
        
        # 임베딩 캐시 파일 경로
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.embedding_cache_path = os.path.join(current_dir, 'optimized_comprehensive_embeddings_cache.pkl')
        
        # 성능 모니터링
        self.performance_stats = {
            'data_load_time': 0,
            'preprocessing_time': 0,
            'embedding_time': 0,
            'cache_load_time': 0
        }
        
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

    def get_data_hash(self) -> str:
        """데이터 해시 생성 (변경 감지용)"""
        if self.case_summary is None or 'combined_text' not in self.case_summary.columns:
            return ""
        
        text_data = str(self.case_summary['combined_text'].tolist())
        return hashlib.md5(text_data.encode()).hexdigest()

    def load_kosimcse_model(self):
        """KoSimCSE 모델 로드 (최적화)"""
        start_time = time.time()
        print("🤖 KoSimCSE 모델 로드 중...")
        
        try:
            # 모델 경로
            model_path = 'BM-K/KoSimCSE-roberta'
            
            # GPU 사용 가능시 GPU 사용
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"📱 사용 디바이스: {device}")
            
            self.kosimcse_tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.kosimcse_model = AutoModel.from_pretrained(model_path).to(device)
            
            # 모델을 평가 모드로 설정
            self.kosimcse_model.eval()
            
            load_time = time.time() - start_time
            print(f"✅ KoSimCSE 모델 로드 완료 ({load_time:.2f}초)")
                
        except Exception as e:
            print(f"⚠️ KoSimCSE 모델 로드 실패: {str(e)}")
            print("⚠️ KoSimCSE 모델을 찾을 수 없습니다. 텍스트 유사도만 사용합니다.")
            self.kosimcse_model = None
            self.kosimcse_tokenizer = None

    def load_embeddings_cache(self) -> bool:
        """임베딩 캐시 로드 (개선된 버전)"""
        start_time = time.time()
        
        try:
            if not os.path.exists(self.embedding_cache_path):
                print("📂 캐시 파일이 없습니다. 새로 생성합니다.")
                return False
            
            print("📂 임베딩 캐시 로드 중...")
            with open(self.embedding_cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # 캐시 구조 검증
            required_keys = ['embeddings', 'data_hash', 'model_name', 'created_at']
            if not all(key in cache_data for key in required_keys):
                print("⚠️ 캐시 파일 구조가 올바르지 않습니다.")
                return False
            
            # 데이터 변경 확인
            current_hash = self.get_data_hash()
            if cache_data['data_hash'] != current_hash:
                print("⚠️ 데이터가 변경되어 캐시를 무시합니다.")
                return False
            
            # 캐시 로드
            self.text_embeddings = cache_data['embeddings']
            
            cache_time = time.time() - start_time
            self.performance_stats['cache_load_time'] = cache_time
            
            print(f"✅ 임베딩 캐시 로드 완료: {self.text_embeddings.shape} ({cache_time:.2f}초)")
            print(f"📅 캐시 생성일: {cache_data['created_at']}")
            
            return True
            
        except Exception as e:
            print(f"⚠️ 임베딩 캐시 로드 실패: {str(e)}")
            return False

    def save_embeddings_cache(self):
        """임베딩 캐시 저장 (개선된 버전)"""
        if self.text_embeddings is None:
            return
        
        try:
            cache_data = {
                'embeddings': self.text_embeddings,
                'data_hash': self.get_data_hash(),
                'model_name': 'BM-K/KoSimCSE-roberta',
                'created_at': datetime.now().isoformat(),
                'data_size': len(self.case_summary),
                'embedding_shape': self.text_embeddings.shape
            }
            
            with open(self.embedding_cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            print(f"💾 임베딩 캐시 저장 완료: {self.embedding_cache_path}")
            
        except Exception as e:
            print(f"⚠️ 임베딩 캐시 저장 실패: {str(e)}")

    def encode_text_with_kosimcse(self, texts: List[str]) -> Optional[np.ndarray]:
        """KoSimCSE를 사용하여 텍스트 임베딩 생성 (상세 진행률 표시)"""
        if self.kosimcse_model is None or self.kosimcse_tokenizer is None:
            return None
        
        start_time = time.time()
        total_texts = len(texts)
        
        # GPU/CPU 감지 및 예상 시간 계산
        device = next(self.kosimcse_model.parameters()).device
        is_gpu = device.type == 'cuda'
        estimated_time = (total_texts / 1000) * (2 if is_gpu else 6)  # GPU: 2초/1k, CPU: 6초/1k
        
        print(f"🤖 KoSimCSE 임베딩 생성 시작")
        print(f"   📊 총 텍스트: {total_texts:,}개")
        print(f"   💻 사용 디바이스: {device}")
        print(f"   ⏱️  예상 소요시간: {estimated_time/60:.1f}분")
        print(f"   {'='*50}")
        
        try:
            batch_size = 32 if is_gpu else 16  # GPU일 때 더 큰 배치 사용
            embeddings = []
            last_reported_progress = -1
            
            for i in range(0, total_texts, batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # 10%마다 진행률 표시
                progress = (i / total_texts) * 100
                if progress >= last_reported_progress + 10:
                    elapsed_time = time.time() - start_time
                    remaining_time = (elapsed_time / (i + 1)) * (total_texts - i - 1) if i > 0 else estimated_time
                    
                    print(f"   🔄 진행률: {progress:.0f}% ({i:,}/{total_texts:,})")
                    print(f"      ⏱️  경과: {elapsed_time:.0f}초 | 남은시간: {remaining_time:.0f}초")
                    last_reported_progress = progress
                
                # 토크나이징
                inputs = self.kosimcse_tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(device)
                
                # 임베딩 생성
                with torch.no_grad():
                    outputs = self.kosimcse_model(**inputs)
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    embeddings.extend(batch_embeddings)
                
                # 메모리 정리
                del inputs, outputs
                if is_gpu:
                    torch.cuda.empty_cache()
            
            embedding_time = time.time() - start_time
            self.performance_stats['embedding_time'] = embedding_time
            
            print(f"   🎉 100% 완료!")
            print(f"   {'='*50}")
            print(f"✅ KoSimCSE 임베딩 생성 완료")
            print(f"   📊 생성된 임베딩: {len(embeddings):,}개")
            print(f"   ⏱️  실제 소요시간: {embedding_time/60:.1f}분")
            print(f"   💾 임베딩 크기: {np.array(embeddings).nbytes / 1024 / 1024:.1f}MB")
            
            return np.array(embeddings)
            
        except Exception as e:
            print(f"❌ KoSimCSE 임베딩 생성 실패: {str(e)}")
            return None

    def load_and_preprocess_data(self):
        """데이터 로드 및 전처리 (최적화)"""
        start_time = time.time()
        print("📊 데이터 로드 중...")
        
        try:
            # 데이터 로드
            self.data = pd.read_csv(self.data_path, encoding='cp949')
            
            data_load_time = time.time() - start_time
            self.performance_stats['data_load_time'] = data_load_time
            
            print(f"✅ 데이터 로드 완료: {len(self.data)}개 레코드 ({data_load_time:.2f}초)")
            
            # 사례 요약 생성
            preprocess_start = time.time()
            self.create_case_summary()
            
            # 캐시 먼저 시도
            cache_loaded = False
            if self.case_summary is not None and 'combined_text' in self.case_summary.columns:
                cache_loaded = self.load_embeddings_cache()
            
            # 캐시가 없거나 실패한 경우에만 모델 로드
            if not cache_loaded:
                self.load_kosimcse_model()
            
            # 특성 전처리
            self.preprocess_features(skip_embeddings=cache_loaded)
            
            preprocess_time = time.time() - preprocess_start
            self.performance_stats['preprocessing_time'] = preprocess_time
            
            print(f"✅ 전처리 완료 ({preprocess_time:.2f}초)")
            
            # 성능 통계 출력
            self.print_performance_stats()
            
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {str(e)}")
            raise

    def create_case_summary(self):
        """사건별로 그룹화하여 사례 요약 생성"""
        print("📋 사례 요약 생성 중...")
        
        required_columns = ['보상파일번호', '사고번호']
        available_columns = [col for col in required_columns if col in self.data.columns]
        
        if len(available_columns) < 2:
            print("⚠️ 보상파일번호 또는 사고번호 컬럼이 없습니다. 전체 데이터를 사용합니다.")
            self.case_summary = self.data.copy()
            return
        
        # 사건별 그룹화
        case_groups = self.data.groupby(['보상파일번호', '사고번호'])
        
        case_summaries = []
        total_groups = len(case_groups)
        
        for idx, ((보상파일번호, 사고번호), group) in enumerate(case_groups):
            if idx % 1000 == 0:  # 진행률 표시
                progress = (idx / total_groups) * 100
                print(f"   진행률: {progress:.1f}% ({idx}/{total_groups})")
            
            case_summary = {
                '보상파일번호': 보상파일번호,
                '사고번호': 사고번호
            }
            
            # 단일 값 컬럼들
            single_value_columns = [
                '사고접수일자', '사고유형명', '수입국', '수출자', '보험종목',
                '사고금액', '결제금액', '수출보증금액', '사고설명', '수입자명'
            ]
            
            for col in single_value_columns:
                if col in group.columns:
                    case_summary[col] = group[col].iloc[0] if not group[col].isna().all() else None
            
            # 리스트 컬럼들
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

    def preprocess_features(self, skip_embeddings=False):
        """특성 전처리 (최적화)"""
        print("🔧 특성 전처리 중...")
        
        # 텍스트 필드 결합
        text_columns = ['사고설명']
        available_text_columns = [col for col in text_columns if col in self.case_summary.columns]
        
        if available_text_columns:
            self.case_summary['combined_text'] = self.case_summary[available_text_columns].fillna('').agg(' '.join, axis=1)
        else:
            self.case_summary['combined_text'] = ''
        
        # KoSimCSE 임베딩 생성 (캐시되지 않은 경우에만)
        if not skip_embeddings and self.kosimcse_model is not None:
            texts = self.case_summary['combined_text'].fillna('').tolist()
            self.text_embeddings = self.encode_text_with_kosimcse(texts)
            
            if self.text_embeddings is not None:
                # 새로 생성된 임베딩 캐시 저장
                self.save_embeddings_cache()
        
        # 범주형 변수 인코딩
        categorical_columns = [
            '판정진행상태', '사고진행상태', '수출자', '수입자', 
            '수출보증금액통화', '결제금액통화', '사고금액통화', 
            '사고유형명', '수입국', '보험종목', '결제방법', 
            '결제방법설명', '결제조건', '향후결제전망', '수입자명',
            '판정구분', '판정사유'
        ]
        
        for col in categorical_columns:
            if col in self.case_summary.columns:
                if self.case_summary[col].apply(lambda x: isinstance(x, list)).any():
                    values = self.case_summary[col].apply(
                        lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None
                    )
                else:
                    values = self.case_summary[col]
                
                le = LabelEncoder()
                self.case_summary[f'{col}_encoded'] = le.fit_transform(values.astype(str))
                self.label_encoders[col] = le
        
        # 수치형 변수 스케일링
        numerical_columns = []
        for col in ['사고금액', '결제금액', '수출보증금액', '판정금액']:
            if col in self.case_summary.columns:
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

    def print_performance_stats(self):
        """성능 통계 출력"""
        print("\n📊 성능 통계:")
        print(f"   데이터 로드: {self.performance_stats['data_load_time']:.2f}초")
        print(f"   전처리: {self.performance_stats['preprocessing_time']:.2f}초")
        print(f"   임베딩 생성: {self.performance_stats['embedding_time']:.2f}초")
        print(f"   캐시 로드: {self.performance_stats['cache_load_time']:.2f}초")
        
        total_time = sum(self.performance_stats.values())
        print(f"   총 시간: {total_time:.2f}초")

    def search_similar_cases(self, query: Dict, top_k: int = 5, verbose: bool = True) -> List[Dict]:
        """유사한 사고 사례 검색 (개선된 버전)"""
        search_start = time.time()
        
        if verbose:
            print(f"\n🔍 유사 사례 검색 중... (상위 {top_k}개)")
            print(f"📝 검색 입력: {list(query.keys())}")
        
        # 쿼리 전처리
        processed_query = self.preprocess_query(query)
        
        # 유사도 계산
        similarities, similarity_details = self.calculate_similarity_with_details(processed_query)
        
        # 상위 k개 선택
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for i, idx in enumerate(top_indices):
            case = self.case_summary.iloc[idx]
            similarity = similarities[idx]
            details = similarity_details[idx]
            
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
                    '판정금액': case.get('판정금액', 0),
                    '사고설명': case.get('사고설명', 'N/A')
                },
                'decision_process': decision_process,
                'similarity_details': details  # 유사도 근거 추가
            }
            
            results.append(result)
        
        search_time = time.time() - search_start
        
        if verbose:
            print(f"✅ 검색 완료: {len(results)}개 결과 ({search_time:.2f}초)")
        
        return results

    def calculate_similarity_with_details(self, processed_query: Dict) -> Tuple[np.ndarray, List[Dict]]:
        """유사도 계산 (상세 정보 포함)"""
        similarities = []
        similarity_details = []
        
        for idx, case in self.case_summary.iterrows():
            details = {
                'text_similarity': 0,
                'categorical_similarity': 0,
                'numerical_similarity': 0,
                'categorical_matches': {},
                'numerical_differences': {}
            }
            
            # 텍스트 유사도
            text_similarity = 0
            if 'text_embedding' in processed_query and self.text_embeddings is not None:
                query_embedding = processed_query['text_embedding']
                case_embedding = self.text_embeddings[idx]
                
                similarity = cosine_similarity([query_embedding], [case_embedding])[0][0]
                text_similarity = max(0, similarity)
                details['text_similarity'] = text_similarity
            
            # 범주형 변수 유사도
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
                    query_value = processed_query[f'{field}_encoded']
                    case_value = case.get(f'{field}_encoded', -1)
                    
                    if query_value == case_value:
                        categorical_matches += 1
                        details['categorical_matches'][field] = True
                    else:
                        details['categorical_matches'][field] = False
            
            if total_categorical > 0:
                categorical_similarity = categorical_matches / total_categorical
                details['categorical_similarity'] = categorical_similarity
            
            # 수치형 변수 유사도
            numerical_similarity = 0
            if 'numerical_features' in processed_query:
                query_numerical = processed_query['numerical_features']
                case_numerical = []
                
                numerical_columns = ['사고금액', '결제금액', '수출보증금액', '판정금액']
                for i, col in enumerate(numerical_columns):
                    case_value = case.get(col, 0)
                    case_numerical.append(case_value)
                    
                    # 개별 차이 저장
                    if i < len(query_numerical):
                        details['numerical_differences'][col] = abs(query_numerical[i] - case_value)
                
                distance = np.sqrt(np.sum((query_numerical - case_numerical) ** 2))
                numerical_similarity = 1 / (1 + distance)
                details['numerical_similarity'] = numerical_similarity
            
            # 가중 평균
            if self.text_embeddings is not None:
                final_similarity = (
                    0.6 * text_similarity +
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
            similarity_details.append(details)
        
        return np.array(similarities), similarity_details

    def preprocess_query(self, query: Dict) -> Dict:
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

    def create_decision_process(self, case):
        """판정 과정 상세 정보 생성"""
        process = []
        
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
            process.append({
                '회차': 1,
                '날짜': 'N/A',
                '판정구분': '정보없음',
                '판정금액': 0,
                '판정사유': '정보없음',
                '진행상태': '정보없음'
            })
        
        return process

    def print_detailed_result(self, result: Dict):
        """상세 결과 출력 (유사도 근거 포함)"""
        print(f"\n{'='*80}")
        print(f"🏆 순위: {result['rank']} (전체 유사도: {result['similarity']:.3f})")
        print(f"📋 사건 ID: {result['case_id']}")
        
        # 유사도 상세 분석
        details = result['similarity_details']
        print(f"\n📊 유사도 분석:")
        print(f"   📝 텍스트 유사도: {details['text_similarity']:.3f}")
        print(f"   🏷️  범주형 유사도: {details['categorical_similarity']:.3f}")
        print(f"   🔢 수치형 유사도: {details['numerical_similarity']:.3f}")
        
        # 범주형 매칭 상세
        if details['categorical_matches']:
            print(f"\n🏷️  범주형 필드 매칭:")
            for field, match in details['categorical_matches'].items():
                status = "✅" if match else "❌"
                print(f"   {status} {field}")
        
        # 사건 정보
        print(f"\n📋 사건 정보:")
        case_info = result['case_info']
        for key, value in case_info.items():
            if key == '사고설명':
                # 사고설명은 길 수 있으므로 일부만 표시
                display_value = str(value)[:100] + "..." if len(str(value)) > 100 else value
                print(f"   - {key}: {display_value}")
            elif isinstance(value, (int, float)) and key.endswith('금액'):
                print(f"   - {key}: {value:,.0f}")
            else:
                print(f"   - {key}: {value}")
        
        # 판정 과정
        print(f"\n📋 판정 과정:")
        for i, decision in enumerate(result['decision_process']):
            print(f"   {i+1}차 판정:")
            print(f"     - 날짜: {decision['날짜']}")
            print(f"     - 판정: {decision['판정구분']}")
            print(f"     - 금액: {decision['판정금액']:,.0f}")
            print(f"     - 사유: {decision['판정사유']}")
            print(f"     - 상태: {decision['진행상태']}")
        
        print(f"{'='*80}")

    def get_available_options(self) -> Dict:
        """드롭다운 옵션들 반환"""
        options = {}
        
        for field in self.input_fields:
            if self.input_fields[field] == 'dropdown':
                if field in self.case_summary.columns:
                    options[field] = sorted(self.case_summary[field].dropna().unique().tolist())
        
        return options

# 사용 예시
if __name__ == "__main__":
    print("🚀 최적화된 종합 유사도 검색기 시작")
    
    # 검색기 초기화 (첫 실행시 시간이 걸리지만, 두 번째부터는 캐시 사용)
    search_engine = OptimizedComprehensiveSimilaritySearch()
    
    # 사용 가능한 옵션 확인
    options = search_engine.get_available_options()
    print("\n📋 사용 가능한 옵션:")
    for field, values in options.items():
        print(f"   {field}: {len(values)}개 옵션")
    
    # 검색 예시
    query = {
        '사고유형명': '지급거절',
        '수입국': '미국',
        '보험종목': '단기수출보험',
        '사고설명': '수입자가 지급을 거절함'
    }
    
    print(f"\n🔍 검색 쿼리: {query}")
    results = search_engine.search_similar_cases(query, top_k=3)
    
    # 결과 출력
    for result in results:
        search_engine.print_detailed_result(result)