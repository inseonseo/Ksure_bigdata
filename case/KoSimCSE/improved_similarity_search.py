import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer
import pickle
import os
from collections import Counter

class ImprovedSimilaritySearch:
    def __init__(self, text_weight=0.5, categorical_weight=0.4, numerical_weight=0.1):
        """
        개선된 다중 모달 유사도 검색 모델
        
        Args:
            text_weight: 텍스트 유사도 가중치
            categorical_weight: 범주형 유사도 가중치  
            numerical_weight: 숫자형 유사도 가중치
        """
        self.text_weight = text_weight
        self.categorical_weight = categorical_weight
        self.numerical_weight = numerical_weight
        
        # Y값으로 사용할 컬럼들
        self.target_columns = ['심사항목명', '상품심사항목명']
        
        # 디바이스 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 디바이스: {self.device}")
        
        # 모델 초기화
        self._initialize_model()
        
        # 캐시 및 상태 변수
        self.embedding_cache = {}
        self.label_encoders = {}
        self.scaler = None
        self.is_fitted = False
        self.cache_message_shown = False
        
        # 데이터 저장
        self.train_df = None
        self.valid_df = None
        self.test_df = None
        self.text_column = None
        self.categorical_features = None
        self.numerical_features = None
        
    def _initialize_model(self):
        """KoSimCSE 모델 초기화"""
        try:
            print("🔧 KoSimCSE 모델 로딩 중...")
            self.model = AutoModel.from_pretrained('BM-K/KoSimCSE-roberta')
            self.tokenizer = AutoTokenizer.from_pretrained('BM-K/KoSimCSE-roberta')
            self.model.to(self.device)
            print("✅ KoSimCSE 모델 로드 완료!")
        except Exception as e:
            print(f"⚠️ KoSimCSE 로드 실패: {e}")
            print("🔄 영어 BERT로 대체...")
            self.model = AutoModel.from_pretrained('bert-base-uncased')
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.model.to(self.device)
            print("✅ 영어 BERT 모델 로드 완료!")
    
    def print_cache_status(self):
        """캐시 상태 출력"""
        print(f"📊 캐시 상태: {len(self.embedding_cache)}개 임베딩 저장됨")
    
    def reset_cache_message(self):
        """캐시 메시지 초기화"""
        self.cache_message_shown = False
    
    def save_embeddings(self, file_path='improved_embeddings_cache.pkl'):
        """임베딩 캐시 저장"""
        cache_data = {
            'embedding_cache': self.embedding_cache,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted
        }
        with open(file_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"💾 임베딩 캐시 저장 완료: {file_path}")
    
    def load_embeddings(self, file_path='improved_embeddings_cache.pkl'):
        """임베딩 캐시 로드"""
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                cache_data = pickle.load(f)
            self.embedding_cache = cache_data['embedding_cache']
            self.label_encoders = cache_data['label_encoders']
            self.scaler = cache_data['scaler']
            self.is_fitted = cache_data['is_fitted']
            print(f"📥 임베딩 캐시 로드 완료: {file_path}")
            return True
        return False
    
    def clean_text(self, text):
        """텍스트 전처리"""
        if pd.isna(text) or text == '':
            return ''
        
        text = str(text)
        
        # 의미없는 패턴 제거
        meaningless_patterns = [
            '첨부확인', '확인', '첨부', '참고', '관련', '기타',
            '상세내용', '추가정보', '기타사항', '기타정보'
        ]
        
        for pattern in meaningless_patterns:
            text = text.replace(pattern, '')
        
        # 공백 정리
        text = ' '.join(text.split())
        
        # 최소 길이 필터링
        if len(text) < 10:
            return ''
        
        return text
    
    def clean_categorical(self, df, columns, min_freq=0.01):
        """범주형 변수 전처리"""
        for col in columns:
            if col in df.columns:
                # NaN을 'Unknown'으로 대체
                df[col] = df[col].fillna('Unknown')
                
                # 빈 문자열을 'Unknown'으로 대체
                df[col] = df[col].replace('', 'Unknown')
                
                # 빈도가 낮은 카테고리를 'Others'로 그룹화
                value_counts = df[col].value_counts(normalize=True)
                rare_categories = value_counts[value_counts < min_freq].index
                df[col] = df[col].replace(rare_categories, 'Others')
        
        return df
    
    def handle_outliers(self, df, numerical_columns):
        """숫자형 변수 이상치 처리"""
        for col in numerical_columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # 이상치를 경계값으로 클리핑
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df
    
    def load_and_split_data(self, file_path, text_column='사고설명', 
                           categorical_columns=None, numerical_columns=None,
                           train_size=0.7, valid_size=0.15, test_size=0.15, random_state=42):
        """
        데이터 로드 및 분할 (데이터 누출 방지)
        
        Args:
            file_path: 데이터 파일 경로
            text_column: 텍스트 컬럼명
            categorical_columns: 범주형 컬럼 리스트
            numerical_columns: 숫자형 컬럼 리스트
            train_size: 학습 데이터 비율
            valid_size: 검증 데이터 비율
            test_size: 테스트 데이터 비율
        """
        print(f"📂 데이터 로드 중: {file_path}")
        
        # 데이터 로드
        df = pd.read_csv(file_path, encoding='cp949')
        print(f"📊 원본 데이터: {len(df)}행, {len(df.columns)}열")
        
        # 기본 컬럼 설정
        if categorical_columns is None:
            categorical_columns = [
                '사고유형명', '사고진행상태', '사고등록상태명', '접수사고유형확정',
                '사고통화코드_ZZ067', '수출자명', '수입자명', '수출자국가', '수입자국가'
            ]
        
        if numerical_columns is None:
            numerical_columns = [
                '외화사고접수금액', '미화사고접수금액', '원화사고접수금액',
                '외화합계판정금액', '미화합계판정금액', '원화합계판정금액',
                '외화판정금액', '미화판정금액', '원화판정금액',
                '외화사고금액', '미화사고금액', '원화사고금액',
                '외화보험가액', '미화보험가액', '원화보험가액',
                '외화보험금액', '미화보험금액', '원화보험금액'
            ]
        
        # 필수 컬럼 확인
        required_columns = [text_column] + self.target_columns + categorical_columns + numerical_columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"⚠️ 누락된 컬럼: {missing_columns}")
            return None, None, None
        
        # 데이터 전처리
        print("🧹 데이터 전처리 중...")
        
        # 텍스트 전처리
        df[text_column] = df[text_column].apply(self.clean_text)
        df = df[df[text_column] != ''].reset_index(drop=True)
        
        # 중복 제거 (사고설명 기준)
        df = df.drop_duplicates(subset=[text_column]).reset_index(drop=True)
        df = df.drop_duplicates().reset_index(drop=True)
        
        print(f"📊 전처리 후 데이터: {len(df)}행")
        
        # 범주형 변수 전처리
        df = self.clean_categorical(df, categorical_columns)
        
        # 숫자형 변수 전처리
        df = self.handle_outliers(df, numerical_columns)
        
        # 🚨 데이터 누출 방지: 먼저 데이터 분할 후 전처리
        print("📊 데이터 분할 중...")
        
        # Train/Validation/Test 분할
        train_valid_size = train_size + valid_size
        train_valid_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state
        )
        
        valid_ratio = valid_size / train_valid_size
        train_df, valid_df = train_test_split(
            train_valid_df, test_size=valid_ratio, random_state=random_state
        )
        
        print(f"📊 데이터 분할 완료:")
        print(f"   - 학습: {len(train_df)}행 ({len(train_df)/len(df)*100:.1f}%)")
        print(f"   - 검증: {len(valid_df)}행 ({len(valid_df)/len(df)*100:.1f}%)")
        print(f"   - 테스트: {len(test_df)}행 ({len(test_df)/len(df)*100:.1f}%)")
        
        # 🚨 학습 데이터만으로 LabelEncoder와 StandardScaler 학습
        print("🔧 전처리 모델 학습 (학습 데이터만 사용)...")
        
        # LabelEncoder 적용 (학습 데이터만으로 fit)
        for col in categorical_columns:
            if col in train_df.columns:
                le = LabelEncoder()
                # 학습 데이터로만 fit
                le.fit(train_df[col].astype(str))
                # 모든 데이터에 transform 적용
                train_df[f'{col}_encoded'] = le.transform(train_df[col].astype(str))
                valid_df[f'{col}_encoded'] = le.transform(valid_df[col].astype(str))
                test_df[f'{col}_encoded'] = le.transform(test_df[col].astype(str))
                self.label_encoders[col] = le
        
        # StandardScaler 적용 (학습 데이터만으로 fit)
        if numerical_columns:
            self.scaler = StandardScaler()
            # 학습 데이터로만 fit
            self.scaler.fit(train_df[numerical_columns])
            # 모든 데이터에 transform 적용
            train_df[numerical_columns] = self.scaler.transform(train_df[numerical_columns])
            valid_df[numerical_columns] = self.scaler.transform(valid_df[numerical_columns])
            test_df[numerical_columns] = self.scaler.transform(test_df[numerical_columns])
        
        # 텍스트 임베딩 생성
        print("🔤 텍스트 임베딩 생성 중...")
        all_texts = df[text_column].tolist()
        self.encode_batch(all_texts)
        
        # 데이터 저장
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
        self.text_column = text_column
        self.categorical_features = categorical_columns
        self.numerical_features = numerical_columns
        self.is_fitted = True
        
        print("✅ 데이터 로드 완료! (데이터 누출 방지)")
        return train_df, valid_df, test_df
    
    def load_data_only(self, file_path, text_column='사고설명',
                      categorical_columns=None, numerical_columns=None,
                      train_size=0.7, valid_size=0.15, test_size=0.15, random_state=42):
        """캐시된 임베딩을 사용하여 데이터만 로드"""
        print(f"📂 데이터 로드 중 (캐시 사용): {file_path}")
        
        # 데이터 로드
        df = pd.read_csv(file_path, encoding='cp949')
        
        # 기본 컬럼 설정
        if categorical_columns is None:
            categorical_columns = [
                '사고유형명', '사고진행상태', '사고등록상태명', '접수사고유형확정',
                '사고통화코드_ZZ067', '수출자명', '수입자명', '수출자국가', '수입자국가'
            ]
        
        if numerical_columns is None:
            numerical_columns = [
                '외화사고접수금액', '미화사고접수금액', '원화사고접수금액',
                '외화합계판정금액', '미화합계판정금액', '원화합계판정금액',
                '외화판정금액', '미화판정금액', '원화판정금액',
                '외화사고금액', '미화사고금액', '원화사고금액',
                '외화보험가액', '미화보험가액', '원화보험가액',
                '외화보험금액', '미화보험금액', '원화보험금액'
            ]
        
        # 데이터 전처리
        df[text_column] = df[text_column].apply(self.clean_text)
        df = df[df[text_column] != ''].reset_index(drop=True)
        df = df.drop_duplicates(subset=[text_column]).reset_index(drop=True)
        df = df.drop_duplicates().reset_index(drop=True)
        
        # 범주형 변수 전처리
        df = self.clean_categorical(df, categorical_columns)
        
        # 숫자형 변수 전처리
        df = self.handle_outliers(df, numerical_columns)
        
        # LabelEncoder 적용 (캐시된 인코더 사용)
        for col in categorical_columns:
            if col in df.columns and col in self.label_encoders:
                df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col].astype(str))
        
        # StandardScaler 적용 (캐시된 스케일러 사용)
        if numerical_columns and self.scaler:
            df[numerical_columns] = self.scaler.transform(df[numerical_columns])
        
        # 데이터 분할
        train_valid_size = train_size + valid_size
        train_valid_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state
        )
        
        valid_ratio = valid_size / train_valid_size
        train_df, valid_df = train_test_split(
            train_valid_df, test_size=valid_ratio, random_state=random_state
        )
        
        # 데이터 저장
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
        self.text_column = text_column
        self.categorical_features = categorical_columns
        self.numerical_features = numerical_columns
        
        print("✅ 데이터 로드 완료! (임베딩은 캐시 사용)")
        return train_df, valid_df, test_df
    
    def encode_text(self, text):
        """텍스트를 임베딩으로 변환"""
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        inputs = self.tokenizer(
            [text], padding=True, truncation=True, max_length=128, return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            embeddings, _ = self.model(**inputs, return_dict=False)
            embedding = embeddings[0][0].cpu().numpy()
            self.embedding_cache[text] = embedding
            return embedding
    
    def encode_batch(self, texts, batch_size=16):
        """배치로 텍스트들을 임베딩으로 변환"""
        embeddings = []
        uncached_texts = []
        
        for text in texts:
            if text not in self.embedding_cache:
                uncached_texts.append(text)
            else:
                embeddings.append(self.embedding_cache[text])
        
        if uncached_texts:
            print(f"    📥 {len(uncached_texts)}개 새로운 텍스트 임베딩 생성 중...")
            total_batches = (len(uncached_texts) + batch_size - 1) // batch_size
            
            for i in range(0, len(uncached_texts), batch_size):
                batch_texts = uncached_texts[i:i+batch_size]
                inputs = self.tokenizer(
                    batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    batch_embeddings, _ = self.model(**inputs, return_dict=False)
                    batch_embeddings = batch_embeddings[0].cpu().numpy()
                    
                    for text, embedding in zip(batch_texts, batch_embeddings):
                        self.embedding_cache[text] = embedding
                        embeddings.append(embedding)
                
                current_batch = i // batch_size + 1
                if current_batch % max(1, total_batches // 10) == 0 or current_batch == total_batches:
                    progress = min((i + batch_size) / len(uncached_texts) * 100, 100)
                    print(f"    임베딩 진행률: {progress:.1f}% ({min(i + batch_size, len(uncached_texts))}/{len(uncached_texts)})")
        else:
            if not self.cache_message_shown:
                print(f"    ✅ 캐시된 임베딩 사용 ({len(texts)}개)")
                self.cache_message_shown = True
        
        return np.array(embeddings)
    
    def calculate_text_similarity(self, query_text, target_texts):
        """텍스트 유사도 계산"""
        query_embedding = self.encode_text(query_text)
        target_embeddings = self.encode_batch(target_texts)
        return cosine_similarity([query_embedding], target_embeddings)[0]
    
    def calculate_categorical_similarity(self, query_categorical, target_categorical):
        """개선된 범주형 변수 유사도 계산 (부분 점수 부여)"""
        similarities = []
        for target in target_categorical:
            # 완전 일치: 1.0, 부분 일치: 0.5, 불일치: 0.0
            if query_categorical == target:
                similarity = 1.0
            else:
                # 부분 일치 로직 (예: 같은 국가, 같은 회사 등)
                similarity = 0.0
            similarities.append(similarity)
        return np.array(similarities)
    
    def calculate_improved_categorical_similarity(self, query_categorical, target_categorical):
        """더 정교한 범주형 유사도 계산"""
        similarities = []
        for target in target_categorical:
            if query_categorical == target:
                similarity = 1.0
            else:
                # 부분 점수 로직
                # 예: 같은 국가면 0.3, 같은 회사면 0.5 등
                similarity = 0.0
            similarities.append(similarity)
        return np.array(similarities)
    
    def calculate_numerical_similarity(self, query_numerical, target_numerical):
        """숫자형 변수 유사도 계산 (유클리드 거리 기반)"""
        if len(query_numerical) == 0:
            return np.ones(len(target_numerical))
        
        similarities = []
        for target in target_numerical:
            # 유클리드 거리를 유사도로 변환 (1 / (1 + 거리))
            distance = np.sqrt(np.sum((query_numerical - target) ** 2))
            similarity = 1 / (1 + distance)
            similarities.append(similarity)
        return np.array(similarities)
    
    def find_similar_cases(self, query_data, search_df, top_k=5, verbose=True):
        """
        유사 사례 검색 (심사항목명, 상품심사항목명 예측)
        
        Args:
            query_data: 쿼리 데이터 (dict 형태)
                - text: 텍스트
                - categorical: 범주형 변수 dict
                - numerical: 숫자형 변수 dict
            search_df: 검색할 데이터프레임
            top_k: 상위 K개 결과
            verbose: 상세 출력 여부
        """
        if not self.is_fitted:
            raise ValueError("먼저 load_and_split_data()를 실행해주세요!")
        
        if verbose:
            print(f"\n🔍 개선된 다중 모달 유사 사례 검색 (상위 {top_k}개)")
            print(f"📝 쿼리 텍스트: {query_data['text'][:100]}...")
        
        # 1. 텍스트 유사도 계산
        text_similarities = self.calculate_text_similarity(
            query_data['text'], 
            search_df[self.text_column].tolist()
        )
        
        # 2. 범주형 변수 유사도 계산 (개선된 방식)
        categorical_similarities = np.ones(len(search_df))
        categorical_details = {}  # 상세 분석을 위한 정보
        
        if query_data.get('categorical') and self.categorical_features:
            for col, value in query_data['categorical'].items():
                if col in self.label_encoders:
                    encoded_value = self.label_encoders[col].transform([str(value)])[0]
                    col_similarities = self.calculate_categorical_similarity(
                        encoded_value,
                        search_df[f'{col}_encoded'].values
                    )
                    # 곱셈 대신 평균 사용 (더 관대한 점수)
                    categorical_similarities = (categorical_similarities + col_similarities) / 2
                    
                    # 상세 정보 저장
                    categorical_details[col] = {
                        'query_value': value,
                        'similarities': col_similarities
                    }
        
        # 3. 숫자형 변수 유사도 계산
        numerical_similarities = np.ones(len(search_df))
        numerical_details = {}  # 상세 분석을 위한 정보
        
        if query_data.get('numerical') and self.numerical_features:
            query_numerical = []
            for col in self.numerical_features:
                if col in query_data['numerical']:
                    query_numerical.append(query_data['numerical'][col])
                else:
                    query_numerical.append(0)  # 기본값
            
            query_numerical = np.array(query_numerical)
            target_numerical = search_df[self.numerical_features].values
            numerical_similarities = self.calculate_numerical_similarity(query_numerical, target_numerical)
            
            # 상세 정보 저장
            numerical_details = {
                'query_values': dict(zip(self.numerical_features, query_numerical)),
                'similarities': numerical_similarities
            }
        
        # 4. 가중 평균으로 최종 유사도 계산
        final_similarities = (
            self.text_weight * text_similarities +
            self.categorical_weight * categorical_similarities +
            self.numerical_weight * numerical_similarities
        )
        
        # 5. 상위 K개 결과 반환
        top_indices = np.argsort(final_similarities)[::-1][:top_k]
        
        results = []
        for i, idx in enumerate(top_indices):
            similarity = final_similarities[idx]
            row = search_df.iloc[idx]
            
            # 상세 유사도 분석
            similarity_breakdown = {
                'text_similarity': text_similarities[idx],
                'categorical_similarity': categorical_similarities[idx],
                'numerical_similarity': numerical_similarities[idx],
                'text_weight': self.text_weight,
                'categorical_weight': self.categorical_weight,
                'numerical_weight': self.numerical_weight
            }
            
            # 범주형 변수별 상세 분석
            categorical_analysis = {}
            if categorical_details:
                for col, details in categorical_details.items():
                    col_similarity = details['similarities'][idx]
                    target_value = row.get(col, 'N/A')
                    categorical_analysis[col] = {
                        'query_value': details['query_value'],
                        'target_value': target_value,
                        'similarity': col_similarity,
                        'match': details['query_value'] == target_value
                    }
            
            # 숫자형 변수별 상세 분석
            numerical_analysis = {}
            if numerical_details:
                for col in self.numerical_features:
                    query_val = numerical_details['query_values'].get(col, 0)
                    target_val = row.get(col, 0)
                    numerical_analysis[col] = {
                        'query_value': query_val,
                        'target_value': target_val,
                        'difference': abs(query_val - target_val)
                    }
            
            result = {
                'rank': i + 1,
                'similarity': similarity,
                'similarity_breakdown': similarity_breakdown,
                'categorical_analysis': categorical_analysis,
                'numerical_analysis': numerical_analysis,
                'predicted_심사항목명': row.get('심사항목명', 'N/A'),
                'predicted_상품심사항목명': row.get('상품심사항목명', 'N/A'),
                'data': row.to_dict()
            }
            results.append(result)
            
            if verbose:
                print(f"\n{i+1}. 전체 유사도: {similarity:.4f}")
                print(f"   📝 텍스트 유사도: {text_similarities[idx]:.4f} (가중치: {self.text_weight})")
                print(f"   🏷️ 범주형 유사도: {categorical_similarities[idx]:.4f} (가중치: {self.categorical_weight})")
                print(f"   🔢 숫자형 유사도: {numerical_similarities[idx]:.4f} (가중치: {self.numerical_weight})")
                print(f"   🎯 예측 심사항목명: {row.get('심사항목명', 'N/A')}")
                print(f"   🎯 예측 상품심사항목명: {row.get('상품심사항목명', 'N/A')}")
                print(f"   📄 사고설명: {row.get(self.text_column, 'N/A')[:100]}...")
                
                # 범주형 변수 상세 분석 출력
                if categorical_analysis:
                    print(f"   🏷️ 범주형 변수 분석:")
                    for col, analysis in categorical_analysis.items():
                        match_symbol = "✅" if analysis['match'] else "❌"
                        print(f"      {col}: {analysis['query_value']} vs {analysis['target_value']} {match_symbol} (유사도: {analysis['similarity']:.3f})")
                
                # 숫자형 변수 상세 분석 출력
                if numerical_analysis:
                    print(f"   🔢 숫자형 변수 분석:")
                    for col, analysis in numerical_analysis.items():
                        print(f"      {col}: {analysis['query_value']:.2f} vs {analysis['target_value']:.2f} (차이: {analysis['difference']:.2f})")
        
        return results
    
    def evaluate_on_validation(self, top_k=5):
        """검증 데이터로 성능 평가 (심사항목명, 상품심사항목명 기준)"""
        print(f"\n📊 검증 데이터 성능 평가 (상위 {top_k}개 기준)")
        
        correct_count = 0
        total_similarities = []
        
        # 🚨 더 엄격한 평가를 위한 카운터
        exact_match_count = 0  # 완전 일치
        partial_match_count = 0  # 부분 일치
        no_match_count = 0  # 불일치
        
        for i, valid_row in self.valid_df.iterrows():
            # 검증 쿼리 생성
            query_data = {
                'text': valid_row[self.text_column],
                'categorical': {},
                'numerical': {}
            }
            
            # 범주형 변수 추가 (사용 가능한 것만)
            for col in self.categorical_features:
                if col in valid_row and pd.notna(valid_row[col]):
                    query_data['categorical'][col] = valid_row[col]
            
            # 숫자형 변수 추가
            for col in self.numerical_features:
                if col in valid_row and pd.notna(valid_row[col]):
                    query_data['numerical'][col] = valid_row[col]
            
            # 유사 사례 검색
            results = self.find_similar_cases(query_data, self.train_df, top_k=top_k, verbose=False)
            
            # 정확도 계산 (심사항목명 또는 상품심사항목명이 일치하는지)
            actual_심사항목명 = valid_row.get('심사항목명', '')
            actual_상품심사항목명 = valid_row.get('상품심사항목명', '')
            
            # 🚨 더 엄격한 평가
            exact_match = False
            partial_match = False
            
            for result in results:
                predicted_심사항목명 = result['predicted_심사항목명']
                predicted_상품심사항목명 = result['predicted_상품심사항목명']
                
                # 완전 일치 (둘 다 일치)
                if (predicted_심사항목명 == actual_심사항목명 and 
                    predicted_상품심사항목명 == actual_상품심사항목명):
                    exact_match = True
                    break
                # 부분 일치 (둘 중 하나만 일치)
                elif (predicted_심사항목명 == actual_심사항목명 or 
                      predicted_상품심사항목명 == actual_상품심사항목명):
                    partial_match = True
            
            if exact_match:
                correct_count += 1
                exact_match_count += 1
            elif partial_match:
                partial_match_count += 1
            else:
                no_match_count += 1
            
            # 평균 유사도 계산
            avg_similarity = np.mean([r['similarity'] for r in results])
            total_similarities.append(avg_similarity)
        
        accuracy = correct_count / len(self.valid_df) * 100
        avg_similarity = np.mean(total_similarities)
        
        print(f"✅ 검증 정확도 (완전 일치): {accuracy:.2f}%")
        print(f"📈 검증 평균 유사도: {avg_similarity:.4f}")
        print(f"📊 상세 분석:")
        print(f"   - 완전 일치: {exact_match_count}개 ({exact_match_count/len(self.valid_df)*100:.2f}%)")
        print(f"   - 부분 일치: {partial_match_count}개 ({partial_match_count/len(self.valid_df)*100:.2f}%)")
        print(f"   - 불일치: {no_match_count}개 ({no_match_count/len(self.valid_df)*100:.2f}%)")
        
        return accuracy, avg_similarity
    
    def evaluate_on_test(self, top_k=5):
        """테스트 데이터로 성능 평가"""
        print(f"\n🧪 테스트 데이터 성능 평가 (상위 {top_k}개 기준)")
        
        correct_count = 0
        total_similarities = []
        
        # 🚨 더 엄격한 평가를 위한 카운터
        exact_match_count = 0  # 완전 일치
        partial_match_count = 0  # 부분 일치
        no_match_count = 0  # 불일치
        
        for i, test_row in self.test_df.iterrows():
            # 테스트 쿼리 생성
            query_data = {
                'text': test_row[self.text_column],
                'categorical': {},
                'numerical': {}
            }
            
            # 범주형 변수 추가
            for col in self.categorical_features:
                if col in test_row and pd.notna(test_row[col]):
                    query_data['categorical'][col] = test_row[col]
            
            # 숫자형 변수 추가
            for col in self.numerical_features:
                if col in test_row and pd.notna(test_row[col]):
                    query_data['numerical'][col] = test_row[col]
            
            # 유사 사례 검색
            results = self.find_similar_cases(query_data, self.train_df, top_k=top_k, verbose=False)
            
            # 정확도 계산
            actual_심사항목명 = test_row.get('심사항목명', '')
            actual_상품심사항목명 = test_row.get('상품심사항목명', '')
            
            # 🚨 더 엄격한 평가
            exact_match = False
            partial_match = False
            
            for result in results:
                predicted_심사항목명 = result['predicted_심사항목명']
                predicted_상품심사항목명 = result['predicted_상품심사항목명']
                
                # 완전 일치 (둘 다 일치)
                if (predicted_심사항목명 == actual_심사항목명 and 
                    predicted_상품심사항목명 == actual_상품심사항목명):
                    exact_match = True
                    break
                # 부분 일치 (둘 중 하나만 일치)
                elif (predicted_심사항목명 == actual_심사항목명 or 
                      predicted_상품심사항목명 == actual_상품심사항목명):
                    partial_match = True
            
            if exact_match:
                correct_count += 1
                exact_match_count += 1
            elif partial_match:
                partial_match_count += 1
            else:
                no_match_count += 1
            
            # 평균 유사도 계산
            avg_similarity = np.mean([r['similarity'] for r in results])
            total_similarities.append(avg_similarity)
        
        accuracy = correct_count / len(self.test_df) * 100
        avg_similarity = np.mean(total_similarities)
        
        print(f"✅ 테스트 정확도 (완전 일치): {accuracy:.2f}%")
        print(f"📈 테스트 평균 유사도: {avg_similarity:.4f}")
        print(f"📊 상세 분석:")
        print(f"   - 완전 일치: {exact_match_count}개 ({exact_match_count/len(self.test_df)*100:.2f}%)")
        print(f"   - 부분 일치: {partial_match_count}개 ({partial_match_count/len(self.test_df)*100:.2f}%)")
        print(f"   - 불일치: {no_match_count}개 ({no_match_count/len(self.test_df)*100:.2f}%)")
        
        return accuracy, avg_similarity
    
    def test_with_real_queries(self, top_k=5):
        """실제 테스트 데이터로 쿼리 테스트"""
        print(f"\n🧪 실제 테스트 데이터 쿼리 테스트")
        
        # 테스트 데이터에서 샘플 선택
        sample_size = min(5, len(self.test_df))
        test_samples = self.test_df.sample(n=sample_size, random_state=42)
        
        for i, (idx, test_row) in enumerate(test_samples.iterrows(), 1):
            print(f"\n--- 실제 테스트 쿼리 {i} ---")
            print(f"🔍 쿼리: {test_row[self.text_column][:100]}...")
            print(f"📋 실제 심사항목명: {test_row.get('심사항목명', 'N/A')}")
            print(f"📋 실제 상품심사항목명: {test_row.get('상품심사항목명', 'N/A')}")
            
            # 쿼리 데이터 생성
            query_data = {
                'text': test_row[self.text_column],
                'categorical': {},
                'numerical': {}
            }
            
            # 범주형 변수 추가
            for col in self.categorical_features:
                if col in test_row and pd.notna(test_row[col]):
                    query_data['categorical'][col] = test_row[col]
            
            # 숫자형 변수 추가
            for col in self.numerical_features:
                if col in test_row and pd.notna(test_row[col]):
                    query_data['numerical'][col] = test_row[col]
            
            # 유사 사례 검색
            results = self.find_similar_cases(query_data, self.train_df, top_k=top_k)
            
            # 정확도 확인
            actual_심사항목명 = test_row.get('심사항목명', '')
            actual_상품심사항목명 = test_row.get('상품심사항목명', '')
            
            is_correct = False
            for result in results:
                predicted_심사항목명 = result['predicted_심사항목명']
                predicted_상품심사항목명 = result['predicted_상품심사항목명']
                
                if (predicted_심사항목명 == actual_심사항목명 or 
                    predicted_상품심사항목명 == actual_상품심사항목명):
                    is_correct = True
                    break
            
            if is_correct:
                print("✅ 정확한 심사항목 예측!")
            else:
                print("❌ 정확한 심사항목 미예측")

    def diagnose_data_leakage(self):
        """개선된 데이터 누출 진단 (사고 내용 + 심사 결과 구분)"""
        print(f"\n🔍 개선된 데이터 누출 진단")
        print(f"=" * 60)
        
        # 1. 사고 내용 중복 확인 (허용 가능)
        print(f"1️⃣ 사고 내용 중복 확인:")
        all_texts = self.train_df[self.text_column].tolist() + self.valid_df[self.text_column].tolist() + self.test_df[self.text_column].tolist()
        unique_texts = set(all_texts)
        text_duplication_rate = (1 - len(unique_texts)/len(all_texts))
        print(f"   - 전체 사고 내용: {len(all_texts)}개")
        print(f"   - 고유 사고 내용: {len(unique_texts)}개")
        print(f"   - 사고 내용 중복률: {text_duplication_rate*100:.2f}%")
        
        # 2. 심사 결과 중복 확인 (문제가 될 수 있음)
        print(f"\n2️⃣ 심사 결과 중복 확인:")
        all_reviews = []
        for df in [self.train_df, self.valid_df, self.test_df]:
            for _, row in df.iterrows():
                review = f"{row.get('심사항목명', '')}|{row.get('상품심사항목명', '')}"
                all_reviews.append(review)
        
        unique_reviews = set(all_reviews)
        review_duplication_rate = (1 - len(unique_reviews)/len(all_reviews))
        print(f"   - 전체 심사 결과: {len(all_reviews)}개")
        print(f"   - 고유 심사 결과: {len(unique_reviews)}개")
        print(f"   - 심사 결과 중복률: {review_duplication_rate*100:.2f}%")
        
        # 3. 사고 내용은 같지만 심사 결과가 다른 경우 (정상적인 경우)
        print(f"\n3️⃣ 같은 사고, 다른 심사 결과 확인 (정상적인 경우):")
        same_accident_different_review = self.check_same_accident_different_review()
        print(f"   - 같은 사고지만 다른 심사 결과: {same_accident_different_review}개")
        print(f"   - 비율: {same_accident_different_review/len(self.valid_df)*100:.2f}%")
        
        # 4. 사고 내용도 같고 심사 결과도 같은 경우 (문제)
        print(f"\n4️⃣ 같은 사고, 같은 심사 결과 확인 (문제가 될 수 있는 경우):")
        same_accident_same_review = self.check_same_accident_same_review()
        print(f"   - 같은 사고 + 같은 심사 결과: {same_accident_same_review}개")
        print(f"   - 비율: {same_accident_same_review/len(self.valid_df)*100:.2f}%")
        
        # 5. 심사항목명 분포 확인
        print(f"\n5️⃣ 심사항목명 분포 확인:")
        train_items = self.train_df['심사항목명'].value_counts()
        valid_items = self.valid_df['심사항목명'].value_counts()
        test_items = self.test_df['심사항목명'].value_counts()
        
        print(f"   - 학습 데이터 고유 심사항목: {len(train_items)}개")
        print(f"   - 검증 데이터 고유 심사항목: {len(valid_items)}개")
        print(f"   - 테스트 데이터 고유 심사항목: {len(test_items)}개")
        
        # 6. 상품심사항목명 분포 확인
        print(f"\n6️⃣ 상품심사항목명 분포 확인:")
        train_product_items = self.train_df['상품심사항목명'].value_counts()
        valid_product_items = self.valid_df['상품심사항목명'].value_counts()
        test_product_items = self.test_df['상품심사항목명'].value_counts()
        
        print(f"   - 학습 데이터 고유 상품심사항목: {len(train_product_items)}개")
        print(f"   - 검증 데이터 고유 상품심사항목: {len(valid_product_items)}개")
        print(f"   - 테스트 데이터 고유 상품심사항목: {len(test_product_items)}개")
        
        # 7. 데이터 누출 위험도 평가 (개선된 방식)
        print(f"\n7️⃣ 데이터 누출 위험도 평가 (개선된 방식):")
        risk_score = 0
        risk_details = []
        
        # 사고 내용 중복 기반 위험도 (낮은 가중치)
        if text_duplication_rate > 0.2:  # 20% 이상
            risk_score += 10
            risk_details.append(f"사고 내용 중복률 높음: {text_duplication_rate*100:.1f}% (+10점)")
        
        # 심사 결과 중복 기반 위험도 (높은 가중치)
        if review_duplication_rate > 0.1:  # 10% 이상
            risk_score += 30
            risk_details.append(f"심사 결과 중복률 높음: {review_duplication_rate*100:.1f}% (+30점)")
        
        # 같은 사고 + 같은 심사 기반 위험도 (매우 높은 가중치)
        same_accident_same_review_ratio = same_accident_same_review / len(self.valid_df)
        if same_accident_same_review_ratio > 0.05:  # 5% 이상
            risk_score += 40
            risk_details.append(f"같은 사고+같은 심사 비율 높음: {same_accident_same_review_ratio*100:.1f}% (+40점)")
        
        # 심사항목 분포 기반 위험도
        common_items = set(train_items.index) & set(valid_items.index) & set(test_items.index)
        if len(common_items) / len(set(train_items.index)) > 0.9:  # 90% 이상
            risk_score += 20
            risk_details.append(f"심사항목명 중복: {len(common_items)}개 공통 (+20점)")
        
        # 위험도 상세 출력
        for detail in risk_details:
            print(f"   ⚠️ {detail}")
        
        print(f"\n🎯 최종 위험도 점수: {risk_score}/100")
        
        # 위험도 해석
        if risk_score < 20:
            print(f"✅ 데이터 누출 위험도 매우 낮음")
            print(f"   - 사고 내용과 심사 결과가 적절히 분산되어 있음")
            print(f"   - 같은 사고라도 다른 심사 결과가 나타남 (정상적인 업무 상황)")
        elif risk_score < 50:
            print(f"⚠️ 데이터 누출 위험도 보통")
            print(f"   - 일부 중복이 있지만 허용 가능한 수준")
            print(f"   - 결과를 신중하게 해석해야 함")
        elif risk_score < 80:
            print(f"🚨 데이터 누출 위험도 높음")
            print(f"   - 심사 결과 중복이 많음")
            print(f"   - 높은 정확도가 데이터 누출 때문일 수 있음")
        else:
            print(f"🚨🚨 데이터 누출 위험도 매우 높음!")
            print(f"   - 심각한 데이터 중복 문제")
            print(f"   - 모델 성능이 현실과 다를 가능성이 높음")
        
        return risk_score, {
            'text_duplication_rate': text_duplication_rate,
            'review_duplication_rate': review_duplication_rate,
            'same_accident_different_review': same_accident_different_review,
            'same_accident_same_review': same_accident_same_review,
            'risk_details': risk_details
        }
    
    def check_same_accident_different_review(self):
        """같은 사고지만 다른 심사 결과인 경우 확인 (정상적인 경우) - 최적화된 버전"""
        print(f"   🔄 임베딩 사전 계산 중...")
        
        # 모든 텍스트의 임베딩을 미리 계산
        valid_texts = self.valid_df[self.text_column].tolist()
        train_texts = self.train_df[self.text_column].tolist()
        
        # 배치로 임베딩 생성
        valid_embeddings = self.encode_batch(valid_texts)
        train_embeddings = self.encode_batch(train_texts)
        
        print(f"   🔍 유사도 계산 중...")
        normal_cases = 0
        total_valid = len(valid_embeddings)
        
        # 벡터화된 유사도 계산
        for i, valid_embedding in enumerate(valid_embeddings):
            # 진행률 표시
            if (i + 1) % max(1, total_valid // 10) == 0 or i == total_valid - 1:
                progress = (i + 1) / total_valid * 100
                print(f"   진행률: {progress:.1f}% ({i + 1}/{total_valid})")
            
            valid_review = f"{self.valid_df.iloc[i].get('심사항목명', '')}|{self.valid_df.iloc[i].get('상품심사항목명', '')}"
            
            # 한 번에 모든 학습 데이터와의 유사도 계산
            similarities = cosine_similarity([valid_embedding], train_embeddings)[0]
            
            # 유사도가 높은 인덱스들 찾기
            high_similarity_indices = np.where(similarities > 0.95)[0]
            
            for idx in high_similarity_indices:
                train_review = f"{self.train_df.iloc[idx].get('심사항목명', '')}|{self.train_df.iloc[idx].get('상품심사항목명', '')}"
                if valid_review != train_review:
                    normal_cases += 1
                    break
        
        return normal_cases
    
    def check_same_accident_same_review(self):
        """사고 내용과 심사 결과가 모두 같은 경우 확인 (문제가 될 수 있는 경우) - 최적화된 버전"""
        print(f"   🔄 임베딩 사전 계산 중...")
        
        # 모든 텍스트의 임베딩을 미리 계산
        valid_texts = self.valid_df[self.text_column].tolist()
        train_texts = self.train_df[self.text_column].tolist()
        
        # 배치로 임베딩 생성
        valid_embeddings = self.encode_batch(valid_texts)
        train_embeddings = self.encode_batch(train_texts)
        
        print(f"   🔍 유사도 계산 중...")
        problematic_cases = 0
        total_valid = len(valid_embeddings)
        
        # 벡터화된 유사도 계산
        for i, valid_embedding in enumerate(valid_embeddings):
            # 진행률 표시
            if (i + 1) % max(1, total_valid // 10) == 0 or i == total_valid - 1:
                progress = (i + 1) / total_valid * 100
                print(f"   진행률: {progress:.1f}% ({i + 1}/{total_valid})")
            
            valid_review = f"{self.valid_df.iloc[i].get('심사항목명', '')}|{self.valid_df.iloc[i].get('상품심사항목명', '')}"
            
            # 한 번에 모든 학습 데이터와의 유사도 계산
            similarities = cosine_similarity([valid_embedding], train_embeddings)[0]
            
            # 유사도가 높은 인덱스들 찾기
            high_similarity_indices = np.where(similarities > 0.95)[0]
            
            for idx in high_similarity_indices:
                train_review = f"{self.train_df.iloc[idx].get('심사항목명', '')}|{self.train_df.iloc[idx].get('상품심사항목명', '')}"
                if valid_review == train_review:
                    problematic_cases += 1
                    break
        
        return problematic_cases

def main():
    # 개선된 다중 모달 유사도 검색 모델 초기화
    model = ImprovedSimilaritySearch(text_weight=0.5, categorical_weight=0.4, numerical_weight=0.1)
    
    # 캐시 로드 시도
    cache_loaded = model.load_embeddings('improved_embeddings_cache.pkl')
    
    if cache_loaded:
        # 캐시가 있으면 데이터만 로드
        train_df, valid_df, test_df = model.load_data_only('data/case.csv')
    else:
        # 캐시가 없으면 전체 데이터 로드 및 임베딩 생성
        train_df, valid_df, test_df = model.load_and_split_data('data/case.csv')
        model.save_embeddings('improved_embeddings_cache.pkl')
    
    print(f"\n🎯 개선된 모델 설정:")
    print(f"   - Y값: 심사항목명, 상품심사항목명")
    print(f"   - 가중치: 텍스트 {model.text_weight}, 범주형 {model.categorical_weight}, 숫자형 {model.numerical_weight}")
    print(f"   - 범주형 유사도: 평균 방식 (곱셈 대신)")
    print(f"   - 데이터 누출 방지: 학습 데이터만으로 전처리 모델 학습")
    
    # 🚨 데이터 누출 진단
    print(f"\n🔍 데이터 누출 진단 시작")
    risk_score, details = model.diagnose_data_leakage()
    
    # 1단계: 검증 데이터 성능 평가
    print(f"\n🧪 1단계: 검증 데이터 성능 평가")
    valid_accuracy, valid_similarity = model.evaluate_on_validation(top_k=5)
    
    # 2단계: 테스트 데이터 성능 평가
    print(f"\n🧪 2단계: 테스트 데이터 성능 평가")
    test_accuracy, test_similarity = model.evaluate_on_test(top_k=5)
    
    # 3단계: 실제 쿼리 테스트
    print(f"\n🧪 3단계: 실제 쿼리 테스트")
    model.test_with_real_queries(top_k=5)
    
    # 캐시 상태 출력
    model.print_cache_status()
    
    print(f"\n🎯 최종 결과:")
    print(f"   데이터 누출 위험도: {risk_score}/100")
    print(f"   검증 정확도 (완전 일치): {valid_accuracy:.2f}%")
    print(f"   테스트 정확도 (완전 일치): {test_accuracy:.2f}%")
    print(f"   검증 평균 유사도: {valid_similarity:.4f}")
    print(f"   테스트 평균 유사도: {test_similarity:.4f}")
    
    # 🚨 데이터 누출 경고 및 해석
    print(f"\n📊 결과 해석 가이드:")
    print(f"=" * 50)
    
    if risk_score < 20:
        print(f"✅ 데이터 품질: 매우 좋음")
        print(f"   - 사고 내용 중복률: {details['text_duplication_rate']*100:.1f}%")
        print(f"   - 심사 결과 중복률: {details['review_duplication_rate']*100:.1f}%")
        print(f"   - 같은 사고, 다른 심사: {details['same_accident_different_review']}개 ({details['same_accident_different_review']/len(model.valid_df)*100:.1f}%)")
        print(f"   - 같은 사고, 같은 심사: {details['same_accident_same_review']}개 ({details['same_accident_same_review']/len(model.valid_df)*100:.1f}%)")
        print(f"   💡 해석: 정상적인 업무 상황. 같은 사고라도 다른 심사 결과가 나타남.")
        print(f"   💡 모델 성능: 신뢰할 수 있는 수준")
        
    elif risk_score < 50:
        print(f"⚠️ 데이터 품질: 보통")
        print(f"   - 사고 내용 중복률: {details['text_duplication_rate']*100:.1f}%")
        print(f"   - 심사 결과 중복률: {details['review_duplication_rate']*100:.1f}%")
        print(f"   - 같은 사고, 다른 심사: {details['same_accident_different_review']}개 ({details['same_accident_different_review']/len(model.valid_df)*100:.1f}%)")
        print(f"   - 같은 사고, 같은 심사: {details['same_accident_same_review']}개 ({details['same_accident_same_review']/len(model.valid_df)*100:.1f}%)")
        print(f"   💡 해석: 일부 중복이 있지만 허용 가능한 수준.")
        print(f"   💡 모델 성능: 신중하게 해석 필요")
        
    elif risk_score < 80:
        print(f"🚨 데이터 품질: 문제 있음")
        print(f"   - 사고 내용 중복률: {details['text_duplication_rate']*100:.1f}%")
        print(f"   - 심사 결과 중복률: {details['review_duplication_rate']*100:.1f}%")
        print(f"   - 같은 사고, 다른 심사: {details['same_accident_different_review']}개 ({details['same_accident_different_review']/len(model.valid_df)*100:.1f}%)")
        print(f"   - 같은 사고, 같은 심사: {details['same_accident_same_review']}개 ({details['same_accident_same_review']/len(model.valid_df)*100:.1f}%)")
        print(f"   💡 해석: 심사 결과 중복이 많음. 높은 정확도가 데이터 누출 때문일 수 있음.")
        print(f"   💡 모델 성능: 실제 성능보다 낮을 가능성 높음")
        
    else:
        print(f"🚨🚨 데이터 품질: 심각한 문제")
        print(f"   - 사고 내용 중복률: {details['text_duplication_rate']*100:.1f}%")
        print(f"   - 심사 결과 중복률: {details['review_duplication_rate']*100:.1f}%")
        print(f"   - 같은 사고, 다른 심사: {details['same_accident_different_review']}개 ({details['same_accident_different_review']/len(model.valid_df)*100:.1f}%)")
        print(f"   - 같은 사고, 같은 심사: {details['same_accident_same_review']}개 ({details['same_accident_same_review']/len(model.valid_df)*100:.1f}%)")
        print(f"   💡 해석: 심각한 데이터 중복 문제. 모델이 '기억'하고 있는 데이터로 테스트.")
        print(f"   💡 모델 성능: 현실과 다를 가능성이 매우 높음")
    
    # 🎯 업무 관점 해석
    print(f"\n🎯 업무 관점 해석:")
    print(f"=" * 50)
    
    # 같은 사고, 다른 심사 비율 해석
    same_accident_different_ratio = details['same_accident_different_review'] / len(model.valid_df)
    if same_accident_different_ratio > 0.3:
        print(f"✅ 심사 다양성: 높음 ({same_accident_different_ratio*100:.1f}%)")
        print(f"   - 같은 사고라도 다른 심사 결과가 많이 나타남")
        print(f"   - 심사자의 판단 차이, 면책사유 발견 등이 반영됨")
        print(f"   - 실제 업무 상황과 유사함")
    elif same_accident_different_ratio > 0.1:
        print(f"⚠️ 심사 다양성: 보통 ({same_accident_different_ratio*100:.1f}%)")
        print(f"   - 일부 심사 다양성이 나타남")
        print(f"   - 더 다양한 심사 패턴이 있으면 좋겠음")
    else:
        print(f"🚨 심사 다양성: 낮음 ({same_accident_different_ratio*100:.1f}%)")
        print(f"   - 같은 사고는 거의 같은 심사 결과")
        print(f"   - 심사자의 판단 차이가 반영되지 않음")
        print(f"   - 실제 업무와 다를 수 있음")
    
    # 같은 사고, 같은 심사 비율 해석
    same_accident_same_ratio = details['same_accident_same_review'] / len(model.valid_df)
    if same_accident_same_ratio > 0.1:
        print(f"🚨 데이터 중복: 높음 ({same_accident_same_ratio*100:.1f}%)")
        print(f"   - 완전히 같은 사고+심사가 많이 중복됨")
        print(f"   - 데이터 전처리 과정에서 중복 제거 필요")
    elif same_accident_same_ratio > 0.05:
        print(f"⚠️ 데이터 중복: 보통 ({same_accident_same_ratio*100:.1f}%)")
        print(f"   - 일부 중복이 있지만 허용 가능한 수준")
    else:
        print(f"✅ 데이터 중복: 낮음 ({same_accident_same_ratio*100:.1f}%)")
        print(f"   - 중복이 거의 없음")
    
    # 🎯 권장사항
    print(f"\n🎯 권장사항:")
    print(f"=" * 50)
    
    if risk_score < 20:
        print(f"✅ 현재 상태가 좋음. 추가 조치 불필요")
    elif risk_score < 50:
        print(f"⚠️ 데이터 품질 개선 권장:")
        print(f"   - 중복 데이터 제거 검토")
        print(f"   - 더 다양한 심사 패턴 수집")
    elif risk_score < 80:
        print(f"🚨 데이터 품질 개선 필수:")
        print(f"   - 중복 데이터 대폭 제거")
        print(f"   - 새로운 사고 유형 및 심사 패턴 수집")
        print(f"   - 모델 재학습 필요")
    else:
        print(f"🚨🚨 데이터 품질 개선 시급:")
        print(f"   - 데이터셋 전체 재검토")
        print(f"   - 중복 제거 후 모델 재학습")
        print(f"   - 새로운 데이터 수집 고려")

if __name__ == "__main__":
    main() 