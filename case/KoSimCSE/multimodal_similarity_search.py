import torch
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
import pickle
import os
warnings.filterwarnings('ignore')

class MultimodalSimilaritySearch:
    def __init__(self, text_weight=0.6, categorical_weight=0.3, numerical_weight=0.1):
        """
        다중 모달 유사도 검색 모델
        
        Args:
            text_weight: 텍스트 유사도 가중치 (0.6)
            categorical_weight: 범주형 변수 가중치 (0.3) 
            numerical_weight: 숫자형 변수 가중치 (0.1)
        """
        self.text_weight = text_weight
        self.categorical_weight = categorical_weight
        self.numerical_weight = numerical_weight
        
        # KoSimCSE 모델 초기화
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 디바이스: {self.device}")
        
        print("📥 KoSimCSE 모델 로딩 중...")
        self.model = AutoModel.from_pretrained('BM-K/KoSimCSE-roberta')
        self.tokenizer = AutoTokenizer.from_pretrained('BM-K/KoSimCSE-roberta')
        self.model.to(self.device)
        self.model.eval()
        print("✅ KoSimCSE 모델 로드 완료!")
        
        # 캐시 및 인코더 초기화
        self.embedding_cache = {}
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # 데이터 분할 저장
        self.train_df = None
        self.valid_df = None
        self.test_df = None
        
        # 캐시 메시지 제어
        self.cache_message_shown = False
        
        # 전처리 설정
        self.meaningless_patterns = [
            '첨부확인', '확인', '첨부', '참조', '별첨', 
            '상세내용은 첨부파일 참조', '첨부파일 확인',
            '첨부파일', '파일참조', '별도첨부', '첨부서류'
        ]
    
    def print_cache_status(self):
        """캐시 상태 출력"""
        print(f"🔍 캐시 상태: {len(self.embedding_cache)}개 임베딩 저장됨")
    
    def reset_cache_message(self):
        """캐시 메시지 상태 리셋"""
        self.cache_message_shown = False
    
    def save_embeddings(self, file_path='embeddings_cache.pkl'):
        """임베딩 캐시를 파일로 저장"""
        cache_data = {
            'embedding_cache': self.embedding_cache,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted
        }
        with open(file_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"💾 임베딩 캐시 저장 완료: {file_path}")
    
    def load_embeddings(self, file_path='embeddings_cache.pkl'):
        """임베딩 캐시를 파일에서 로드"""
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            self.embedding_cache = cache_data['embedding_cache']
            self.label_encoders = cache_data['label_encoders']
            self.scaler = cache_data['scaler']
            self.is_fitted = cache_data['is_fitted']
            
            print(f"📂 임베딩 캐시 로드 완료: {len(self.embedding_cache)}개")
            return True
        else:
            print(f"⚠️ 캐시 파일이 없습니다: {file_path}")
            return False
    
    def clean_text(self, text):
        """텍스트 정제 함수"""
        if pd.isna(text) or text == '':
            return ''
        
        text = str(text)
        
        # 의미없는 패턴 제거
        for pattern in self.meaningless_patterns:
            text = text.replace(pattern, '')
        
        # 공백 정리
        text = ' '.join(text.split())
        
        # 최소 길이 필터링 (10자 미만은 빈 문자열로)
        return text if len(text) >= 10 else ''
    
    def clean_categorical(self, df, columns, min_freq=0.01):
        """범주형 데이터 정제 함수"""
        print("🏷️ 범주형 데이터 정제 중...")
        
        for col in columns:
            if col not in df.columns:
                print(f"   ⚠️ {col} 컬럼이 없습니다. 건너뜁니다.")
                continue
                
            # 결측치 처리
            df[col] = df[col].fillna('Unknown')
            
            # 빈도 계산
            value_counts = df[col].value_counts(normalize=True)
            rare_values = value_counts[value_counts < min_freq].index
            
            # 희귀 값들을 'Others'로 변경
            if len(rare_values) > 0:
                df[col] = df[col].replace(rare_values, 'Others')
                print(f"   ✅ {col}: {len(rare_values)}개 희귀값을 'Others'로 변경")
            
            print(f"   📊 {col}: {df[col].nunique()}개 고유값")
        
        return df
    
    def handle_outliers(self, df, numerical_columns):
        """숫자형 데이터 이상치 처리 함수"""
        print("🔢 숫자형 데이터 이상치 처리 중...")
        
        for col in numerical_columns:
            if col not in df.columns:
                print(f"   ⚠️ {col} 컬럼이 없습니다. 건너뜁니다.")
                continue
            
            # IQR 방식으로 이상치 탐지
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 이상치 개수 확인
            outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            
            # 이상치를 경계값으로 대체
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            
            if outliers_count > 0:
                print(f"   ✅ {col}: {outliers_count}개 이상치 처리")
        
        return df
        
    def load_and_split_data(self, file_path, text_column='사고설명', 
                           categorical_columns=None, numerical_columns=None,
                           train_size=0.7, valid_size=0.15, test_size=0.15, random_state=42):
        """
        데이터 로드 및 Train/Validation/Test 3단계 분할
        
        Args:
            file_path: CSV 파일 경로
            text_column: 텍스트 컬럼명
            categorical_columns: 범주형 컬럼 리스트
            numerical_columns: 숫자형 컬럼 리스트
            train_size: 학습 데이터 비율 (0.7)
            valid_size: 검증 데이터 비율 (0.15)
            test_size: 테스트 데이터 비율 (0.15)
            random_state: 랜덤 시드
        """
        print("📊 데이터 로드 및 전처리 중...")
        
        # 데이터 로드
        df = pd.read_csv(file_path, encoding='cp949')
        print(f"📈 원본 데이터: {len(df)}개")
        
        # 컬럼 타입 설정
        if categorical_columns is None:
            categorical_columns = ['사고유형명', '사고진행상태', '사고등록상태명', '접수사고유형확정', 
                                 '사고통화코드_ZZ067', '심사항목명', '상품심사항목명','상품분류명', 
                                 '상품분류그룹명', '상품명', '상세사고보험상품', '향후결제전망', 
                                 '수출자명', '수입자명', '수출자국가', '수입자국가']
        
        if numerical_columns is None:
            numerical_columns = ['외화사고접수금액', '미화사고접수금액', '원화사고접수금액',
                               '외화합계판정금액', '미화합계판정금액', '원화합계판정금액', 
                               '외화판정금액', '미화판정금액', '원화판정금액',
                               '외화사고금액', '미화사고금액', '원화사고금액', 
                               '외화보험가액', '미화보험가액', '원화보험가액', 
                               '외화보험금액', '미화보험금액', '원화보험금액']
        
        print(f"📝 텍스트 컬럼: {text_column}")
        print(f"🏷️ 범주형 컬럼: {len(categorical_columns)}개")
        print(f"🔢 숫자형 컬럼: {len(numerical_columns)}개")
        
        # 1. 텍스트 데이터 정제
        print(f"\n📝 텍스트 데이터 정제 중...")
        df[text_column] = df[text_column].apply(self.clean_text)
        
        # 빈 텍스트 제거
        before_count = len(df)
        df = df[df[text_column] != '']
        after_count = len(df)
        print(f"   ✅ 빈 텍스트 제거: {before_count - after_count}개 제거됨")
        
        # 2. 범주형 데이터 정제
        df = self.clean_categorical(df, categorical_columns)
        
        # 3. 숫자형 데이터 이상치 처리
        df = self.handle_outliers(df, numerical_columns)
        
        # 4. 중복 제거 (사고설명 기준)
        before_count = len(df)
        df = df.drop_duplicates(subset=[text_column])
        after_count = len(df)
        print(f"   ✅ 사고설명 기준 중복 제거: {before_count - after_count}개 제거됨")
        
        # 추가 중복 제거 (모든 컬럼 기준)
        before_count = len(df)
        df = df.drop_duplicates()
        after_count = len(df)
        print(f"   ✅ 전체 컬럼 기준 중복 제거: {before_count - after_count}개 제거됨")
        
        print(f"\n📊 전처리 완료: {len(df)}개 데이터")
        
        # 5. 3단계 분할
        print(f"\n📊 데이터 3단계 분할 중...")
        
        # 1단계: Train + (Validation + Test) 분할
        train_df, temp_df = train_test_split(
            df, test_size=(valid_size + test_size), random_state=random_state
        )
        
        # 2단계: Validation + Test 분할
        valid_ratio = valid_size / (valid_size + test_size)
        valid_df, test_df = train_test_split(
            temp_df, test_size=(1 - valid_ratio), random_state=random_state
        )
        
        print(f"📚 학습 데이터: {len(train_df)}개 ({len(train_df)/len(df)*100:.1f}%)")
        print(f"🔍 검증 데이터: {len(valid_df)}개 ({len(valid_df)/len(df)*100:.1f}%)")
        print(f"🧪 테스트 데이터: {len(test_df)}개 ({len(test_df)/len(df)*100:.1f}%)")
        
        # 6. 범주형 변수 인코딩 (전체 데이터 기준)
        categorical_features = []
        for col in categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                # 전체 데이터로 fit
                le.fit(df[col].astype(str))
                
                # 각 분할 데이터에 transform 적용
                train_df[f'{col}_encoded'] = le.transform(train_df[col].astype(str))
                valid_df[f'{col}_encoded'] = le.transform(valid_df[col].astype(str))
                test_df[f'{col}_encoded'] = le.transform(test_df[col].astype(str))
                
                self.label_encoders[col] = le
                categorical_features.append(f'{col}_encoded')
                print(f"   ✅ {col} 인코딩 완료 (고유값: {len(le.classes_)}개)")
        
        # 7. 숫자형 변수 스케일링 (전체 데이터 기준)
        numerical_features = []
        if numerical_columns:
            for col in numerical_columns:
                if col in df.columns:
                    numerical_features.append(col)
            
            if numerical_features:
                # 전체 데이터로 fit
                self.scaler.fit(df[numerical_features])
                
                # 각 분할 데이터에 transform 적용
                train_df[numerical_features] = self.scaler.transform(train_df[numerical_features])
                valid_df[numerical_features] = self.scaler.transform(valid_df[numerical_features])
                test_df[numerical_features] = self.scaler.transform(test_df[numerical_features])
                
                print(f"   ✅ 숫자형 변수 스케일링 완료: {len(numerical_features)}개")
        
        # 데이터 저장
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
        self.text_column = text_column
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.is_fitted = True
        
        print("✅ 데이터 3단계 분할 및 전처리 완료!")
        return train_df, valid_df, test_df
    
    def load_data_only(self, file_path, text_column='사고설명',
                      categorical_columns=None, numerical_columns=None,
                      train_size=0.7, valid_size=0.15, test_size=0.15, random_state=42):
        """캐시가 있을 때 데이터만 로드 (임베딩 생성 안함)"""
        print("📊 데이터만 로드 중... (임베딩은 캐시 사용)")
        
        # 데이터 로드
        df = pd.read_csv(file_path, encoding='cp949')
        print(f"📈 원본 데이터: {len(df)}개")
        
        # 컬럼 타입 설정
        if categorical_columns is None:
            categorical_columns = ['사고유형명', '사고진행상태', '사고등록상태명', '접수사고유형확정',
                                  '사고통화코드_ZZ067', '심사항목명', '상품심사항목명','상품분류명',
                                  '상품분류그룹명', '상품명', '상세사고보험상품', '향후결제전망',
                                  '수출자명', '수입자명', '수출자국가', '수입자국가']

        if numerical_columns is None:
            numerical_columns = ['외화사고접수금액', '미화사고접수금액', '원화사고접수금액',
                                '외화합계판정금액', '미화합계판정금액', '원화합계판정금액',
                                '외화판정금액', '미화판정금액', '원화판정금액',
                                '외화사고금액', '미화사고금액', '원화사고금액',
                                '외화보험가액', '미화보험가액', '원화보험가액',
                                '외화보험금액', '미화보험금액', '원화보험금액']
        
        # 전처리 (임베딩 생성 없이)
        df[text_column] = df[text_column].apply(self.clean_text)
        df = df[df[text_column] != '']
        df = self.clean_categorical(df, categorical_columns)
        df = self.handle_outliers(df, numerical_columns)
        df = df.drop_duplicates(subset=[text_column])
        df = df.drop_duplicates()
        
        print(f"📊 전처리 완료: {len(df)}개 데이터")
        
        # 3단계 분할
        train_df, temp_df = train_test_split(
            df, test_size=(valid_size + test_size), random_state=random_state
        )
        valid_ratio = valid_size / (valid_size + test_size)
        valid_df, test_df = train_test_split(
            temp_df, test_size=(1 - valid_ratio), random_state=random_state
        )
        
        print(f"📚 학습 데이터: {len(train_df)}개 ({len(train_df)/len(df)*100:.1f}%)")
        print(f"🔍 검증 데이터: {len(valid_df)}개 ({len(valid_df)/len(df)*100:.1f}%)")
        print(f"🧪 테스트 데이터: {len(test_df)}개 ({len(test_df)/len(df)*100:.1f}%)")
        
        # 인코딩 및 스케일링 (기존 인코더 사용)
        categorical_features = []
        for col in categorical_columns:
            if col in df.columns and col in self.label_encoders:
                train_df[f'{col}_encoded'] = self.label_encoders[col].transform(train_df[col].astype(str))
                valid_df[f'{col}_encoded'] = self.label_encoders[col].transform(valid_df[col].astype(str))
                test_df[f'{col}_encoded'] = self.label_encoders[col].transform(test_df[col].astype(str))
                categorical_features.append(f'{col}_encoded')
        
        numerical_features = []
        if numerical_columns:
            for col in numerical_columns:
                if col in df.columns:
                    numerical_features.append(col)
            
            if numerical_features:
                train_df[numerical_features] = self.scaler.transform(train_df[numerical_features])
                valid_df[numerical_features] = self.scaler.transform(valid_df[numerical_features])
                test_df[numerical_features] = self.scaler.transform(test_df[numerical_features])
        
        # 데이터 저장
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
        self.text_column = text_column
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        
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
            # 캐시된 텍스트만 있을 때는 한 번만 메시지 출력
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
        """범주형 변수 유사도 계산 (Jaccard 유사도)"""
        similarities = []
        for target in target_categorical:
            # 범주형 변수는 정확히 일치하면 1, 아니면 0
            similarity = 1.0 if query_categorical == target else 0.0
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
        유사 사례 검색
        
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
            print(f"\n🔍 다중 모달 유사 사례 검색 (상위 {top_k}개)")
            print(f"📝 쿼리 텍스트: {query_data['text'][:100]}...")
        
        # 1. 텍스트 유사도 계산
        text_similarities = self.calculate_text_similarity(
            query_data['text'], 
            search_df[self.text_column].tolist()
        )
        
        # 2. 범주형 변수 유사도 계산
        categorical_similarities = np.ones(len(search_df))
        if query_data.get('categorical') and self.categorical_features:
            for col, value in query_data['categorical'].items():
                if col in self.label_encoders:
                    encoded_value = self.label_encoders[col].transform([str(value)])[0]
                    col_similarities = self.calculate_categorical_similarity(
                        encoded_value,
                        search_df[f'{col}_encoded'].values
                    )
                    categorical_similarities *= col_similarities
        
        # 3. 숫자형 변수 유사도 계산
        numerical_similarities = np.ones(len(search_df))
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
            
            result = {
                'rank': i + 1,
                'similarity': similarity,
                'text_similarity': text_similarities[idx],
                'categorical_similarity': categorical_similarities[idx],
                'numerical_similarity': numerical_similarities[idx],
                'data': row.to_dict()
            }
            results.append(result)
            
            if verbose:
                print(f"\n{i+1}. 전체 유사도: {similarity:.4f}")
                print(f"   📝 텍스트 유사도: {text_similarities[idx]:.4f}")
                print(f"   🏷️ 범주형 유사도: {categorical_similarities[idx]:.4f}")
                print(f"   🔢 숫자형 유사도: {numerical_similarities[idx]:.4f}")
                print(f"   📋 사고유형: {row.get('사고유형명', 'N/A')}")
                print(f"   📄 사고설명: {row.get(self.text_column, 'N/A')}")
        
        return results
    
    def evaluate_on_validation(self, top_k=5):
        """검증 데이터로 성능 평가"""
        print(f"\n📊 검증 데이터 성능 평가 (상위 {top_k}개 기준)")
        
        correct_count = 0
        total_similarities = []
        
        for i, valid_row in self.valid_df.iterrows():
            # 검증 쿼리 생성
            query_data = {
                'text': valid_row[self.text_column],
                'categorical': {},
                'numerical': {}
            }
            
            # 범주형 변수 추가
            for col in self.label_encoders.keys():
                if col in valid_row:
                    query_data['categorical'][col] = valid_row[col]
            
            # 숫자형 변수 추가
            for col in self.numerical_features:
                if col in valid_row:
                    query_data['numerical'][col] = valid_row[col]
            
            # 학습 데이터에서 유사 사례 검색
            results = self.find_similar_cases(query_data, self.train_df, top_k=top_k, verbose=False)
            
            # 정확도 계산
            valid_accident_type = valid_row.get('사고유형명', '')
            for result in results:
                if result['data'].get('사고유형명') == valid_accident_type:
                    correct_count += 1
                    break
            
            # 평균 유사도
            total_similarities.extend([r['similarity'] for r in results])
        
        accuracy = correct_count / len(self.valid_df) * 100
        avg_similarity = np.mean(total_similarities)
        
        print(f"✅ 검증 정확도: {accuracy:.2f}%")
        print(f"📈 검증 평균 유사도: {avg_similarity:.4f}")
        
        return accuracy, avg_similarity
    
    def evaluate_on_test(self, top_k=5):
        """테스트 데이터로 성능 평가"""
        print(f"\n🧪 테스트 데이터 성능 평가 (상위 {top_k}개 기준)")
        
        correct_count = 0
        total_similarities = []
        
        for i, test_row in self.test_df.iterrows():
            # 테스트 쿼리 생성
            query_data = {
                'text': test_row[self.text_column],
                'categorical': {},
                'numerical': {}
            }
            
            # 범주형 변수 추가
            for col in self.label_encoders.keys():
                if col in test_row:
                    query_data['categorical'][col] = test_row[col]
            
            # 숫자형 변수 추가
            for col in self.numerical_features:
                if col in test_row:
                    query_data['numerical'][col] = test_row[col]
            
            # 학습 데이터에서 유사 사례 검색
            results = self.find_similar_cases(query_data, self.train_df, top_k=top_k, verbose=False)
            
            # 정확도 계산
            test_accident_type = test_row.get('사고유형명', '')
            for result in results:
                if result['data'].get('사고유형명') == test_accident_type:
                    correct_count += 1
                    break
            
            # 평균 유사도
            total_similarities.extend([r['similarity'] for r in results])
        
        accuracy = correct_count / len(self.test_df) * 100
        avg_similarity = np.mean(total_similarities)
        
        print(f"✅ 테스트 정확도: {accuracy:.2f}%")
        print(f"📈 테스트 평균 유사도: {avg_similarity:.4f}")
        
        return accuracy, avg_similarity
    
    def test_with_real_queries(self, top_k=5):
        """실제 테스트 데이터로 쿼리 테스트"""
        print(f"\n🧪 실제 테스트 데이터 쿼리 테스트")
        
        # 테스트 데이터에서 샘플 선택 (사고유형별 1개씩)
        test_samples = []
        for accident_type in self.test_df['사고유형명'].unique():
            type_samples = self.test_df[self.test_df['사고유형명'] == accident_type].head(1)
            test_samples.extend(type_samples['사고설명'].tolist())
        
        test_queries = test_samples[:5]  # 최대 5개 샘플
        
        for i, query_text in enumerate(test_queries, 1):
            print(f"\n--- 실제 테스트 쿼리 {i} ---")
            
            # 해당 테스트 데이터 찾기
            test_row = self.test_df[self.test_df[self.text_column] == query_text].iloc[0]
            
            # 쿼리 데이터 생성
            query_data = {
                'text': query_text,
                'categorical': {},
                'numerical': {}
            }
            
            # 범주형 변수 추가
            for col in self.label_encoders.keys():
                if col in test_row:
                    query_data['categorical'][col] = test_row[col]
            
            # 숫자형 변수 추가
            for col in self.numerical_features:
                if col in test_row:
                    query_data['numerical'][col] = test_row[col]
            
            print(f"🔍 쿼리: {query_text}")
            print(f"📋 실제 사고유형: {test_row.get('사고유형명', 'N/A')}")
            
            # 학습 데이터에서 유사 사례 검색
            results = self.find_similar_cases(query_data, self.train_df, top_k=top_k)
            
            # 정확도 확인
            correct_found = False
            for result in results:
                if result['data'].get('사고유형명') == test_row.get('사고유형명'):
                    correct_found = True
                    break
            
            if correct_found:
                print(f"✅ 정확한 사고유형 발견!")
            else:
                print(f"❌ 정확한 사고유형 미발견")

    def test_weight_combinations(self, top_k=3):
        """다양한 가중치 조합으로 성능 테스트"""
        print(f"\n🧪 실험 2: 가중치 조합 테스트 (상위 {top_k}개 기준)")
        
        weight_combinations = [
            {
                'name': '텍스트 중심 (80%)',
                'weights': {'text': 0.8, 'categorical': 0.15, 'numerical': 0.05}
            },
            {
                'name': '범주형 중심 (50%)',
                'weights': {'text': 0.4, 'categorical': 0.5, 'numerical': 0.1}
            },
            {
                'name': '균등 분배 (33%)',
                'weights': {'text': 0.33, 'categorical': 0.33, 'numerical': 0.34}
            },
            {
                'name': '텍스트+범주형 (50%+40%)',
                'weights': {'text': 0.5, 'categorical': 0.4, 'numerical': 0.1}
            },
            {
                'name': '텍스트만 (100%)',
                'weights': {'text': 1.0, 'categorical': 0.0, 'numerical': 0.0}
            }
        ]
        
        results = []
        
        for combo in weight_combinations:
            print(f"\n📊 {combo['name']} 테스트 중...")
            
            # 가중치 임시 변경
            original_weights = {
                'text': self.text_weight,
                'categorical': self.categorical_weight,
                'numerical': self.numerical_weight
            }
            
            self.text_weight = combo['weights']['text']
            self.categorical_weight = combo['weights']['categorical']
            self.numerical_weight = combo['weights']['numerical']
            
            # 성능 평가
            accuracy, avg_similarity = self.evaluate_on_test(top_k=top_k)
            
            # 결과 저장
            results.append({
                'name': combo['name'],
                'weights': combo['weights'],
                'accuracy': accuracy,
                'avg_similarity': avg_similarity
            })
            
            # 가중치 복원
            self.text_weight = original_weights['text']
            self.categorical_weight = original_weights['categorical']
            self.numerical_weight = original_weights['numerical']
        
        # 결과 요약
        print(f"\n🎯 가중치 조합 테스트 결과:")
        print(f"{'조합':<25} {'정확도':<10} {'평균유사도':<12}")
        print("-" * 50)
        for result in results:
            print(f"{result['name']:<25} {result['accuracy']:<10.2f}% {result['avg_similarity']:<12.4f}")
        
        # 최고 성능 조합 찾기
        best_result = max(results, key=lambda x: x['accuracy'])
        print(f"\n🏆 최고 성능: {best_result['name']} ({best_result['accuracy']:.2f}%)")
        
        return results

    def test_evaluation_criteria(self):
        """다양한 평가 기준으로 성능 테스트"""
        print(f"\n🧪 실험 3: 평가 기준 테스트")
        
        evaluation_criteria = [
            {
                'name': '상위 1개만 평가',
                'top_k': 1,
                'success_condition': lambda results, target: results[0]['data'].get('사고유형명') == target
            },
            {
                'name': '상위 3개 중 1개 이상',
                'top_k': 3,
                'success_condition': lambda results, target: any(r['data'].get('사고유형명') == target for r in results)
            },
            {
                'name': '상위 3개 중 과반수 (2개 이상)',
                'top_k': 3,
                'success_condition': lambda results, target: sum(1 for r in results if r['data'].get('사고유형명') == target) >= 2
            },
            {
                'name': '상위 5개 중 1개 이상 (기존)',
                'top_k': 5,
                'success_condition': lambda results, target: any(r['data'].get('사고유형명') == target for r in results)
            }
        ]
        
        results = []
        
        for criteria in evaluation_criteria:
            print(f"\n📊 {criteria['name']} 테스트 중...")
            
            correct_count = 0
            total_similarities = []
            
            for i, test_row in self.test_df.iterrows():
                # 테스트 쿼리 생성
                query_data = {
                    'text': test_row[self.text_column],
                    'categorical': {},
                    'numerical': {}
                }
                
                # 범주형 변수 추가
                for col in self.label_encoders.keys():
                    if col in test_row:
                        query_data['categorical'][col] = test_row[col]
                
                # 숫자형 변수 추가
                for col in self.numerical_features:
                    if col in test_row:
                        query_data['numerical'][col] = test_row[col]
                
                # 학습 데이터에서 유사 사례 검색
                search_results = self.find_similar_cases(query_data, self.train_df, top_k=criteria['top_k'], verbose=False)
                
                # 정확도 계산
                test_accident_type = test_row.get('사고유형명', '')
                if criteria['success_condition'](search_results, test_accident_type):
                    correct_count += 1
                
                # 평균 유사도
                total_similarities.extend([r['similarity'] for r in search_results])
            
            accuracy = correct_count / len(self.test_df) * 100
            avg_similarity = np.mean(total_similarities)
            
            results.append({
                'name': criteria['name'],
                'top_k': criteria['top_k'],
                'accuracy': accuracy,
                'avg_similarity': avg_similarity
            })
        
        # 결과 요약
        print(f"\n🎯 평가 기준 테스트 결과:")
        print(f"{'기준':<25} {'상위K':<8} {'정확도':<10} {'평균유사도':<12}")
        print("-" * 60)
        for result in results:
            print(f"{result['name']:<25} {result['top_k']:<8} {result['accuracy']:<10.2f}% {result['avg_similarity']:<12.4f}")
        
        return results

def main():
    # 다중 모달 유사도 검색 모델 초기화
    multimodal_search = MultimodalSimilaritySearch(
        text_weight=0.6,      # 텍스트 가중치
        categorical_weight=0.3, # 범주형 가중치
        numerical_weight=0.1   # 숫자형 가중치
    )
    
    # 캐시 파일 로드 시도
    cache_file = 'multimodal_embeddings_cache.pkl'
    if multimodal_search.load_embeddings(cache_file):
        print(f"✅ 캐시 파일 로드 성공! 임베딩 재사용 가능")
        # 데이터만 로드 (임베딩은 이미 캐시됨)
        train_df, valid_df, test_df = multimodal_search.load_data_only(
            file_path='data/case.csv',
            text_column='사고설명',
            # 범주형 변수 (16개)
            categorical_columns=['사고유형명', '사고진행상태', '사고등록상태명', '접수사고유형확정',
                                '사고통화코드_ZZ067', '심사항목명', '상품심사항목명','상품분류명',
                                '상품분류그룹명', '상품명', '상세사고보험상품', '향후결제전망',
                                '수출자명', '수입자명', '수출자국가', '수입자국가'],
            # 숫자형 변수 (18개)
            numerical_columns=['외화사고접수금액', '미화사고접수금액', '원화사고접수금액',
                              '외화합계판정금액', '미화합계판정금액', '원화합계판정금액',
                              '외화판정금액', '미화판정금액', '원화판정금액',
                              '외화사고금액', '미화사고금액', '원화사고금액',
                              '외화보험가액', '미화보험가액', '원화보험가액',
                              '외화보험금액', '미화보험금액', '원화보험금액']
        )
    else:
        print(f"⚠️ 캐시 파일이 없습니다. 처음부터 실행합니다.")
        # 데이터 로드 및 3단계 분할 (전처리 포함)
        train_df, valid_df, test_df = multimodal_search.load_and_split_data(
            file_path='data/case.csv',
        text_column='사고설명',
            # 범주형 변수 (16개)
            categorical_columns=['사고유형명', '사고진행상태', '사고등록상태명', '접수사고유형확정',
                                '사고통화코드_ZZ067', '심사항목명', '상품심사항목명','상품분류명',
                                '상품분류그룹명', '상품명', '상세사고보험상품', '향후결제전망',
                                '수출자명', '수입자명', '수출자국가', '수입자국가'],
            # 숫자형 변수 (18개)
            numerical_columns=['외화사고접수금액', '미화사고접수금액', '원화사고접수금액',
                              '외화합계판정금액', '미화합계판정금액', '원화합계판정금액',
                              '외화판정금액', '미화판정금액', '원화판정금액',
                              '외화사고금액', '미화사고금액', '원화사고금액',
                              '외화보험가액', '미화보험가액', '원화보험가액',
                              '외화보험금액', '미화보험금액', '원화보험금액']
        )
    
    # 1단계: 검증 데이터로 성능 평가
    print(f"\n📊 1단계: 검증 데이터 성능 평가")
    multimodal_search.reset_cache_message()  # 캐시 메시지 리셋
    valid_accuracy, valid_similarity = multimodal_search.evaluate_on_validation()
    
    # 2단계: 테스트 데이터로 성능 평가
    print(f"\n🧪 2단계: 테스트 데이터 성능 평가")
    multimodal_search.reset_cache_message()  # 캐시 메시지 리셋
    test_accuracy, test_similarity = multimodal_search.evaluate_on_test()
    
    # 3단계: 실제 테스트 데이터로 쿼리 테스트
    print(f"\n🔍 3단계: 실제 테스트 데이터 쿼리 테스트")
    multimodal_search.reset_cache_message()  # 캐시 메시지 리셋
    multimodal_search.test_with_real_queries()
    
    # 4단계: 캐시 상태 출력
    multimodal_search.print_cache_status()
    
    # 5단계: 캐시 저장 (다음 실행 시 재사용)
    multimodal_search.save_embeddings('multimodal_embeddings_cache.pkl')
    
    # 6단계: 가중치 조합 테스트
    print(f"\n🧪 6단계: 가중치 조합 테스트")
    multimodal_search.test_weight_combinations()
    
    # 7단계: 평가 기준 테스트
    print(f"\n🧪 7단계: 평가 기준 테스트")
    multimodal_search.test_evaluation_criteria()
    
    print(f"\n🎯 최종 결과:")
    print(f"   검증 정확도: {valid_accuracy:.2f}%")
    print(f"   테스트 정확도: {test_accuracy:.2f}%")
    print(f"   검증 평균 유사도: {valid_similarity:.4f}")
    print(f"   테스트 평균 유사도: {test_similarity:.4f}")
    print(f"   📊 데이터 품질:")
    print(f"      - 고유 사고설명: {len(train_df[multimodal_search.text_column].unique())}개")
    print(f"      - 중복 제거 효과: {len(train_df[multimodal_search.text_column].unique()) / len(train_df) * 100:.1f}% 고유도")
    print(f"   🔍 캐시 효율성:")
    print(f"      - 캐시된 임베딩: {len(multimodal_search.embedding_cache)}개")
    print(f"      - 캐시 파일 저장: multimodal_embeddings_cache.pkl")

if __name__ == "__main__":
    main() 