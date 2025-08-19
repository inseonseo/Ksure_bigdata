import numpy as np
# NumPy 2.0 compatibility aliases for removed dtypes
if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_
if not hasattr(np, "string_"):
    np.string_ = np.bytes_
try:
    np.bool
except AttributeError:
    np.bool = np.bool_
try:
    np.object
except AttributeError:
    np.object = np.object_
try:
    np.int
except AttributeError:
    np.int = int
try:
    np.float
except AttributeError:
    np.float = float

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from collections import Counter
from scipy import stats
import pickle
import warnings
import re
import time
warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(
    page_title=" 사고별 판정심사 사례 분석",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 깔끔한 CSS 스타일
st.markdown("""
<style>
    .main-header {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #007bff;
        margin-bottom: 2rem;
    }
    
    .metric-box {
        background-color: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #007bff;
        text-align: center;
    }
    
    .confidence-box {
        background-color: #e3f2fd;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #2196f3;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #e8f5e8;
        border: 2px solid #4caf50;
    }
    
    .error-box {
        background-color: #ffebee;
        border: 2px solid #f44336;
    }
    
    .warning-box {
        background-color: #fff3e0;
        border: 2px solid #ff9800;
    }
</style>
""", unsafe_allow_html=True)

class CountryProcessor:
    """개선된 국가 처리 클래스"""
    
    def __init__(self):
        # 개별 유지 국가 (상위 16개)
        self.individual_countries = [
            '미국', '중국', '브라질', '일본', '독일', '영국', '러시아', '인도',
            '베트남', '이탈리아', '홍콩', '아랍에미리트연합', '튀르키예', 
            '인도네시아', '스페인', '대만'
        ]
        
        # 지역별 분류
        self.regions = {
            'asia': {
                '개별': ['중국', '일본', '인도', '베트남', '홍콩', '대만', '인도네시아'],
                '기타': ['태국', '말레이시아', '필리핀', '싱가포르', '미얀마', '캄보디아', '몽골', '부탄']
            },
            'europe': {
                '개별': ['독일', '영국', '이탈리아', '스페인'],
                '기타': ['프랑스', '네덜란드', '벨기엄', '스위스', '오스트리아', '그리스', '몰타', '헝가리', '체코', 'Estonia']
            },
            'americas': {
                '개별': ['미국', '브라질'],
                '기타': ['멕시코', '콜롬비아', '아르헨티나', '페루', '칠레', '과테말라', '볼리비아', '온두라스', '파나마', '자메이카']
            },
            'middle_east': {
                '개별': ['아랍에미리트연합', '튀르키예'],
                '기타': ['사우디아라비아', '카타르', '쿠웨이트', '바레인', '이스라엘']
            },
            'africa': {
                '개별': [],
                '기타': ['가나', '케냐', '세네갈', '에티오피아', '우간다', '르완다', '가봉', '감비아', '라이베리아', '마다가스카르', '말라위', '모로코', '베냉', '시에라리온', '토고']
            },
            'oceania': {
                '개별': [],
                '기타': ['호주', '뉴질랜드']
            },
            'other': {
                '개별': ['러시아'],
                '기타': ['라트비아', '몰도바', '마케도니아', '보스니아헤르체고비나', '세르비아', '슬로바키아', '우즈베키스탄', '조지아']
            }
        }
    
    def get_country_region(self, country):
        """국가의 지역 반환"""
        for region, countries in self.regions.items():
            if country in countries['개별'] or country in countries['기타']:
                return region
        return 'other'
    
    def is_individual_country(self, country):
        """개별 유지 국가인지 확인"""
        return country in self.individual_countries
    
    def is_minor_country(self, country):
        """소규모 국가인지 확인"""
        for region, countries in self.regions.items():
            if country in countries['기타']:
                return True
        return False
    
    def preprocess_country(self, country):
        """국가 전처리"""
        if pd.isna(country):
            return '정보없음'
        
        # 1. 개별 유지 국가
        if country in self.individual_countries:
            return country
        
        # 2. 지역별 그룹화
        region = self.get_country_region(country)
        if region != 'other':
            return f'{region}_기타'
        
        # 3. 완전히 알 수 없는 국가
        return '기타국가'
    
    def calculate_country_similarity(self, country1, country2):
        """계층적 국가 유사도 계산"""
        if pd.isna(country1) or pd.isna(country2):
            return 0.0
        
        # 완전 일치
        if country1 == country2:
            return 1.0
        
        region1 = self.get_country_region(country1)
        region2 = self.get_country_region(country2)
        
        is_individual1 = self.is_individual_country(country1)
        is_individual2 = self.is_individual_country(country2)
        is_minor1 = self.is_minor_country(country1)
        is_minor2 = self.is_minor_country(country2)
        
        # 둘 다 개별 국가인 경우
        if is_individual1 and is_individual2:
            return 0.6 if region1 == region2 else 0.3
        
        # 둘 다 소규모 국가인 경우
        if is_minor1 and is_minor2:
            return 0.7 if region1 == region2 else 0.4
        
        # 하나는 개별, 하나는 소규모인 경우
        if (is_individual1 and is_minor2) or (is_minor1 and is_individual2):
            return 0.5 if region1 == region2 else 0.2
        
        # 기타 경우
        return 0.2

class HybridConfidenceCalculator:
    """하이브리드 신뢰도 계산기"""
    
    @staticmethod
    def calculate_weighted_confidence(decisions, similarity_scores):
        """가중 신뢰도 계산"""
        if not decisions or not similarity_scores:
            return 0.0, '정보없음'
        
        # 가장 많은 판정구분 찾기
        decision_counts = Counter(decisions)
        most_common_decision = decision_counts.most_common(1)[0][0]
        
        # 해당 판정의 가중 점수 합
        weighted_sum = sum(score for decision, score in zip(decisions, similarity_scores) 
                          if decision == most_common_decision)
        total_weight = sum(similarity_scores)
        
        if total_weight == 0:
            return 0.0, most_common_decision
        
        return weighted_sum / total_weight, most_common_decision
    
    @staticmethod
    def calculate_bayesian_confidence(positive_count, total_count, confidence_level=0.95):
        """베이지안 신뢰구간 계산"""
        if total_count == 0:
            return 0.0, (0.0, 0.0)
        
        # 베타 분포 파라미터 (무정보 사전분포)
        alpha = 1 + positive_count
        beta = 1 + total_count - positive_count
        
        # 베이지안 추정값
        posterior_mean = alpha / (alpha + beta)
        
        # 신뢰구간
        alpha_level = 1 - confidence_level
        lower = stats.beta.ppf(alpha_level/2, alpha, beta)
        upper = stats.beta.ppf(1 - alpha_level/2, alpha, beta)
        
        return posterior_mean, (lower, upper)
    
    @classmethod
    def hybrid_confidence(cls, decisions, similarity_scores):
        """하이브리드 신뢰도 계산"""
        if not decisions:
            return {
                'confidence': 0.0,
                'credible_interval': (0.0, 0.0),
                'predicted_decision': '정보없음',
                'sample_size': 0,
                'avg_similarity': 0.0,
                'interpretation': '유사사례가 없습니다.',
                'grade': '신뢰불가'
            }
        
        # 1. 가중 신뢰도 계산
        weighted_conf, predicted_decision = cls.calculate_weighted_confidence(decisions, similarity_scores)
        
        # 2. 베이지안 신뢰구간 계산
        positive_count = sum(1 for d in decisions if d == predicted_decision)
        bayesian_mean, (lower, upper) = cls.calculate_bayesian_confidence(positive_count, len(decisions))
        
        # 3. 표본 크기 및 품질 보정
        sample_size = len(decisions)
        avg_similarity = np.mean(similarity_scores) if similarity_scores else 0.0
        
        # 표본 크기 보정 (더 보수적으로)
        if sample_size >= 10:
            sample_factor = 1.0
        elif sample_size >= 5:
            sample_factor = 0.9
        else:
            sample_factor = 0.8
        
        # 유사도 품질 보정
        if avg_similarity >= 0.7:
            quality_factor = 1.1
        elif avg_similarity >= 0.5:
            quality_factor = 1.0
        else:
            quality_factor = 0.9
        
        # 최종 신뢰도 (가중 방식 + 보정)
        final_confidence = min(0.95, weighted_conf * sample_factor * quality_factor)
        
        # 신뢰도 등급
        if final_confidence >= 0.8 and sample_size >= 5:
            grade = '높음'
        elif final_confidence >= 0.6 and sample_size >= 3:
            grade = '보통'
        else:
            grade = '낮음'
        
        # 해석 메시지
        interpretation = f"{final_confidence:.1%} (95% 구간: {lower:.1%}-{upper:.1%})"
        
        return {
            'confidence': final_confidence,
            'credible_interval': (lower, upper),
            'predicted_decision': predicted_decision,
            'sample_size': sample_size,
            'avg_similarity': avg_similarity,
            'interpretation': interpretation,
            'grade': grade,
            'bayesian_mean': bayesian_mean,
            'weighted_confidence': weighted_conf
        }

class ImprovedInsuranceSystem:
    def __init__(self):
        self.model = None
        self.kosimcse_model = None
        self.kosimcse_tokenizer = None
        self.text_vectorizer = None
        self.label_encoders = {}
        self.feature_importance = None
        
        # 컴포넌트 초기화
        self.country_processor = CountryProcessor()
        self.confidence_calculator = HybridConfidenceCalculator()
        
        # 최적화된 가중치 (확장된 변수 포함)
        self.optimal_weights = {
            # 핵심 변수 (개별 가중치)
            'text_similarity': 0.35,      # 텍스트 유사도
            'accident_type': 0.20,        # 사고유형
            'country_similarity': 0.12,   # 국가 유사도
            
            # 숫자형 변수 그룹 (15%)
            'amount_similarity': 0.08,    # 금액 유사도
            'coverage_rate': 0.05,        # 부보율
            'payment_terms': 0.02,        # 결제조건
            
            # 범주형 변수 그룹 (18%)
            'insurance_type': 0.05,       # 보험종목
            'product_category': 0.08,     # 상품분류
            'payment_method': 0.04,       # 결제방법
            'future_outlook': 0.01        # 향후전망
        }
        
        # 캐시 관리
        self.embeddings_cache = {}
        self.similarity_cache = {}
        
        # 런타임 가중치 오버라이드(세션 중 일시 적용)
        self.runtime_weight_overrides = None
    
    @st.cache_resource
    def load_kosimcse_model(_self):
        """KoSimCSE 모델 로드"""
        try:
            # Lazy import to avoid hard dependency at module import time
            from transformers import AutoModel, AutoTokenizer
            model_name = "BM-K/KoSimCSE-roberta-multitask"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            model.eval()
            return model, tokenizer
        except Exception as e:
            st.error(f"AI 모델 로드 실패: {e}")
            return None, None
    
    def initialize_ai_model(self):
        """AI 모델 초기화"""
        if self.kosimcse_model is None:
            with st.spinner("AI 모델을 로드하는 중..."):
                self.kosimcse_model, self.kosimcse_tokenizer = self.load_kosimcse_model()
        return self.kosimcse_model is not None
    
    def preprocess_text(self, text):
        """텍스트 전처리"""
        if pd.isna(text) or text == '':
            return ""
        
        text = str(text).strip()
        
        # 의미없는 텍스트 필터링
        meaningless_patterns = [
            r'^설명없음$', r'^첨부파일참고$', r'^해당없음$', r'^-$',
            r'^없음$', r'^기타$', r'^미상$'
        ]
        
        for pattern in meaningless_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return ""
        
        return text
    
    def get_text_embeddings(self, texts, batch_size=4):
        """텍스트 임베딩 생성"""
        if not self.initialize_ai_model():
            return None
        
        processed_texts = [self.preprocess_text(text) for text in texts]
        valid_texts = [text for text in processed_texts if text]
        
        if not valid_texts:
            return None
        
        embeddings = []
        
        try:
            # Lazy import torch to avoid ImportError if not installed
            import torch
            for i in range(0, len(valid_texts), batch_size):
                batch_texts = valid_texts[i:i + batch_size]
                
                inputs = self.kosimcse_tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt"
                )
                
                with torch.no_grad():
                    outputs = self.kosimcse_model(**inputs)
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    embeddings.extend(batch_embeddings)
            
            return np.array(embeddings)
            
        except Exception as e:
            st.error(f"텍스트 분석 오류: {e}")
            return None
    
    def smart_candidate_filtering(self, query_case, candidates_df, max_candidates=150):
        """스마트 후보 필터링"""
        query_country = query_case.get('수입국', '')
        
        # 1. 쿼리 국가가 개별 유지 국가인 경우
        if self.country_processor.is_individual_country(query_country):
            # 같은 국가 우선
            same_country = candidates_df[candidates_df['수입국'] == query_country]
            if len(same_country) >= 20:
                return same_country.sample(n=min(max_candidates, len(same_country)), random_state=42)
            
            # 같은 지역으로 확장
            query_region = self.country_processor.get_country_region(query_country)
            same_region = candidates_df[candidates_df['수입국'].apply(
                lambda x: self.country_processor.get_country_region(x) == query_region
            )]
            if len(same_region) >= 50:
                return same_region.sample(n=min(max_candidates, len(same_region)), random_state=42)
        
        # 2. 쿼리 국가가 소규모 국가인 경우
        elif self.country_processor.is_minor_country(query_country):
            query_region = self.country_processor.get_country_region(query_country)
            
            # 같은 지역의 소규모 국가들 우선
            same_region_minor = candidates_df[candidates_df['수입국'].apply(
                lambda x: (self.country_processor.get_country_region(x) == query_region and 
                          self.country_processor.is_minor_country(x))
            )]
            
            if len(same_region_minor) >= 10:
                return same_region_minor.sample(n=min(max_candidates//2, len(same_region_minor)), random_state=42)
            
            # 같은 지역 전체로 확장
            same_region_all = candidates_df[candidates_df['수입국'].apply(
                lambda x: self.country_processor.get_country_region(x) == query_region
            )]
            
            if len(same_region_all) >= 20:
                return same_region_all.sample(n=min(max_candidates, len(same_region_all)), random_state=42)
        
        # 3. 전체 검색 (무작위 샘플링)
        return candidates_df.sample(n=min(max_candidates, len(candidates_df)), random_state=42)
    
    def calculate_similarity_scores(self, query_case, candidates_df):
        """개선된 유사도 점수 계산"""
        
        # 0) 데이터 누출 방지: 동일 케이스/중복 텍스트 후보 제거
        safe_candidates = candidates_df.copy()
        try:
            # 동일 사고번호/보상파일번호 배제
            for key in ['사고번호', '보상파일번호']:
                if key in safe_candidates.columns and key in query_case:
                    safe_candidates = safe_candidates[safe_candidates[key] != query_case.get(key)]
            # 동일 텍스트(전처리 후) 완전 일치 + 주요 메타 동일시 배제
            q_text_norm = self.preprocess_text(query_case.get('사고설명', ''))
            if q_text_norm:
                def _norm_text(x):
                    return self.preprocess_text(x)
                txt_eq = safe_candidates['사고설명'].apply(_norm_text) == q_text_norm
                meta_eq = True
                if '수입국' in safe_candidates.columns and '수입국' in query_case:
                    meta_eq = meta_eq & (safe_candidates['수입국'] == query_case.get('수입국'))
                if '보험종목' in safe_candidates.columns and '보험종목' in query_case:
                    meta_eq = meta_eq & (safe_candidates['보험종목'] == query_case.get('보험종목'))
                dup_mask = txt_eq & meta_eq
                if dup_mask.any():
                    safe_candidates = safe_candidates[~dup_mask]
        except Exception:
            pass

        # 스마트 필터링으로 후보 수 제한
        filtered_candidates = self.smart_candidate_filtering(query_case, safe_candidates)
        
        # 면책 사례와 비면책 사례 분리
        exemption_candidates = filtered_candidates[filtered_candidates['판정구분'] == '면책']
        non_exemption_candidates = filtered_candidates[filtered_candidates['판정구분'] != '면책']
        
        similarities = []
        
        # 텍스트 유사도 계산
        query_text = query_case.get('사고설명', '')
        if query_text and len(query_text) > 10:
            candidate_texts = filtered_candidates['사고설명'].tolist()
            
            # AI 기반 유사도 계산
            all_texts = [query_text] + candidate_texts
            embeddings = self.get_text_embeddings(all_texts)
            
            if embeddings is not None and len(embeddings) > 1:
                query_embedding = embeddings[0:1]
                candidate_embeddings = embeddings[1:]
                text_similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]
                
                # 면책 키워드 기반 유사도 보정 적용
                text_similarities = self._apply_exemption_keyword_boost(
                    query_text, candidate_texts, text_similarities, filtered_candidates
                )
            else:
                # 폴백: 간단한 텍스트 유사도
                text_similarities = []
                query_words = set(query_text.lower().split())
                for candidate_text in candidate_texts:
                    if pd.notna(candidate_text):
                        candidate_words = set(str(candidate_text).lower().split())
                        if query_words and candidate_words:
                            jaccard_sim = len(query_words.intersection(candidate_words)) / len(query_words.union(candidate_words))
                            text_similarities.append(jaccard_sim)
                        else:
                            text_similarities.append(0.0)
                    else:
                        text_similarities.append(0.0)
                text_similarities = np.array(text_similarities)
                
                # 면책 키워드 기반 유사도 보정 적용 (폴백에도)
                text_similarities = self._apply_exemption_keyword_boost(
                    query_text, candidate_texts, text_similarities, filtered_candidates
                )
        else:
            text_similarities = np.zeros(len(filtered_candidates))
        
        # 통합 유사도 계산
        for i, (idx, candidate) in enumerate(filtered_candidates.iterrows()):
            score = 0.0
            # 런타임 오버라이드가 있으면 우선 적용
            weights = self.runtime_weight_overrides if self.runtime_weight_overrides else self.optimal_weights
            
            # 1. 텍스트 유사도
            if i < len(text_similarities):
                score += weights['text_similarity'] * text_similarities[i]
            
            # 2. 사고유형 유사도
            if query_case.get('사고유형명') == candidate.get('사고유형명'):
                score += weights['accident_type']
            elif self._group_accident_type(query_case.get('사고유형명')) == self._group_accident_type(candidate.get('사고유형명')):
                score += weights['accident_type'] * 0.7
            
            # 3. 국가 유사도 (개선됨)
            country_sim = self.country_processor.calculate_country_similarity(
                query_case.get('수입국'), candidate.get('수입국')
            )
            score += weights['country_similarity'] * country_sim
            
            # 4. 금액대 유사도
            query_amount = query_case.get('원화사고금액', 0)
            candidate_amount = candidate.get('원화사고금액', 0)
            
            if query_amount > 0 and candidate_amount > 0:
                amount_ratio = min(query_amount, candidate_amount) / max(query_amount, candidate_amount)
                score += weights['amount_similarity'] * amount_ratio
            
            # 5. 보험종목 일치
            if query_case.get('보험종목') == candidate.get('보험종목'):
                score += weights['insurance_type']
            
            # 6. 상품분류 유사도 (그룹명 + 상세분류명)
            if 'product_category' in weights:
                query_product = query_case.get('상품분류명', '')
                candidate_product = candidate.get('상품분류명', '')
                query_group = query_case.get('상품분류그룹명', '')
                candidate_group = candidate.get('상품분류그룹명', '')
                
                # 완전 일치 (상세분류명)
                if query_product == candidate_product:
                    score += weights['product_category']
                # 그룹명 일치
                elif query_group == candidate_group:
                    score += weights['product_category'] * 0.8
                # 그룹화된 상품분류 일치
                elif self._group_product_category(query_product) == self._group_product_category(candidate_product):
                    score += weights['product_category'] * 0.6
            
            # 7. 부보율 유사도
            if 'coverage_rate' in weights:
                query_coverage = query_case.get('부보율', 0)
                candidate_coverage = candidate.get('부보율', 0)
                
                coverage_sim = self._calculate_coverage_similarity(query_coverage, candidate_coverage)
                score += weights['coverage_rate'] * coverage_sim
            
            # 8. 결제방법 유사도
            if 'payment_method' in weights:
                query_payment = query_case.get('결제방법', '')
                candidate_payment = candidate.get('결제방법', '')
                
                if query_payment == candidate_payment:
                    score += weights['payment_method']
                elif self._group_payment_method(query_payment) == self._group_payment_method(candidate_payment):
                    score += weights['payment_method'] * 0.8
            
            # 9. 결제조건 유사도
            if 'payment_terms' in weights:
                query_terms = query_case.get('결제조건', '')
                candidate_terms = candidate.get('결제조건', '')
                
                terms_sim = self._calculate_payment_terms_similarity(query_terms, candidate_terms)
                score += weights['payment_terms'] * terms_sim
            
            # 10. 향후결제전망 유사도
            if 'future_outlook' in weights:
                query_outlook = query_case.get('향후결제전망', '')
                candidate_outlook = candidate.get('향후결제전망', '')
                
                outlook_sim = self._calculate_future_outlook_similarity(query_outlook, candidate_outlook)
                score += weights['future_outlook'] * outlook_sim
            
            similarities.append((
                score, 
                text_similarities[i] if i < len(text_similarities) else 0.0, 
                country_sim,
                candidate
            ))
        
        # 면책 사례와 비면책 사례 분리
        exemption_similarities = [(score, text_sim, country_sim, candidate) 
                                for score, text_sim, country_sim, candidate in similarities 
                                if candidate['판정구분'] == '면책']
        
        non_exemption_similarities = [(score, text_sim, country_sim, candidate) 
                                    for score, text_sim, country_sim, candidate in similarities 
                                    if candidate['판정구분'] != '면책']
        
        # 각각 정렬
        exemption_similarities.sort(key=lambda x: x[0], reverse=True)
        non_exemption_similarities.sort(key=lambda x: x[0], reverse=True)
        
        # 강제 면책 포함 결과 구성
        final_results = []
        
        # 1. 가장 유사한 면책 사례 1건 (있다면 무조건 포함)
        if exemption_similarities:
            final_results.append(exemption_similarities[0])
            print(f"🛡️ 면책 경고: 가장 유사한 면책 사례 포함 (유사도: {exemption_similarities[0][0]:.3f})")
        
        # 2. 나머지는 전체에서 상위 순으로 (면책 제외하고)
        remaining_slots = max(0, len(similarities) - len(final_results))
        if remaining_slots > 0:
            final_results.extend(non_exemption_similarities[:remaining_slots])
        
        return final_results
    
    def _group_accident_type(self, accident_type):
        """사고유형 그룹화"""
        if pd.isna(accident_type):
            return '기타'
        
        if '신용위험' in accident_type:
            if '지급지체' in accident_type:
                return '신용위험-지급지체'
            elif '파산' in accident_type:
                return '신용위험-파산'
            elif '지급불능' in accident_type or '지급거절' in accident_type:
                return '신용위험-지급불능/거절'
            else:
                return '신용위험-기타'
        elif '비상위험' in accident_type:
            return '비상위험'
        elif '검역위험' in accident_type:
            return '검역위험'
        else:
            return '기타위험'
    
    def _group_product_category(self, product_name):
        """상품분류 그룹화"""
        if pd.isna(product_name):
            return '기타'
        
        product_name = str(product_name).lower()
        
        # 의류 및 직물류
        if any(word in product_name for word in ['의류', '직물', '섬유', '패션', 'textile', 'clothing']):
            return '의류_직물류'
        # 전자제품
        elif any(word in product_name for word in ['전자', '반도체', '컴퓨터', 'electronics', 'semiconductor']):
            return '전자제품'
        # 농수산물
        elif any(word in product_name for word in ['농산물', '수산물', '식품', 'agriculture', 'food']):
            return '농수산물'
        # 자동차 및 부품
        elif any(word in product_name for word in ['자동차', '부품', 'auto', 'parts']):
            return '자동차_부품'
        # 화학제품
        elif any(word in product_name for word in ['화학', '플라스틱', 'chemical', 'plastic']):
            return '화학제품'
        else:
            return '기타제품'
    
    def _group_payment_method(self, payment_method):
        """결제방법 그룹화"""
        if pd.isna(payment_method):
            return '기타'
        
        payment_method = str(payment_method).upper()
        
        # 신용도별 그룹화
        if any(method in payment_method for method in ['L/C', 'LC', '신용장']):
            return 'L/C'
        elif any(method in payment_method for method in ['D/P', 'DP', '도착지지급']):
            return 'D/P'
        elif any(method in payment_method for method in ['D/A', 'DA', '도착지인수']):
            return 'D/A'
        elif any(method in payment_method for method in ['NET', 'OPEN', '무신용장']):
            return 'NET'
        else:
            return '기타결제'
    
    def _calculate_coverage_similarity(self, query_rate, candidate_rate):
        """부보율 유사도 계산"""
        if pd.isna(query_rate) or pd.isna(candidate_rate):
            return 0.5  # 중간값
        
        # 비율 차이 계산
        rate_diff = abs(query_rate - candidate_rate) / 100
        
        # 차이가 클수록 낮은 유사도
        return max(0.1, 1.0 - rate_diff)
    
    def _calculate_payment_terms_similarity(self, query_terms, candidate_terms):
        """결제조건 유사도 계산"""
        if pd.isna(query_terms) or pd.isna(candidate_terms):
            return 0.5
        
        query_terms = str(query_terms).lower()
        candidate_terms = str(candidate_terms).lower()
        
        # 완전 일치
        if query_terms == candidate_terms:
            return 1.0
        
        # 조건 유형별 그룹화
        if 'days' in query_terms and 'days' in candidate_terms:
            return 0.8
        elif 'sight' in query_terms and 'sight' in candidate_terms:
            return 0.8
        elif 'invoice' in query_terms and 'invoice' in candidate_terms:
            return 0.7
        
        return 0.3
    
    def _calculate_future_outlook_similarity(self, query_outlook, candidate_outlook):
        """향후결제전망 유사도 계산"""
        if pd.isna(query_outlook) or pd.isna(candidate_outlook):
            return 0.5
        
        query_outlook = str(query_outlook).lower()
        candidate_outlook = str(candidate_outlook).lower()
        
        # 완전 일치
        if query_outlook == candidate_outlook:
            return 1.0
        
        # 긍정적 vs 부정적 그룹화
        positive_terms = ['지급예정', '회수예정', '긍정적', 'positive']
        negative_terms = ['회수불가', '지급불가', '부정적', 'negative']
        unknown_terms = ['판단불가', '미상', 'unknown']
        
        query_group = None
        candidate_group = None
        
        if any(term in query_outlook for term in positive_terms):
            query_group = 'positive'
        elif any(term in query_outlook for term in negative_terms):
            query_group = 'negative'
        elif any(term in query_outlook for term in unknown_terms):
            query_group = 'unknown'
        
        if any(term in candidate_outlook for term in positive_terms):
            candidate_group = 'positive'
        elif any(term in candidate_outlook for term in negative_terms):
            candidate_group = 'negative'
        elif any(term in candidate_outlook for term in unknown_terms):
            candidate_group = 'unknown'
        
        if query_group == candidate_group:
            return 0.8
        elif query_group == 'unknown' or candidate_group == 'unknown':
            return 0.5
        else:
            return 0.2
    
    def _apply_exemption_keyword_boost(self, query_text, candidate_texts, similarities, candidates_df):
        """면책 관련 키워드 기반 유사도 보정 (개선된 버전)"""
        
        # 더 구체적인 면책 관련 핵심 패턴 (맥락을 고려)
        exemption_patterns = {
            '고의과실': [
                ('R급.*기등록', 0.15),  # 정규식 패턴과 가중치
                ('신용조회.*소홀', 0.12),
                ('고의.*위반', 0.10),
                ('과실.*발생', 0.08)
            ],
            '연속수출': [
                ('이전.*미수금.*회수', 0.20),
                ('연속.*수출.*위반', 0.18),
                ('경합수출자.*존재', 0.10),
                ('동일.*수입자.*거래', 0.08)
            ],
            '보상한도초과': [
                ('보상한도.*초과', 0.25),
                ('한도.*초과', 0.20),
                ('책임한도.*부족', 0.15)
            ],
            '보험관계성립': [
                ('허위.*서류.*제출', 0.20),
                ('보험관계.*성립.*불가', 0.18),
                ('신청.*정보.*허위', 0.15)
            ],
            '주의의무해태': [
                ('파산신청.*상태', 0.25),
                ('재무악화.*징후.*무시', 0.20),
                ('주의의무.*해태', 0.18),
                ('적절한.*조치.*부재', 0.12)
            ]
        }
        
        boosted_similarities = similarities.copy()
        
        for i, (candidate_text, candidate_row) in enumerate(zip(candidate_texts, candidates_df.itertuples())):
            # 실제 판정구분이 면책인 경우만 처리
            if hasattr(candidate_row, '판정구분') and candidate_row.판정구분 == '면책':
                
                total_boost = 0.0  # 누적 부스트 대신 총합으로 계산
                matched_patterns = []
                
                # 각 패턴별로 매칭 확인
                for pattern_type, pattern_list in exemption_patterns.items():
                    for pattern, weight in pattern_list:
                        # 쿼리와 후보 모두에서 패턴 검색
                        import re
                        query_match = re.search(pattern, query_text, re.IGNORECASE)
                        candidate_match = re.search(pattern, str(candidate_text), re.IGNORECASE)
                        
                        if query_match and candidate_match:
                            # 판정사유도 패턴과 일치하는지 확인
                            candidate_reason = getattr(candidate_row, '판정사유', '')
                            if pattern_type in ['고의과실'] and '고의' in candidate_reason:
                                total_boost += weight
                                matched_patterns.append(f"{pattern_type}:{pattern}")
                            elif pattern_type in ['연속수출'] and '연속' in candidate_reason:
                                total_boost += weight
                                matched_patterns.append(f"{pattern_type}:{pattern}")
                            elif pattern_type in ['보상한도초과'] and '초과' in candidate_reason:
                                total_boost += weight
                                matched_patterns.append(f"{pattern_type}:{pattern}")
                            elif pattern_type in ['보험관계성립'] and '성립' in candidate_reason:
                                total_boost += weight
                                matched_patterns.append(f"{pattern_type}:{pattern}")
                            elif pattern_type in ['주의의무해태'] and '해태' in candidate_reason:
                                total_boost += weight
                                matched_patterns.append(f"{pattern_type}:{pattern}")
                
                # 최종 부스트 적용 (최대 10% 부스트로 제한)
                if total_boost > 0:
                    final_boost = min(0.10, total_boost)  # 최대 10% 부스트
                    boosted_similarities[i] = min(1.0, similarities[i] + final_boost)  # 곱하기 대신 더하기
                        
        return boosted_similarities
    
    def _analyze_judgment_reasons(self, decisions, reasons, similarity_scores):
        """판정사유 분석"""
        
        # 판정구분별 사유 분석
        decision_reasons = {}
        for decision, reason, sim_score in zip(decisions, reasons, similarity_scores):
            if decision not in decision_reasons:
                decision_reasons[decision] = []
            decision_reasons[decision].append({
                'reason': reason,
                'similarity': sim_score
            })
        
        # 예상 판정사유 도출
        predicted_reasons = {}
        
        for decision, reason_list in decision_reasons.items():
            # 유사도 기반 가중치 적용
            reason_weights = {}
            for item in reason_list:
                reason = item['reason']
                weight = item['similarity']
                
                if reason in reason_weights:
                    reason_weights[reason] += weight
                else:
                    reason_weights[reason] = weight
            
            # 가중치 순으로 정렬
            sorted_reasons = sorted(reason_weights.items(), key=lambda x: x[1], reverse=True)
            predicted_reasons[decision] = sorted_reasons
        
        return {
            'decision_reasons': decision_reasons,
            'predicted_reasons': predicted_reasons,
            'top_reasons': self._get_top_reasons_by_decision(predicted_reasons)
        }
    
    def _get_top_reasons_by_decision(self, predicted_reasons):
        """판정구분별 상위 사유 요약"""
        top_reasons = {}
        
        for decision, reasons in predicted_reasons.items():
            if reasons:
                # 상위 3개 사유
                top_3 = reasons[:3]
                total_weight = sum([weight for _, weight in reasons])
                
                top_reasons[decision] = {
                    'reasons': [(reason, weight/total_weight) for reason, weight in top_3],
                    'total_cases': len(reasons)
                }
        
        return top_reasons

@st.cache_data
def load_data():
    """데이터 로드"""
    try:
        df = pd.read_csv('data/design.csv', encoding='cp949')
        
        # 날짜 컬럼 처리
        date_columns = ['판정일', '판정결재일', '사고접수일자', '보험금청구일']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # 금액 컬럼을 숫자로 변환
        amount_columns = ['원화사고금액', '원화판정금액', ]
        for col in amount_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"데이터 로드 오류: {e}")
        return None

def create_country_analysis_tab(df, country_processor):
    """국가 분석 탭"""
    st.subheader("🌍 국가 처리 분석")
    
    # 국가 전처리 적용
    df_processed = df.copy()
    df_processed['수입국_processed'] = df_processed['수입국'].apply(country_processor.preprocess_country)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**📊 전처리 전후 비교**")
        
        before_count = df['수입국'].nunique()
        after_count = df_processed['수입국_processed'].nunique()
        
        st.metric("전처리 전 국가 수", f"{before_count}개")
        st.metric("전처리 후 카테고리 수", f"{after_count}개", f"-{before_count-after_count}개")
        
        # 전처리 후 분포
        processed_counts = df_processed['수입국_processed'].value_counts().head(15)
        fig1 = px.bar(
            x=processed_counts.values,
            y=processed_counts.index,
            orientation='h',
            title="전처리 후 국가 카테고리 분포",
            color_discrete_sequence=['#007bff']
        )
        fig1.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.write("**🔍 소규모 국가 상세 내역**")
        
        # 기타 국가에 포함된 국가들 표시
        minor_countries = []
        for region, countries in country_processor.regions.items():
            minor_countries.extend(countries['기타'])
        
        st.write("**기타국가에 포함된 39개국:**")
        for i, country in enumerate(minor_countries, 1):
            if country in df['수입국'].values:
                count = (df['수입국'] == country).sum()
                st.write(f"{i:2d}. {country} ({count}건)")
        
        # 지역별 그룹화 결과
        st.write("**지역별 그룹화 결과:**")
        for region in country_processor.regions.keys():
            region_key = f"{region}_기타"
            if region_key in df_processed['수입국_processed'].values:
                count = (df_processed['수입국_processed'] == region_key).sum()
                st.write(f"• {region_key}: {count}건")

def create_similarity_search_interface(system, df):
    """개선된 유사사례 검색 인터페이스"""
    st.subheader("🔍 유사사례 검색")
    
    # AI 모델 상태
    model_ready = system.initialize_ai_model()
    if model_ready:
        st.success("✅ AI 텍스트 분석 모델 준비 완료")
    else:
        st.warning("⚠️ AI 모델 사용 불가 - 기본 검색 모드")
    
    
    # 가중치 조정 (form 밖에 배치)
    with st.expander("🔧 가중치 조정 (고급 설정)", expanded=False):
        st.write("**현재 가중치 설정:**")
        
        # 텍스트 유사도
        text_weight = st.slider(
            "텍스트 유사도 가중치:",
            min_value=0.0,
            max_value=1.0,
            value=system.optimal_weights['text_similarity'],
            step=0.05,
            help="사고설명 텍스트의 중요도"
        )
        
        # 사고유형
        accident_weight = st.slider(
            "사고유형 가중치:",
            min_value=0.0,
            max_value=1.0,
            value=system.optimal_weights['accident_type'],
            step=0.05,
            help="사고유형의 중요도"
        )
        
        # 국가
        country_weight = st.slider(
            "국가 유사도 가중치:",
            min_value=0.0,
            max_value=1.0,
            value=system.optimal_weights['country_similarity'],
            step=0.05,
            help="수입국의 중요도"
        )
        
        # 금액
        amount_weight = st.slider(
            "금액 유사도 가중치:",
            min_value=0.0,
            max_value=1.0,
            value=system.optimal_weights['amount_similarity'],
            step=0.05,
            help="사고금액의 중요도"
        )
        
        # 보험종목
        insurance_weight = st.slider(
            "보험종목 가중치:",
            min_value=0.0,
            max_value=1.0,
            value=system.optimal_weights['insurance_type'],
            step=0.05,
            help="보험종목의 중요도"
        )
        
        # 선택 비활성 체크박스: 특정 피처 영향 제외
        st.write("**피처 사용 여부(선택 안함 가능):**")
        use_text = st.checkbox("텍스트 사용", value=True)
        use_accident = st.checkbox("사고유형 사용", value=True)
        use_country = st.checkbox("국가 사용", value=True)
        use_amount = st.checkbox("금액 사용", value=True)
        use_insurance = st.checkbox("보험종목 사용", value=True)

        # 사용 안함이면 해당 가중치를 0으로 간주
        text_weight = text_weight if use_text else 0.0
        accident_weight = accident_weight if use_accident else 0.0
        country_weight = country_weight if use_country else 0.0
        amount_weight = amount_weight if use_amount else 0.0
        insurance_weight = insurance_weight if use_insurance else 0.0

        # 가중치 합계 확인(정규화는 하지 않음: 절대 가중으로 처리)
        total_weight = text_weight + accident_weight + country_weight + amount_weight + insurance_weight
        st.info(f"가중치 합계: {total_weight:.2f} (절대 가중)")
        
        # 가중치 적용 버튼
        if st.button("가중치 적용"):
            # 런타임 오버라이드 반영(세션 동안만 적용)
            system.runtime_weight_overrides = {
                'text_similarity': text_weight,
                'accident_type': accident_weight,
                'country_similarity': country_weight,
                'amount_similarity': amount_weight,
                'insurance_type': insurance_weight,
                # 나머지 항목은 원래 최적값 유지(오버라이드에서 제공 안 함)
                'product_category': system.optimal_weights.get('product_category', 0.0),
                'coverage_rate': system.optimal_weights.get('coverage_rate', 0.0),
                'payment_method': system.optimal_weights.get('payment_method', 0.0),
                'payment_terms': system.optimal_weights.get('payment_terms', 0.0),
                'future_outlook': system.optimal_weights.get('future_outlook', 0.0),
            }
            st.success("가중치(사용 안함 포함)가 적용되었습니다!")
    
    # 검색 폼
    with st.form("improved_search_form"):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("**📋 사고 정보 입력**")
            
            input_country = st.selectbox(
                "수입국:",
                options=df['수입국'].value_counts().head(30).index,
                help="사고가 발생한 수입국을 선택하세요"
            )
            
            input_insurance = st.selectbox(
                "보험종목:",
                options=df['보험종목'].value_counts().head(10).index
            )
            
            input_accident_type = st.selectbox(
                "사고유형:",
                options=df['사고유형명'].value_counts().head(10).index
            )
            
            input_amount = st.number_input(
                "사고금액 (원):",
                min_value=0,
                value=50000000,
                step=1000000,
                format="%d"
            )
            
            input_coverage = st.slider(
                "부보율 (%):",
                min_value=0,
                max_value=100,
                value=95,
                step=5,
                help="보험 가입 비율을 선택하세요"
            )
            
            input_description = st.text_area(
                "사고설명:",
                placeholder="사고의 구체적인 상황을 입력하세요...",
                height=120
            )
            
            # 추가 입력 필드들
            # 상품분류 선택
            st.write("**📦 상품분류 선택**")
            
            # 상품분류그룹명 옵션 (실제 데이터 기반 - 상위 10개)
            product_group_options = [
                '의류 및 직물류', '무기 및 유기화학제품', '전기 및 전자제품', '기계류',
                '금속, 비금속류', '고무, 가죽', '운송장비 및 부품', '목재와 펄프, 지물류',
                '농수산물, 식료품', '정밀기기, 시계, 악기, 무기류', '기타'
            ]
            input_product_group = st.selectbox(
                "상품분류그룹명:",
                options=product_group_options,
                help="상품의 대분류를 선택하세요"
            )
            
            # 상품분류명 옵션 (실제 데이터 기반 - 상위 20개)
            product_options = [
                '강력사(폴리에스테르의 것)', '외부표면이 플라스틱쉬트 또는 방직용 섬유제의 것', '인조섬유제의 것',
                '인쇄회로', '4. 폴리카보네이트', '광전지(태양전지, 포토다이오드, 포토커플 및 포토릴레이를 포함한다)',
                '신차', '중고차', '의류 및 직물류', '전자제품', '농수산물', '자동차부품', '화학제품',
                '기타제품', '기타'
            ]
            input_product = st.selectbox(
                "상품분류명 (상세):",
                options=product_options,
                help="수출 상품의 상세 분류를 선택하세요"
            )
            
            # 결제방법 옵션 (실제 데이터 기반)
            payment_method_options = [
                'O/A(T/T 포함)', 'D/A', 'NET', 'CAD', 'L/C Usance', 'COD', 'D/P', 'L/C',
                '신용카드', '기성고방식', '신용장 문면상 Tenor', '선급금지급일', '기타'
            ]
            input_payment_method = st.selectbox(
                "결제방법:",
                options=payment_method_options,
                help="거래 결제 방법을 선택하세요"
            )
            
            # 결제조건 옵션 (실제 데이터 기반)
            payment_terms_options = [
                'Days From B/L Date', 'Days After B/L Date', 'Days After Invoice Date',
                'Days From Invoice Date', 'Days After Sight', '월말 마감후', 'Days After Arrival',
                'After Delivery Date(Net거래로 수입지인도일 기준 보험기산)', 'Days From Nego Date',
                'After Finding Docs', 'At Sight', 'Days After Nego Date', 'Days From Arrival',
                'After Delivery Date(국내수출일 기준 보험기산)', 'At', 'On Arrival Of Goods',
                '매월 15일자 마감후', '기타'
            ]
            input_payment_terms = st.selectbox(
                "결제조건:",
                options=payment_terms_options,
                help="결제 조건을 선택하세요"
            )
            
            # 향후결제전망 옵션 (실제 데이터 기반)
            future_outlook_options = [
                '판단불가', '결제불능', '기타', '전매가능', '일부결제가능', '결제진행중'
            ]
            input_future_outlook = st.selectbox(
                "향후결제전망:",
                options=future_outlook_options,
                help="향후 결제 전망을 선택하세요"
            )
        
        with col2:
            st.write("**🔧 검색 설정**")
            
            max_results = st.slider("최대 결과 수:", 3, 10, 5)
            
            st.write("**📊 국가 처리 정보**")
            
            # 입력 국가의 처리 결과 미리보기
            processed_country = system.country_processor.preprocess_country(input_country)
            is_individual = system.country_processor.is_individual_country(input_country)
            is_minor = system.country_processor.is_minor_country(input_country)
            region = system.country_processor.get_country_region(input_country)
            
            st.write(f"• 입력국가: **{input_country}**")
            st.write(f"• 처리결과: **{processed_country}**")
            st.write(f"• 지역: **{region}**")
            st.write(f"• 유형: **{'개별국가' if is_individual else '소규모국가' if is_minor else '기타'}**")
            
            # 검색 필터 설정
            st.write("**🔍 검색 필터 설정**")
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            
            with filter_col1:
                use_country_filter = st.checkbox("수입국 필터 적용", value=False, help="선택한 수입국과 동일한 사례만 검색")
            with filter_col2:
                use_insurance_filter = st.checkbox("보험종목 필터 적용", value=False, help="선택한 보험종목과 동일한 사례만 검색")
            with filter_col3:
                use_accident_filter = st.checkbox("사고유형 필터 적용", value=False, help="선택한 사고유형과 동일한 사례만 검색")
            
        

            
            st.write("**⚖️ 기본 가중치**")
            st.write("• 텍스트 유사도: 35%")
            st.write("• 사고유형: 20%")
            st.write("• 국가 유사도: 12% ⭐")
            st.write("• 금액유사도: 8%")
            st.write("• 보험종목: 5%")
            st.write("• 상품분류: 8%")
            st.write("• 부보율: 5%")
            st.write("• 결제방법: 4%")
            st.write("• 결제조건: 2%")
            st.write("• 향후전망: 1%")

            
        submitted = st.form_submit_button("🔍 검색 실행", type="primary")
    
    if submitted:
        start_time = time.time()
        
        # 입력 데이터 구성 (확장된 변수 포함)
        case_data = {
            '수입국': input_country,
            '보험종목': input_insurance,
            '사고유형명': input_accident_type,
            '원화사고금액': input_amount,
            '사고설명': input_description,
            '부보율': input_coverage,
            '상품분류명': input_product,
            '상품분류그룹명': input_product_group,
            '결제방법': input_payment_method,
            '결제조건': input_payment_terms,
            '향후결제전망': input_future_outlook
        }
        
        with st.spinner(" 유사사례 검색하는 중..."):
            # 의미있는 설명이 있는 사례 우선
            search_df = df.copy()
            if input_description:
                meaningful_df = search_df[
                    (search_df['사고설명'].notna()) & 
                    (search_df['사고설명'].str.len() > 10) &
                    (~search_df['사고설명'].str.contains('설명없음|첨부파일참고|해당없음', na=False, case=False))
                ]
                if len(meaningful_df) > 50:
                    search_df = meaningful_df
            
            # 필터 적용
            if use_country_filter:
                search_df = search_df[search_df['수입국'] == input_country]
                st.info(f"🔍 수입국 필터 적용: {input_country} ({len(search_df)}건)")
            
            if use_insurance_filter:
                search_df = search_df[search_df['보험종목'] == input_insurance]
                st.info(f"🔍 보험종목 필터 적용: {input_insurance} ({len(search_df)}건)")
            
            if use_accident_filter:
                search_df = search_df[search_df['사고유형명'] == input_accident_type]
                st.info(f"🔍 사고유형 필터 적용: {input_accident_type} ({len(search_df)}건)")
            
            if len(search_df) == 0:
                st.error("❌ 필터 조건에 맞는 사례가 없습니다. 필터를 조정해주세요.")
                return
            
            # 개선된 유사도 계산
            similarities = system.calculate_similarity_scores(case_data, search_df)
            # 상위 5건만, 유사도 0.30 미만 제외
            min_score = 0.30
            top_similar = [r for r in similarities if r[0] >= min_score][:5]
            
            search_time = time.time() - start_time
        
        # 하이브리드 신뢰도 계산
        if top_similar:
            decisions = [case[3]['판정구분'] for case in top_similar]
            reasons = [case[3]['판정사유'] for case in top_similar]
            similarity_scores = [case[0] for case in top_similar]
            
            confidence_result = system.confidence_calculator.hybrid_confidence(decisions, similarity_scores)
            
            # 판정사유 분석
            reason_analysis = system._analyze_judgment_reasons(decisions, reasons, similarity_scores)
            
            # 신뢰도 결과 표시
            pred_decision = confidence_result['predicted_decision']
            confidence = confidence_result['confidence']
            grade = confidence_result['grade']
            
            if pred_decision == '지급':
                box_class = "confidence-box success-box"
            elif pred_decision == '면책':
                box_class = "confidence-box error-box"
            else:
                box_class = "confidence-box warning-box"
            
            st.markdown(f"""
            <div class="{box_class}">
              <!--  <h2>🎯 예상 판정: {pred_decision}</h2>
                <h3>하이브리드 신뢰도: {confidence:.1%} ({grade})</h3>-->
                <p>{confidence_result['interpretation']}</p>
                <small>검색 시간: {search_time:.1f}초 | 유사사례: {confidence_result['sample_size']}개 | 평균 유사도: {confidence_result['avg_similarity']:.1%}</small>
            </div>
            """, unsafe_allow_html=True)
            
            # 신뢰도 상세 분석
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**📊 가중 vs 베이지안 비교**")
                st.metric("가중 신뢰도", f"{confidence_result['weighted_confidence']:.1%}")
                st.metric("베이지안 추정", f"{confidence_result['bayesian_mean']:.1%}")
                
            with col2:
                # 판정구분 분포
                decision_counts = Counter(decisions)
                decision_df = pd.DataFrame(list(decision_counts.items()), columns=['판정구분', '건수'])
                fig_pred = px.pie(
                    decision_df,
                    values='건수',
                    names='판정구분',
                    title="유사사례 판정구분 분포",
                    color_discrete_sequence=['#28a745', '#dc3545', '#ffc107']
                )
                st.plotly_chart(fig_pred, use_container_width=True)
            
            with col3:
                # 신뢰구간 시각화
                lower, upper = confidence_result['credible_interval']
                
                fig_interval = go.Figure()
                fig_interval.add_trace(go.Scatter(
                    x=[confidence_result['bayesian_mean']],
                    y=['베이지안 추정'],
                    mode='markers',
                    marker=dict(size=15, color='blue'),
                    name='추정값'
                ))
                fig_interval.add_shape(
                    type="line",
                    x0=lower, x1=upper,
                    y0=0, y1=0,
                    line=dict(color="red", width=4),
                )
                fig_interval.update_layout(
                    title="95% 신뢰구간",
                    xaxis_title="확률",
                    yaxis=dict(showticklabels=False),
                    height=200
                )
                st.plotly_chart(fig_interval, use_container_width=True)
            
            # 판정사유 분석 추가
            st.subheader("🎯 판정사유 분석")
            
            top_reasons = reason_analysis['top_reasons']
            
            if pred_decision in top_reasons:
                decision_info = top_reasons[pred_decision]
                st.write(f"**{pred_decision} 판정의 주요 사유 (상위 3개):**")
                
                for i, (reason, prob) in enumerate(decision_info['reasons']):
                    emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                    st.write(f"{emoji} **{reason}** ")
                    # ({prob:.1%}) 이건 일단 제외
            else:
                st.info("해당 판정구분의 사유 데이터가 부족합니다.")
            
            # 모든 판정구분별 사유 요약
            if len(top_reasons) > 1:
                st.write("**📊 판정구분별 주요 사유 비교**")
                
                reason_comparison = []
                for decision, info in top_reasons.items():
                    for reason, prob in info['reasons'][:2]:  # 상위 2개만
                        reason_comparison.append({
                            '판정구분': decision,
                            '판정사유': reason,
                            '가중확률': prob,
                            '사례수': info['total_cases']
                        })
                
                if reason_comparison:
                    reason_df = pd.DataFrame(reason_comparison)
                    
                    # 판정사유별 막대 차트
                    fig_reasons = px.bar(
                        reason_df,
                        x='판정사유',
                        y='가중확률',
                        color='판정구분',
                        title="판정구분별 주요 사유",
                        color_discrete_sequence=['#28a745', '#dc3545', '#ffc107']
                    )
                    fig_reasons.update_layout(
                        xaxis_title="판정사유",
                        xaxis_tickangle=45,
                        yaxis_title="가중확률",
                        yaxis_tickformat='.1%'
                    )
                    st.plotly_chart(fig_reasons, use_container_width=True)
            
            # 상세 결과
            st.subheader("📋 유사사례 상세 분석")
            
            # 면책 사례 포함 여부 확인 및 경고 표시
            exemption_cases = [case for _, _, _, case in top_similar if case['판정구분'] == '면책']
            if exemption_cases:
                st.warning(f"🛡️ **면책 경고**: {len(exemption_cases)}건의 면책 사례가 포함되어 있습니다. 반드시 확인하세요!")
                
                # 면책 유사도 기반 분석
                st.subheader("🛡️ 면책 유사도 분석")
                exemption_scores = [(score, case) for score, _, _, case in top_similar if case['판정구분'] == '면책']
                
                if exemption_scores:
                    avg_exemption_score = sum(score for score, _ in exemption_scores) / len(exemption_scores)
                    max_exemption_score = max(score for score, _ in exemption_scores)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("면책 사례 수", len(exemption_scores))
                    with col2:
                        st.metric("평균 면책 유사도", f"{avg_exemption_score:.3f}")
                    with col3:
                        st.metric("최고 면책 유사도", f"{max_exemption_score:.3f}")
                    
                    # 면책 위험도 평가
                    if max_exemption_score > 0.8:
                        st.error("🚨 **높은 면책 위험**: 매우 유사한 면책 사례가 존재합니다!")
                    elif max_exemption_score > 0.6:
                        st.warning("⚠️ **중간 면책 위험**: 유사한 면책 사례가 있습니다.")
                    else:
                        st.info("ℹ️ **낮은 면책 위험**: 면책 사례와의 유사도가 낮습니다.")
            
            for i, (total_score, text_sim, country_sim, similar_case) in enumerate(top_similar):
                # 면책 사례는 특별 표시
                if similar_case['판정구분'] == '면책':
                    expander_title = f"🛡️ #{i+1} 종합유사도 {total_score*100:.1f}% - ⚠️ **{similar_case['판정구분']}** ({similar_case['사고유형명']}) ⚠️"
                    expanded = True  # 면책은 기본 펼쳐짐
                else:
                    expander_title = f"#{i+1} 종합유사도 {total_score*100:.1f}% - {similar_case['판정구분']} ({similar_case['사고유형명']})"
                    expanded = False
                
                with st.expander(expander_title, expanded=expanded):
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**📋 사례 정보**")
                        st.write(f"• 보상파일번호: `{similar_case['보상파일번호']}`")
                        st.write(f"• 사고번호: `{similar_case['사고번호']}`")
                        st.write(f"• 수입국: **{similar_case['수입국']}**")
                        st.write(f"• 보험종목: {similar_case['보험종목']}")
                        
                        if pd.notna(similar_case['원화사고금액']):
                            amount_str = f"{similar_case['원화사고금액']:,.0f}원"
                            if similar_case['원화사고금액'] >= 100000000:
                                amount_str += f" ({similar_case['원화사고금액']/100000000:.1f}억원)"
                            st.write(f"• 사고금액: **{amount_str}**")
                    
                    with col2:
                        st.write("**⚖️ 판정 정보**")
                        
                        if similar_case['판정구분'] == '지급':
                            st.success(f"판정구분: {similar_case['판정구분']}")
                        elif similar_case['판정구분'] == '면책':
                            st.error(f"판정구분: {similar_case['판정구분']}")
                        else:
                            st.warning(f"판정구분: {similar_case['판정구분']}")
                        
                        st.write(f"• 판정사유: **{similar_case['판정사유']}**")
                        st.write(f"• 판정회차: {similar_case['판정회차']}회")
                        st.write(f"• 사고진행상태: {similar_case['사고진행상태']}")
                    
                    # 유사도 상세 분석
                    st.write("**📊 유사도 세부 분석**")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("텍스트 유사도", f"{text_sim*100:.1f}%")
                    with col2:
                        st.metric("국가 유사도", f"{country_sim*100:.1f}%")
                    with col3:
                        st.metric("종합 유사도", f"{total_score*100:.1f}%")
                    
                    # 진행바
                    total_pct = min(total_score * 100, 100)
                    st.progress(float(total_pct) / 100)
                    
                    if pd.notna(similar_case['사고설명']) and len(str(similar_case['사고설명'])) > 10:
                        st.write("**📝 사고설명**")
                        st.markdown(f"> {similar_case['사고설명']}")

                    # 전문가용 핵심 변수 요약
                    st.write("**🔎 핵심 변수**")
                    st.write(f"- 상품분류그룹명: {similar_case.get('상품분류그룹명','-')}")
                    st.write(f"- 결제방법/조건: {similar_case.get('결제방법','-')} / {similar_case.get('결제조건','-')}")
                    st.write(f"- 부보율: {similar_case.get('부보율','-')}")
                    st.write(f"- 향후전망: {similar_case.get('향후결제전망','-')}")
        else:
            st.warning("조건에 맞는 유사사례를 찾을 수 없습니다.")

def create_exemption_reason_tab(df):
    """면책사유별 사례조회 탭"""
    
    st.markdown("""
    <div class="main-header">
        <h2>🛡️ 면책사유별 사례조회</h2>
        <p>특정 면책사유로 실제 면책된 사례들을 탐색하고 분석합니다.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 면책 사례만 필터링
    exemption_df = df[df['판정구분'] == '면책'].copy()
    
    if len(exemption_df) == 0:
        st.warning("면책 사례가 없습니다.")
        return
    
    # 면책사유 통계
    st.subheader("📊 면책사유 통계")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("총 면책 사례", len(exemption_df))
    
    with col2:
        unique_reasons = exemption_df['판정사유'].nunique()
        st.metric("면책사유 종류", unique_reasons)
    
    with col3:
        avg_amount = exemption_df['원화사고금액'].mean()
        if pd.notna(avg_amount):
            st.metric("평균 사고금액", f"{avg_amount:,.0f}원")
        else:
            st.metric("평균 사고금액", "N/A")
    
    # 면책사유별 분포 시각화
    st.subheader("📈 면책사유별 분포")
    
    reason_counts = exemption_df['판정사유'].value_counts().head(15)
    
    fig = px.bar(
        x=reason_counts.values,
        y=reason_counts.index,
        orientation='h',
        title="상위 15개 면책사유별 사례 수",
        labels={'x': '사례 수', 'y': '면책사유'},
        color=reason_counts.values,
        color_continuous_scale='Reds'
    )
    
    fig.update_layout(
        height=500,
        showlegend=False,
        xaxis_title="사례 수",
        yaxis_title="면책사유"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 면책사유 선택 및 사례 조회
    st.subheader("🔍 면책사유별 사례 조회")
    
    # 면책사유 목록 (사례 수와 함께 표시)
    reason_options = []
    for reason, count in reason_counts.items():
        reason_options.append(f"{reason} ({count}건)")
    
    selected_reason_full = st.selectbox(
        "면책사유를 선택하세요:",
        options=reason_options,
        help="사례 수가 많은 순으로 정렬되어 있습니다."
    )
    
    if selected_reason_full:
        # 선택된 면책사유에서 사례 수 제거
        selected_reason = selected_reason_full.split(" (")[0]
        
        # 해당 면책사유의 사례들 필터링
        filtered_cases = exemption_df[exemption_df['판정사유'] == selected_reason].copy()
        
        if len(filtered_cases) > 0:
            st.success(f"✅ **{selected_reason}** 면책사유로 총 **{len(filtered_cases)}건**의 사례를 찾았습니다.")

            # 기본: 면책사유만으로 필터링된 결과를 사용
            display_cases = filtered_cases.copy()

            # 선택: 고급 필터 (기본 비활성화)
            with st.expander("⚙️ 고급 필터 (선택)", expanded=False):
                use_advanced_filters = st.checkbox("고급 필터 사용", value=False)
                if use_advanced_filters:
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        countries = ['전체'] + sorted(display_cases['수입국'].dropna().unique().tolist())
                        selected_country = st.selectbox("수입국 필터:", countries)
                    with col2:
                        insurance_types = ['전체'] + sorted(display_cases['보험종목'].dropna().unique().tolist())
                        selected_insurance = st.selectbox("보험종목 필터:", insurance_types)
                    with col3:
                        accident_types = ['전체'] + sorted(display_cases['사고유형명'].dropna().unique().tolist())
                        selected_accident = st.selectbox("사고유형 필터:", accident_types)

                    if selected_country != '전체':
                        display_cases = display_cases[display_cases['수입국'] == selected_country]
                        st.info(f"🔍 수입국 필터: {selected_country} ({len(display_cases)}건)")
                    if selected_insurance != '전체':
                        display_cases = display_cases[display_cases['보험종목'] == selected_insurance]
                        st.info(f"🔍 보험종목 필터: {selected_insurance} ({len(display_cases)}건)")
                    if selected_accident != '전체':
                        display_cases = display_cases[display_cases['사고유형명'] == selected_accident]
                        st.info(f"🔍 사고유형 필터: {selected_accident} ({len(display_cases)}건)")

            # 키워드 검색(사고설명 내 포함 검색)
            keyword = st.text_input("키워드 검색(사고설명):", value="", help="사고설명에 포함되는 키워드로 간단 검색")
            if keyword:
                mask = display_cases['사고설명'].astype(str).str.contains(keyword, case=False, na=False)
                display_cases = display_cases[mask]
                st.info(f"🔍 키워드 '{keyword}' 결과: {len(display_cases)}건")
            
            if len(display_cases) > 0:
                # 정렬 옵션
                sort_options = {
                    '사고금액 (높은순)': '원화사고금액',
                    '사고금액 (낮은순)': '원화사고금액',
                    '판정회차 (높은순)': '판정회차',
                    '판정회차 (낮은순)': '판정회차',
                    '사고접수일자 (최신순)': '사고접수일자',
                    '사고접수일자 (오래된순)': '사고접수일자'
                }
                
                selected_sort = st.selectbox("정렬 기준:", list(sort_options.keys()))
                
                # 정렬 적용
                if '높은순' in selected_sort or '최신순' in selected_sort:
                    ascending = False
                else:
                    ascending = True
                
                display_cases = display_cases.sort_values(
                    by=sort_options[selected_sort], 
                    ascending=ascending,
                    na_position='last'
                )
                
                # 표시할 컬럼 선택
                st.write("**📋 사례 목록**")
                
                # 표시할 데이터 준비
                display_data = display_cases[[
                    '보상파일번호', '사고번호', '판정회차', '수입국', '보험종목', 
                    '사고유형명', '원화사고금액', '사고설명', '사고진행상태'
                ]].copy()
                
                # 금액 포맷팅
                display_data['사고금액'] = display_data['원화사고금액'].apply(
                    lambda x: f"{x:,.0f}원" if pd.notna(x) else "N/A"
                )
                
                # 사고설명 요약 (50자 제한)
                display_data['사고설명_요약'] = display_data['사고설명'].apply(
                    lambda x: str(x)[:50] + "..." if pd.notna(x) and len(str(x)) > 50 else str(x) if pd.notna(x) else "N/A"
                )
                
                # 최종 표시 컬럼
                final_columns = [
                    '보상파일번호', '사고번호', '판정회차', '수입국', '보험종목',
                    '사고유형명', '사고금액', '사고설명_요약', '사고진행상태'
                ]
                
                # 컬럼명 한글화
                column_mapping = {
                    '보상파일번호': '보상파일번호',
                    '사고번호': '사고번호', 
                    '판정회차': '판정회차',
                    '수입국': '수입국',
                    '보험종목': '보험종목',
                    '사고유형명': '사고유형',
                    '사고금액': '사고금액',
                    '사고설명_요약': '사고설명',
                    '사고진행상태': '진행상태'
                }
                
                display_data = display_data[final_columns].rename(columns=column_mapping)
                
                # 페이지네이션
                items_per_page = 20
                total_items = len(display_data)
                total_pages = (total_items + items_per_page - 1) // items_per_page
                
                if total_pages > 1:
                    current_page = st.selectbox(
                        f"페이지 선택 (총 {total_pages}페이지):",
                        range(1, total_pages + 1)
                    ) - 1
                else:
                    current_page = 0
                
                start_idx = current_page * items_per_page
                end_idx = min(start_idx + items_per_page, total_items)
                
                # 현재 페이지 데이터 표시
                current_data = display_data.iloc[start_idx:end_idx]
                
                st.dataframe(
                    current_data,
                    use_container_width=True,
                    hide_index=True
                )
                
                # 페이지 정보
                if total_pages > 1:
                    st.write(f"📄 {start_idx + 1}~{end_idx} / {total_items}건 (페이지 {current_page + 1}/{total_pages})")
                
                # 유사도 검색 기능 추가
                st.markdown("---")
                st.subheader("🔍 유사도 검색 기능")
                
                # 유사도 검색 방식 선택(기본: 면책사유만 필터된 풀에서 검색)
                search_method = st.radio(
                    "검색 방식을 선택하세요:",
                    [
                        "방식 A: 해당 면책사유 사례들(현재 표시된 목록)에서 유사도 검색",
                        "방식 B: 전체 데이터에서 해당 면책사유와 유사한 사례 검색", 
                        "방식 C: 복합 조건으로 유사도 검색"
                    ],
                    help="A 권장: 과도한 선필터로 0건 방지"
                )
                
                # 검색어 입력
                search_text = st.text_area(
                    "검색할 사고설명을 입력하세요:",
                    placeholder="예: 수출 지연, 지급 거절, 계약 위반 등",
                    height=100,
                    help="사고의 주요 내용을 자유롭게 입력하세요"
                )
                
                # 추가 조건 (방식 C용)
                additional_conditions = {}
                if "방식 C" in search_method:
                    st.write("**추가 검색 조건:**")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        additional_conditions['country'] = st.selectbox(
                            "수입국 (선택사항):",
                            ['전체'] + sorted(df['수입국'].unique().tolist())
                        )
                        additional_conditions['insurance'] = st.selectbox(
                            "보험종목 (선택사항):",
                            ['전체'] + sorted(df['보험종목'].unique().tolist())
                        )
                    
                    with col2:
                        additional_conditions['accident_type'] = st.selectbox(
                            "사고유형 (선택사항):",
                            ['전체'] + sorted(df['사고유형명'].unique().tolist())
                        )
                        additional_conditions['amount_range'] = st.selectbox(
                            "사고금액 범위 (선택사항):",
                            ['전체', '1000만원 이하', '1000만원-5000만원', '5000만원-1억원', '1억원 이상']
                        )
                
                # 유사도 검색 실행
                if st.button("🔍 유사도 검색 실행", type="primary"):
                    if search_text.strip():
                        # 시스템 초기화 확인
                        if 'improved_system' not in st.session_state:
                            st.session_state.improved_system = ImprovedInsuranceSystem()
                        system = st.session_state.improved_system
                        
                        # 검색 방식에 따른 후보 데이터 선택
                        if "방식 A" in search_method:
                            # 현재 표시 목록(면책사유 + 선택적 고급필터 + 키워드)에서 검색
                            candidates_df = display_cases.copy()
                            st.info(f"🔍 방식 A: 현재 표시 {len(candidates_df)}건에서 검색")
                            
                        elif "방식 B" in search_method:
                            # 전체 데이터에서 해당 면책사유와 유사한 사례 검색
                            candidates_df = df.copy()
                            st.info(f"🔍 방식 B: 전체 데이터 {len(candidates_df)}건에서 '{selected_reason}' 관련 사례 검색")
                            
                        else:  # 방식 C
                            # 복합 조건 적용
                            candidates_df = df.copy()
                            
                            # 추가 조건 필터링
                            if additional_conditions['country'] != '전체':
                                candidates_df = candidates_df[candidates_df['수입국'] == additional_conditions['country']]
                            
                            if additional_conditions['insurance'] != '전체':
                                candidates_df = candidates_df[candidates_df['보험종목'] == additional_conditions['insurance']]
                            
                            if additional_conditions['accident_type'] != '전체':
                                candidates_df = candidates_df[candidates_df['사고유형명'] == additional_conditions['accident_type']]
                            
                            if additional_conditions['amount_range'] != '전체':
                                if additional_conditions['amount_range'] == '1000만원 이하':
                                    candidates_df = candidates_df[candidates_df['원화사고금액'] <= 10000000]
                                elif additional_conditions['amount_range'] == '1000만원-5000만원':
                                    candidates_df = candidates_df[(candidates_df['원화사고금액'] > 10000000) & (candidates_df['원화사고금액'] <= 50000000)]
                                elif additional_conditions['amount_range'] == '5000만원-1억원':
                                    candidates_df = candidates_df[(candidates_df['원화사고금액'] > 50000000) & (candidates_df['원화사고금액'] <= 100000000)]
                                elif additional_conditions['amount_range'] == '1억원 이상':
                                    candidates_df = candidates_df[candidates_df['원화사고금액'] > 100000000]
                            
                            st.info(f"🔍 방식 C: 복합 조건 적용 후 {len(candidates_df)}건에서 검색")
                        
                        # 쿼리 케이스 생성
                        query_case = {
                            '사고설명': search_text,
                            '판정사유': selected_reason,
                            '수입국': additional_conditions.get('country', '전체'),
                            '보험종목': additional_conditions.get('insurance', '전체'),
                            '사고유형명': additional_conditions.get('accident_type', '전체')
                        }
                        
                        # 유사도 계산
                        with st.spinner("유사도 계산 중..."):
                            try:
                                similarities = system.calculate_similarity_scores(query_case, candidates_df)

                                # 상위 5건만, 0.30 미만 제외
                                min_score = 0.30
                                top_items = [r for r in similarities if r[0] >= min_score][:5] if similarities else []

                                if top_items:
                                    # 결과 표시
                                    st.success(f"✅ 유사도 검색 완료! 상위 {len(top_items)}개 결과 (임계치 {min_score:.2f} 적용)")

                                    # 결과 테이블 생성
                                    results_data = []
                                    for i, (score, text_sim, country_sim, case) in enumerate(top_items, 1):
                                        results_data.append({
                                            '순위': i,
                                            '유사도(%)': f"{score*100:.1f}%",
                                            '판정구분': case['판정구분'],
                                            '판정사유': case['판정사유'],
                                            '수입국': case['수입국'],
                                            '보험종목': case['보험종목'],
                                            '사고유형': case['사고유형명'],
                                            '사고금액': f"{case['원화사고금액']:,.0f}원" if pd.notna(case['원화사고금액']) else "N/A",
                                            '사고설명': str(case['사고설명'])[:100] + "..." if len(str(case['사고설명'])) > 100 else str(case['사고설명'])
                                        })
                                    
                                    # 결과 테이블 표시
                                    results_df = pd.DataFrame(results_data)
                                    st.dataframe(
                                        results_df,
                                        use_container_width=True,
                                        hide_index=True
                                    )
                                    
                                    # 검색 방식별 통계
                                    st.subheader("📊 검색 결과 분석")
                                    
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        exemption_count = sum(1 for _, _, _, case in top_items if case['판정구분'] == '면책')
                                        st.metric("면책 사례", exemption_count)
                                    
                                    with col2:
                                        avg_similarity = sum(score for score, _, _, _ in top_items) / len(top_items)
                                        st.metric("평균 유사도", f"{avg_similarity*100:.1f}%")
                                    
                                    with col3:
                                        max_similarity = max(score for score, _, _, _ in top_items)
                                        st.metric("최고 유사도", f"{max_similarity*100:.1f}%")
                                    
                                    # 검색 방식별 특징 설명
                                    st.info(f"""
                                    **🔍 {search_method}**
                                    - 검색 대상: {len(candidates_df)}건
                                    - 검색 결과: {len(top_items)}건
                                    - 주요 특징: {'해당 면책사유 사례들 중에서 유사도 기반 정렬' if '방식 A' in search_method else '전체 데이터에서 면책사유 관련성 고려' if '방식 B' in search_method else '복합 조건 + 유사도 검색'}
                                    """)
                                    
                                else:
                                    st.warning("❌ 유사한 사례를 찾을 수 없습니다.")
                                    
                            except Exception as e:
                                st.error(f"❌ 유사도 계산 중 오류가 발생했습니다: {str(e)}")
                    else:
                        st.warning("검색할 사고설명을 입력해주세요.")
                
                # 상세 분석
                st.subheader("📊 선택된 면책사유 상세 분석")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # 수입국별 분포
                    country_dist = display_cases['수입국'].value_counts().head(10)
                    fig_country = px.pie(
                        values=country_dist.values,
                        names=country_dist.index,
                        title=f"수입국별 분포 (상위 10개)"
                    )
                    st.plotly_chart(fig_country, use_container_width=True)
                
                with col2:
                    # 보험종목별 분포
                    insurance_dist = display_cases['보험종목'].value_counts().head(10)
                    fig_insurance = px.pie(
                        values=insurance_dist.values,
                        names=insurance_dist.index,
                        title=f"보험종목별 분포 (상위 10개)"
                    )
                    st.plotly_chart(fig_insurance, use_container_width=True)
                
                # 사고금액 분포
                if display_cases['원화사고금액'].notna().any():
                    fig_amount = px.histogram(
                        display_cases,
                        x='원화사고금액',
                        nbins=20,
                        title=f"사고금액 분포",
                        labels={'원화사고금액': '사고금액 (원)', 'count': '사례 수'}
                    )
                    st.plotly_chart(fig_amount, use_container_width=True)
                
                # 판정회차별 분포
                fig_rounds = px.bar(
                    x=display_cases['판정회차'].value_counts().index,
                    y=display_cases['판정회차'].value_counts().values,
                    title=f"판정회차별 분포",
                    labels={'x': '판정회차', 'y': '사례 수'}
                )
                st.plotly_chart(fig_rounds, use_container_width=True)
                
            else:
                st.warning("선택한 필터 조건에 맞는 사례가 없습니다.")
        else:
            st.warning(f"'{selected_reason}' 면책사유로 면책된 사례가 없습니다.")
    else:
        st.info("면책사유를 선택하면 해당 사유로 면책된 사례들을 확인할 수 있습니다.")

def main():
    """메인 함수"""
    # 데이터 로드
    df = load_data()
    if df is None:
        st.error("데이터를 로드할 수 없습니다.")
        return
    
    # 개선된 시스템 초기화
    if 'improved_system' not in st.session_state:
        st.session_state.improved_system = ImprovedInsuranceSystem()
    
    system = st.session_state.improved_system
    
    # 메인 헤더
    st.markdown("""
    <div class="main-header">
        <h1>📋 국외 청구 심사 유사사례 검색 </h1>
        <p>유사 사고 사례 검색 시스템</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 탭 구성
    tab1, tab2, tab3 = st.tabs([
        "🔍 사례 검색",
        "🛡️ 면책사유별 사례조회",
        "🌍 국가 분석"        
    ])
    
    with tab1:
        create_similarity_search_interface(system, df)
    
    with tab2:
        create_exemption_reason_tab(df)
    
    with tab3:
        create_country_analysis_tab(df, system.country_processor)
        
    
    # 푸터
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6c757d; padding: 1rem;">
        <p>📋 보험사고 분석 시스템</p>
        <small>계층적 국가 처리 + 베이지안 신뢰도 + 스마트 필터링</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()