import streamlit as st
import pandas as pd
import numpy as np
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
from transformers import AutoModel, AutoTokenizer
from collections import Counter
import pickle
import warnings
import re
import time
warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(
    page_title="보험사고 판정 분석 시스템",
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
    
    .prediction-box {
        background-color: #e3f2fd;
        padding: 2rem;
        border-radius: 10px;
        border: 2px solid #2196f3;
        text-align: center;
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
    
    .case-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    
    .similarity-bar {
        background-color: #007bff;
        height: 20px;
        border-radius: 10px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

class PracticalInsuranceSystem:
    def __init__(self):
        self.model = None
        self.kosimcse_model = None
        self.kosimcse_tokenizer = None
        self.text_vectorizer = None
        self.label_encoders = {}
        self.feature_importance = None
        
        # 실험을 통해 최적화된 고정 가중치
        self.optimal_weights = {
            'text_similarity': 0.45,
            'accident_type': 0.25,
            'country': 0.15,
            'amount_similarity': 0.10,
            'insurance_type': 0.05
        }
        
        # 캐시 관리
        self.embeddings_cache = {}
        self.similarity_cache = {}
    
    @st.cache_resource
    def load_kosimcse_model(_self):
        """KoSimCSE 모델 로드 (캐시 적용)"""
        try:
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
        """텍스트 임베딩 생성 (배치 크기 줄여서 속도 개선)"""
        if not self.initialize_ai_model():
            return None
        
        processed_texts = [self.preprocess_text(text) for text in texts]
        valid_texts = [text for text in processed_texts if text]
        
        if not valid_texts:
            return None
        
        embeddings = []
        
        try:
            for i in range(0, len(valid_texts), batch_size):
                batch_texts = valid_texts[i:i + batch_size]
                
                inputs = self.kosimcse_tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=256,  # 길이 제한으로 속도 개선
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
    
    def calculate_similarity_scores(self, query_case, candidates_df, max_candidates=100):
        """유사도 점수 계산 (최적화됨)"""
        # 후보 수 제한으로 속도 개선
        if len(candidates_df) > max_candidates:
            candidates_df = candidates_df.sample(n=max_candidates, random_state=42)
        
        similarities = []
        
        # 텍스트 유사도 계산
        query_text = query_case.get('사고설명', '')
        if query_text and len(query_text) > 10:
            candidate_texts = candidates_df['사고설명'].tolist()
            
            # AI 기반 유사도 계산
            all_texts = [query_text] + candidate_texts
            embeddings = self.get_text_embeddings(all_texts)
            
            if embeddings is not None and len(embeddings) > 1:
                query_embedding = embeddings[0:1]
                candidate_embeddings = embeddings[1:]
                text_similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]
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
        else:
            text_similarities = np.zeros(len(candidates_df))
        
        # 통합 유사도 계산
        for i, (idx, candidate) in enumerate(candidates_df.iterrows()):
            score = 0.0
            weights = self.optimal_weights
            
            # 1. 텍스트 유사도
            if i < len(text_similarities):
                score += weights['text_similarity'] * text_similarities[i]
            
            # 2. 사고유형 유사도
            if query_case.get('사고유형명') == candidate.get('사고유형명'):
                score += weights['accident_type']
            elif self._group_accident_type(query_case.get('사고유형명')) == self._group_accident_type(candidate.get('사고유형명')):
                score += weights['accident_type'] * 0.7
            
            # 3. 수입국 일치
            if query_case.get('수입국') == candidate.get('수입국'):
                score += weights['country']
            
            # 4. 금액대 유사도
            query_amount = query_case.get('원화사고금액', 0)
            candidate_amount = candidate.get('원화사고금액', 0)
            
            if query_amount > 0 and candidate_amount > 0:
                amount_ratio = min(query_amount, candidate_amount) / max(query_amount, candidate_amount)
                score += weights['amount_similarity'] * amount_ratio
            
            # 5. 보험종목 일치
            if query_case.get('보험종목') == candidate.get('보험종목'):
                score += weights['insurance_type']
            
            similarities.append((score, text_similarities[i] if i < len(text_similarities) else 0.0, candidate))
        
        # 정렬 후 반환
        similarities.sort(key=lambda x: x[0], reverse=True)
        return similarities
    
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
    
    def train_prediction_model(self, df):
        """예측 모델 학습 및 성능 평가"""
        try:
            # 특성 전처리
            df_processed = df.copy()
            
            # 금액 구간 생성
            df_processed['금액구간'] = pd.cut(
                df_processed['원화사고금액'].fillna(0),
                bins=[0, 10000000, 50000000, 100000000, 500000000, float('inf')],
                labels=['1천만원미만', '1천만-5천만원', '5천만-1억원', '1억-5억원', '5억원이상']
            )
            
            # 사고유형 그룹화
            df_processed['사고유형그룹'] = df_processed['사고유형명'].apply(self._group_accident_type)
            
            # 범주형 변수 인코딩
            categorical_features = ['수입국', '보험종목', '사고유형그룹', '금액구간', '상품분류그룹명']
            
            for feature in categorical_features:
                le = LabelEncoder()
                df_processed[f'{feature}_encoded'] = le.fit_transform(df_processed[feature].fillna('Unknown'))
                self.label_encoders[feature] = le
            
            # 특성 선택
            feature_columns = [f'{feature}_encoded' for feature in categorical_features]
            feature_columns.extend(['원화사고금액'])
            
            X = df_processed[feature_columns].fillna(0)
            y = df_processed['판정구분']
            
            # 모델 학습
            self.model = RandomForestClassifier(
                n_estimators=50,  # 트리 수 줄여서 속도 개선
                max_depth=8,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1  # 병렬 처리
            )
            
            # 교차 검증
            cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='accuracy')
            
            # 전체 데이터로 학습
            self.model.fit(X, y)
            
            # 특성 중요도
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'feature_importance': self.feature_importance
            }
            
        except Exception as e:
            st.error(f"모델 학습 오류: {e}")
            return None

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
        amount_columns = ['원화사고금액', '원화판정금액']
        for col in amount_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"데이터 로드 오류: {e}")
        return None

def create_main_dashboard(df):
    """메인 대시보드"""
    st.markdown("""
    <div class="main-header">
        <h1>📋 보험사고 판정 분석 시스템</h1>
        <p>AI 기반 유사사례 검색으로 신속하고 정확한 판정 지원</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 핵심 지표
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <h3>전체 사고</h3>
            <h2>{len(df):,}건</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        payment_rate = (df['판정구분'] == '지급').mean() * 100
        st.markdown(f"""
        <div class="metric-box">
            <h3>지급 비율</h3>
            <h2>{payment_rate:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        max_rounds = df['판정회차'].max()
        st.markdown(f"""
        <div class="metric-box">
            <h3>최대 판정회차</h3>
            <h2>{max_rounds}회</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        meaningful_desc = len(df[
            (df['사고설명'].notna()) & 
            (df['사고설명'].str.len() > 10) &
            (~df['사고설명'].str.contains('설명없음|첨부파일참고|해당없음', na=False, case=False))
        ])
        desc_rate = meaningful_desc / len(df) * 100
        st.markdown(f"""
        <div class="metric-box">
            <h3>AI 분석 가능</h3>
            <h2>{desc_rate:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 주요 차트
    col1, col2 = st.columns(2)
    
    with col1:
        # 수입국별 상위 10개
        country_counts = df['수입국'].value_counts().head(10)
        fig1 = px.bar(
            x=country_counts.values,
            y=country_counts.index,
            orientation='h',
            title="상위 10개국 사고 발생 현황",
            color_discrete_sequence=['#007bff']
        )
        fig1.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # 판정구분별 분포
        decision_counts = df['판정구분'].value_counts()
        colors = ['#28a745', '#dc3545', '#ffc107', '#6c757d']
        
        fig2 = px.pie(
            values=decision_counts.values,
            names=decision_counts.index,
            title="판정구분별 분포",
            color_discrete_sequence=colors
        )
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

def create_similarity_search_interface(system, df):
    """유사사례 검색 인터페이스"""
    st.subheader("🔍 유사사례 검색")
    
    # AI 모델 상태
    model_ready = system.initialize_ai_model()
    if model_ready:
        st.success("✅ AI 텍스트 분석 모델 준비 완료")
    else:
        st.warning("⚠️ AI 모델 사용 불가 - 기본 검색 모드")
    
    # 검색 폼
    with st.form("search_form"):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("**📋 사고 정보 입력**")
            
            input_country = st.selectbox(
                "수입국:",
                options=df['수입국'].value_counts().head(20).index
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
            
            input_description = st.text_area(
                "사고설명:",
                placeholder="사고의 구체적인 상황을 입력하세요...",
                height=120
            )
        
        with col2:
            st.write("**🔧 검색 옵션**")
            
            # 필터 옵션
            filter_same_country = st.checkbox("동일 수입국만", value=False)
            filter_same_type = st.checkbox("동일 사고유형만", value=False)
            filter_same_insurance = st.checkbox("동일 보험종목만", value=False)
            
            max_results = st.slider("최대 결과 수:", 3, 10, 5)
            
            st.write("**⚖️ 현재 가중치**")
            st.write("• 텍스트 유사도: 45%")
            st.write("• 사고유형: 25%")
            st.write("• 수입국: 15%")
            st.write("• 금액유사도: 10%")
            st.write("• 보험종목: 5%")
        
        submitted = st.form_submit_button("🔍 유사사례 검색", type="primary")
    
    if submitted:
        start_time = time.time()
        
        # 입력 데이터 구성
        case_data = {
            '수입국': input_country,
            '보험종목': input_insurance,
            '사고유형명': input_accident_type,
            '원화사고금액': input_amount,
            '사고설명': input_description
        }
        
        with st.spinner("유사사례를 검색하는 중..."):
            # 필터링 적용
            search_df = df.copy()
            
            if filter_same_country:
                search_df = search_df[search_df['수입국'] == input_country]
            if filter_same_type:
                search_df = search_df[search_df['사고유형명'] == input_accident_type]
            if filter_same_insurance:
                search_df = search_df[search_df['보험종목'] == input_insurance]
            
            # 의미있는 설명이 있는 사례 우선
            if input_description:
                meaningful_df = search_df[
                    (search_df['사고설명'].notna()) & 
                    (search_df['사고설명'].str.len() > 10) &
                    (~search_df['사고설명'].str.contains('설명없음|첨부파일참고|해당없음', na=False, case=False))
                ]
                if len(meaningful_df) > 50:
                    search_df = meaningful_df
            
            # 유사도 계산
            similarities = system.calculate_similarity_scores(case_data, search_df, max_candidates=200)
            top_similar = similarities[:max_results]
            
            search_time = time.time() - start_time
        
        # 결과 표시
        if top_similar:
            # 예측 결과
            similar_decisions = [case[2]['판정구분'] for case in top_similar]
            decision_counts = Counter(similar_decisions)
            predicted_decision = decision_counts.most_common(1)[0][0]
            confidence = decision_counts[predicted_decision] / len(similar_decisions)
            
            # 예측 결과 박스
            if predicted_decision == '지급':
                box_class = "prediction-box success-box"
            elif predicted_decision == '면책':
                box_class = "prediction-box error-box"
            else:
                box_class = "prediction-box warning-box"
            
            st.markdown(f"""
            <div class="{box_class}">
                <h2>🎯 예상 판정: {predicted_decision}</h2>
                <p>신뢰도: {confidence:.1%} | 검색 시간: {search_time:.1f}초</p>
                <small>상위 {len(top_similar)}개 유사사례 분석 결과</small>
            </div>
            """, unsafe_allow_html=True)
            
            # 통계 차트
            col1, col2 = st.columns(2)
            
            with col1:
                # 판정구분 분포
                decision_df = pd.DataFrame(list(decision_counts.items()), columns=['판정구분', '건수'])
                fig_pred = px.pie(
                    decision_df,
                    values='건수',
                    names='판정구분',
                    title="유사사례 판정구분 분포",
                    color_discrete_sequence=['#28a745', '#dc3545', '#ffc107']
                )
                st.plotly_chart(fig_pred, use_container_width=True)
            
            with col2:
                # 유사도 점수 분포
                scores = [sim[0] for sim in top_similar]
                fig_sim = px.bar(
                    x=list(range(1, len(scores) + 1)),
                    y=scores,
                    title="유사도 점수 분포",
                    labels={'x': '순위', 'y': '유사도 점수'},
                    color_discrete_sequence=['#007bff']
                )
                fig_sim.update_layout(showlegend=False)
                st.plotly_chart(fig_sim, use_container_width=True)
            
            # 상세 결과
            st.subheader("📋 유사사례 상세 정보")
            
            for i, (total_score, text_sim, similar_case) in enumerate(top_similar):
                with st.expander(f"#{i+1} 유사도 {total_score:.3f} - {similar_case['판정구분']} ({similar_case['사고유형명']})"):
                    
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
                    
                    # 유사도 점수 시각화
                    st.write("**📊 유사도 분석**")
                    similarity_pct = min(total_score * 100, 100)  # float32 에러 방지
                    st.progress(float(similarity_pct) / 100)  # float 변환으로 에러 방지
                    st.caption(f"통합 유사도: {total_score:.1%} | 텍스트 유사도: {text_sim:.1%}")
                    
                    if pd.notna(similar_case['사고설명']) and len(str(similar_case['사고설명'])) > 10:
                        st.write("**📝 사고설명**")
                        st.markdown(f"> {similar_case['사고설명']}")
        else:
            st.warning("조건에 맞는 유사사례를 찾을 수 없습니다.")

def create_model_performance_tab(system, df):
    """모델 성능 분석 탭"""
    st.subheader("📊 모델 성능 분석")
    
    if system.model is None:
        if st.button("📚 모델 학습 실행"):
            with st.spinner("모델을 학습하는 중..."):
                results = system.train_prediction_model(df)
                if results:
                    st.session_state.model_results = results
                    st.success("모델 학습 완료!")
    
    if hasattr(st.session_state, 'model_results') and st.session_state.model_results:
        results = st.session_state.model_results
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("교차검증 평균 정확도", f"{results['cv_mean']:.3f}")
        with col2:
            st.metric("표준편차", f"{results['cv_std']:.3f}")
        with col3:
            st.metric("신뢰구간", f"±{1.96*results['cv_std']:.3f}")
        
        # 특성 중요도
        if 'feature_importance' in results:
            fig = px.bar(
                results['feature_importance'].head(10),
                x='importance',
                y='feature',
                orientation='h',
                title="상위 10개 특성 중요도",
                color_discrete_sequence=['#007bff']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # 가중치 분석
        st.write("**⚖️ 최적화된 유사도 가중치 (실험 결과 기반)**")
        weights_df = pd.DataFrame(list(system.optimal_weights.items()), 
                                 columns=['특성', '가중치'])
        
        fig_weights = px.bar(
            weights_df,
            x='가중치',
            y='특성',
            orientation='h',
            title="유사도 계산 가중치 분포",
            color_discrete_sequence=['#28a745']
        )
        fig_weights.update_layout(height=300)
        st.plotly_chart(fig_weights, use_container_width=True)

def main():
    """메인 함수"""
    # 데이터 로드
    df = load_data()
    if df is None:
        st.error("데이터를 로드할 수 없습니다.")
        return
    
    # 시스템 초기화
    if 'insurance_system' not in st.session_state:
        st.session_state.insurance_system = PracticalInsuranceSystem()
    
    system = st.session_state.insurance_system
    
    # 탭 구성
    tab1, tab2, tab3 = st.tabs([
        "📊 전체 현황", 
        "🔍 유사사례 검색", 
        "📈 모델 성능"
    ])
    
    with tab1:
        create_main_dashboard(df)
    
    with tab2:
        create_similarity_search_interface(system, df)
    
    with tab3:
        create_model_performance_tab(system, df)
    
    # 푸터
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6c757d; padding: 1rem;">
        <p>📋 보험사고 판정 분석 시스템</p>
        <small>실무진을 위한 AI 기반 의사결정 지원 도구</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()