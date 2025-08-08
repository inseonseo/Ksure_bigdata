import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoModel, AutoTokenizer
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import pickle
import warnings
import re
from datetime import datetime
import os
warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(
    page_title="보험사고 판정 흐름 분석 및 예측 시스템 (KoSimCSE 강화)",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
    
    .prediction-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .kosimcse-result {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        color: #2d3436;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .similar-case {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: #fafafa;
    }
</style>
""", unsafe_allow_html=True)

class KoSimCSEEnhancedSystem:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.kosimcse_model = None
        self.kosimcse_tokenizer = None
        self.text_vectorizer = None
        self.label_encoders = {}
        self.feature_importance = None
        self.embeddings_cache = {}
        self.cache_file = "enhanced_kosimcse_cache.pkl"
        self.optimal_weights = {
            'kosimcse_similarity': 0.5,  # KoSimCSE 유사도 비중 증가
            'accident_type': 0.2,
            'country': 0.15,
            'amount_range': 0.1,
            'insurance_type': 0.05
        }
    
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
            st.error(f"KoSimCSE 모델 로드 실패: {e}")
            return None, None
    
    def initialize_kosimcse(self):
        """KoSimCSE 모델 초기화"""
        if self.kosimcse_model is None:
            self.kosimcse_model, self.kosimcse_tokenizer = self.load_kosimcse_model()
            if self.kosimcse_model is not None:
                return True
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
    
    def get_kosimcse_embeddings(self, texts, batch_size=8):
        """KoSimCSE를 사용한 임베딩 생성"""
        if not self.initialize_kosimcse():
            return None
        
        # 텍스트 전처리
        processed_texts = [self.preprocess_text(text) for text in texts]
        valid_texts = [text for text in processed_texts if text]
        
        if not valid_texts:
            return None
        
        embeddings = []
        
        try:
            for i in range(0, len(valid_texts), batch_size):
                batch_texts = valid_texts[i:i + batch_size]
                
                # 토큰화
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
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
                    embeddings.extend(batch_embeddings)
            
            return np.array(embeddings)
            
        except Exception as e:
            st.error(f"KoSimCSE 임베딩 생성 오류: {e}")
            return None
    
    def calculate_kosimcse_similarity(self, query_text, candidate_texts):
        """KoSimCSE 기반 유사도 계산"""
        if not query_text or not candidate_texts:
            return []
        
        # 쿼리와 후보 텍스트 임베딩
        all_texts = [query_text] + list(candidate_texts)
        embeddings = self.get_kosimcse_embeddings(all_texts)
        
        if embeddings is None or len(embeddings) < 2:
            return []
        
        # 코사인 유사도 계산
        query_embedding = embeddings[0:1]
        candidate_embeddings = embeddings[1:]
        
        similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]
        
        return similarities
    
    def preprocess_features(self, df):
        """특성 전처리"""
        df_processed = df.copy()
        
        # 금액 구간 생성
        df_processed['금액구간'] = pd.cut(
            df_processed['원화사고금액'].fillna(0),
            bins=[0, 10000000, 50000000, 100000000, 500000000, float('inf')],
            labels=['1천만원미만', '1천만-5천만원', '5천만-1억원', '1억-5억원', '5억원이상']
        )
        
        # 사고유형 그룹화
        df_processed['사고유형그룹'] = df_processed['사고유형명'].apply(self._group_accident_type)
        
        # 텍스트 특성
        df_processed['사고설명_길이'] = df_processed['사고설명'].fillna('').str.len()
        df_processed['사고설명_유효'] = (
            (df_processed['사고설명'].notna()) & 
            (df_processed['사고설명'].str.len() > 10) &
            (~df_processed['사고설명'].str.contains('설명없음|첨부파일참고|해당없음', na=False, case=False))
        )
        
        return df_processed
    
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
    
    def calculate_enhanced_similarity_score(self, query_case, candidate_case, kosimcse_sim=None):
        """KoSimCSE 기반 강화된 유사도 점수 계산"""
        score = 0.0
        weights = self.optimal_weights
        
        # 1. KoSimCSE 유사도 (가장 높은 가중치)
        if kosimcse_sim is not None and 'kosimcse_similarity' in weights:
            score += weights['kosimcse_similarity'] * kosimcse_sim
        
        # 2. 사고유형 유사도
        if 'accident_type' in weights:
            if query_case.get('사고유형명') == candidate_case.get('사고유형명'):
                score += weights['accident_type']
            elif self._group_accident_type(query_case.get('사고유형명')) == self._group_accident_type(candidate_case.get('사고유형명')):
                score += weights['accident_type'] * 0.7
        
        # 3. 수입국 일치
        if 'country' in weights and query_case.get('수입국') == candidate_case.get('수입국'):
            score += weights['country']
        
        # 4. 금액대 유사도
        if 'amount_range' in weights:
            query_amount = query_case.get('원화사고금액', 0)
            candidate_amount = candidate_case.get('원화사고금액', 0)
            
            if query_amount > 0 and candidate_amount > 0:
                amount_ratio = min(query_amount, candidate_amount) / max(query_amount, candidate_amount)
                score += weights['amount_range'] * amount_ratio
        
        # 5. 보험종목 일치
        if 'insurance_type' in weights and query_case.get('보험종목') == candidate_case.get('보험종목'):
            score += weights['insurance_type']
        
        return score

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

def create_enhanced_overview_dashboard(df):
    """강화된 전체 현황 대시보드"""
    st.markdown('<div class="main-header"><h1>🤖 KoSimCSE 강화 보험사고 분석 시스템</h1><p>AI 기반 문맥 이해를 통한 정교한 유사사례 검색</p></div>', unsafe_allow_html=True)
    
    # 핵심 지표
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_cases = len(df)
        st.markdown(f"""
        <div class="metric-card">
            <h3>전체 사고건수</h3>
            <h2>{total_cases:,}건</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        payment_rate = (df['판정구분'] == '지급').mean() * 100
        st.markdown(f"""
        <div class="metric-card">
            <h3>지급 비율</h3>
            <h2>{payment_rate:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # 판정회차 분석
        max_rounds = df['판정회차'].max()
        multi_round_cases = (df['판정회차'] > 1).sum()
        multi_round_rate = multi_round_cases / len(df) * 100
        st.markdown(f"""
        <div class="metric-card">
            <h3>복수회차 사고</h3>
            <h2>{multi_round_rate:.1f}%</h2>
            <small>최대 {max_rounds}회차까지</small>
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
        <div class="metric-card">
            <h3>KoSimCSE 활용 가능</h3>
            <h2>{desc_rate:.1f}%</h2>
            <small>{meaningful_desc:,}건의 유의미한 설명</small>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 판정회차 상세 분석
    st.subheader("🔄 판정회차별 상세 분석")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 판정회차별 분포 (전체)
        round_counts = df['판정회차'].value_counts().sort_index()
        
        fig1 = px.bar(
            x=round_counts.index,
            y=round_counts.values,
            title="판정회차별 사고 분포 (전체)",
            labels={'x': '판정회차', 'y': '건수'},
            color=round_counts.values,
            color_continuous_scale='viridis'
        )
        fig1.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)
        
        # 통계 정보
        st.write("**📊 판정회차 상세 통계:**")
        for round_num in sorted(df['판정회차'].unique()):
            count = (df['판정회차'] == round_num).sum()
            pct = count / len(df) * 100
            st.write(f"• {round_num}회차: {count:,}건 ({pct:.1f}%)")
    
    with col2:
        # 판정회차별 판정구분 분포
        round_decision = df.groupby(['판정회차', '판정구분']).size().unstack(fill_value=0)
        
        fig2 = px.bar(
            round_decision,
            title="판정회차별 판정구분 분포",
            labels={'value': '건수', 'index': '판정회차'},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

def create_kosimcse_prediction_interface(enhanced_system, df):
    """KoSimCSE 강화 예측 인터페이스"""
    st.subheader("🤖 KoSimCSE 기반 고급 예측 및 유사사례 검색")
    
    # KoSimCSE 모델 상태 확인
    model_status = enhanced_system.initialize_kosimcse()
    
    if model_status:
        st.success("✅ KoSimCSE 모델이 성공적으로 로드되었습니다!")
    else:
        st.warning("⚠️ KoSimCSE 모델 로드에 실패했습니다. 기본 유사도 계산을 사용합니다.")
    
    st.write("**새로운 사고 정보를 입력하여 AI 기반 문맥 분석으로 유사사례를 찾아보세요.**")
    
    # 입력 폼
    with st.form("kosimcse_prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📋 기본 정보**")
            
            input_country = st.selectbox(
                "수입국:",
                options=df['수입국'].value_counts().head(20).index,
                help="사고가 발생한 수입국을 선택하세요"
            )
            
            input_insurance = st.selectbox(
                "보험종목:",
                options=df['보험종목'].value_counts().head(15).index,
                help="해당하는 보험종목을 선택하세요"
            )
            
            input_accident_type = st.selectbox(
                "사고유형:",
                options=df['사고유형명'].value_counts().head(15).index,
                help="사고의 유형을 선택하세요"
            )
            
            input_amount = st.number_input(
                "사고금액 (원):",
                min_value=0,
                value=50000000,
                step=1000000,
                format="%d",
                help="사고로 인한 손실 금액을 입력하세요"
            )
        
        with col2:
            st.write("**📝 사고 상세 설명 (KoSimCSE 분석 대상)**")
            
            input_description = st.text_area(
                "사고설명:",
                placeholder="""예시: 
수입자가 L/C 조건에 따른 대금 지급을 지연하고 있습니다. 
당초 약정된 지급일로부터 이미 3개월이 경과했으며, 
수입자의 재정상황 악화로 인해 추가 지연이 예상됩니다. 
현지 변호사를 통해 독촉을 진행하고 있으나 명확한 지급 일정을 제시받지 못하고 있는 상황입니다.""",
                height=150,
                help="사고의 구체적인 상황, 원인, 경과 등을 상세히 설명해주세요. 문맥이 상세할수록 더 정확한 유사사례를 찾을 수 있습니다."
            )
            
            # 검색 옵션
            st.write("**🔍 검색 옵션**")
            max_results = st.slider("최대 결과 수:", 5, 15, 8)
            use_filters = st.checkbox("동일 조건 우선 검색", value=True, 
                                    help="수입국, 사고유형 등이 동일한 사례를 우선적으로 검색")
        
        submitted = st.form_submit_button("🎯 KoSimCSE 분석 실행", type="primary")
    
    if submitted and input_description:
        with st.spinner("🤖 KoSimCSE 모델로 문맥을 분석하고 유사사례를 검색하는 중..."):
            
            # 입력 데이터 구성
            case_data = {
                '수입국': input_country,
                '보험종목': input_insurance,
                '사고유형명': input_accident_type,
                '원화사고금액': input_amount,
                '사고설명': input_description
            }
            
            # 필터링 적용 (선택사항)
            search_df = df.copy()
            if use_filters:
                # 동일 사고유형 또는 유사 그룹 우선
                accident_group = enhanced_system._group_accident_type(input_accident_type)
                search_df = search_df[
                    search_df['사고유형명'].apply(enhanced_system._group_accident_type) == accident_group
                ]
                
                if len(search_df) < 50:  # 너무 적으면 전체 검색
                    search_df = df.copy()
            
            # 의미있는 설명이 있는 사례만 선택
            meaningful_df = search_df[
                (search_df['사고설명'].notna()) & 
                (search_df['사고설명'].str.len() > 10) &
                (~search_df['사고설명'].str.contains('설명없음|첨부파일참고|해당없음', na=False, case=False))
            ].copy()
            
            if len(meaningful_df) == 0:
                st.warning("유의미한 사고설명이 있는 사례를 찾을 수 없습니다.")
                return
            
            # KoSimCSE 유사도 계산
            candidate_descriptions = meaningful_df['사고설명'].tolist()
            
            if model_status:
                # KoSimCSE 기반 유사도
                kosimcse_similarities = enhanced_system.calculate_kosimcse_similarity(
                    input_description, candidate_descriptions
                )
                
                if kosimcse_similarities is not None and len(kosimcse_similarities) > 0:
                    # 통합 유사도 점수 계산
                    similarities = []
                    for i, (idx, row) in enumerate(meaningful_df.iterrows()):
                        if i < len(kosimcse_similarities):
                            kosimcse_sim = kosimcse_similarities[i]
                            total_score = enhanced_system.calculate_enhanced_similarity_score(
                                case_data, row, kosimcse_sim
                            )
                            similarities.append((total_score, kosimcse_sim, row))
                    
                    # 정렬
                    similarities.sort(key=lambda x: x[0], reverse=True)
                    top_similar = similarities[:max_results]
                    
                    # 결과 표시
                    st.markdown(f"""
                    <div class="kosimcse-result">
                        <h2>🤖 KoSimCSE 분석 완료</h2>
                        <p>{len(meaningful_df):,}개 사례 중 상위 {len(top_similar)}개 유사사례를 발견했습니다</p>
                        <small>문맥 기반 AI 분석으로 의미적 유사도를 계산했습니다</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 예측 결과 (유사사례 기반)
                    similar_decisions = [case[2]['판정구분'] for case in top_similar[:5]]
                    decision_counts = Counter(similar_decisions)
                    predicted_decision = decision_counts.most_common(1)[0][0]
                    confidence = decision_counts[predicted_decision] / len(similar_decisions)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="prediction-result">
                            <h2>🎯 예측 결과</h2>
                            <h1>{predicted_decision}</h1>
                            <p>신뢰도: {confidence:.1%}</p>
                            <small>KoSimCSE 기반 상위 5개 유사사례 분석</small>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # 판정구분별 분포
                        decision_df = pd.DataFrame(list(decision_counts.items()), columns=['판정구분', '건수'])
                        fig_pred = px.pie(
                            decision_df,
                            values='건수',
                            names='판정구분',
                            title="유사사례 판정구분 분포",
                            color_discrete_map={'지급': '#2E8B57', '면책': '#DC143C', '지급유예': '#4682B4'}
                        )
                        st.plotly_chart(fig_pred, use_container_width=True)
                    
                    with col2:
                        # KoSimCSE 유사도 vs 통합 유사도 비교
                        kosimcse_scores = [sim[1] for sim in top_similar]
                        total_scores = [sim[0] for sim in top_similar]
                        
                        fig_compare = go.Figure()
                        fig_compare.add_trace(go.Scatter(
                            x=list(range(1, len(kosimcse_scores) + 1)),
                            y=kosimcse_scores,
                            mode='lines+markers',
                            name='KoSimCSE 유사도',
                            line=dict(color='orange')
                        ))
                        fig_compare.add_trace(go.Scatter(
                            x=list(range(1, len(total_scores) + 1)),
                            y=total_scores,
                            mode='lines+markers',
                            name='통합 유사도',
                            line=dict(color='blue')
                        ))
                        fig_compare.update_layout(
                            title="KoSimCSE vs 통합 유사도 비교",
                            xaxis_title="순위",
                            yaxis_title="유사도 점수"
                        )
                        st.plotly_chart(fig_compare, use_container_width=True)
                    
                    # 상세 유사사례
                    st.subheader("📋 KoSimCSE 기반 상위 유사사례")
                    
                    for i, (total_score, kosimcse_sim, similar_case) in enumerate(top_similar):
                        with st.expander(f"#{i+1} 통합유사도 {total_score:.3f} (KoSimCSE: {kosimcse_sim:.3f}) - {similar_case['판정구분']} ({similar_case['사고유형명']})"):
                            
                            # 유사도 비교 진행바
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**KoSimCSE 유사도:**")
                                st.progress(kosimcse_sim)
                                st.caption(f"{kosimcse_sim:.1%}")
                            with col2:
                                st.write("**통합 유사도:**")
                                st.progress(total_score)
                                st.caption(f"{total_score:.1%}")
                            
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
                                    st.info(f"판정구분: {similar_case['판정구분']}")
                                
                                st.write(f"• 판정사유: **{similar_case['판정사유']}**")
                                st.write(f"• 판정회차: {similar_case['판정회차']}회")
                                st.write(f"• 사고진행상태: {similar_case['사고진행상태']}")
                            
                            st.write("**📝 사고설명 (KoSimCSE 분석 대상)**")
                            st.markdown(f"> {similar_case['사고설명']}")
                            
                            # 문맥 유사성 분석 (간단한 키워드 매칭)
                            input_words = set(input_description.lower().split())
                            case_words = set(str(similar_case['사고설명']).lower().split())
                            common_words = input_words.intersection(case_words)
                            
                            if common_words:
                                meaningful_words = [word for word in common_words if len(word) > 2]
                                if meaningful_words:
                                    st.write("**🔑 공통 키워드:**")
                                    st.write(" • ".join([f"`{word}`" for word in meaningful_words[:15]]))
                
                else:
                    st.error("KoSimCSE 유사도 계산에 실패했습니다.")
            else:
                st.warning("KoSimCSE 모델을 사용할 수 없어 기본 검색을 수행합니다.")

def main():
    """메인 함수"""
    # 데이터 로드
    df = load_data()
    if df is None:
        st.error("데이터를 로드할 수 없습니다. 파일 경로를 확인해주세요.")
        return
    
    # KoSimCSE 강화 시스템 초기화
    if 'enhanced_system' not in st.session_state:
        st.session_state.enhanced_system = KoSimCSEEnhancedSystem()
    
    enhanced_system = st.session_state.enhanced_system
    
    # 사이드바 설정
    st.sidebar.header("🤖 KoSimCSE 시스템 설정")
    
    # KoSimCSE 가중치 조정
    st.sidebar.write("**⚖️ KoSimCSE 강화 가중치 조정**")
    st.sidebar.write("*KoSimCSE 문맥 분석의 비중을 조정하세요*")
    
    kosimcse_weight = st.sidebar.slider("KoSimCSE 문맥 유사도", 0.0, 1.0, enhanced_system.optimal_weights['kosimcse_similarity'], 0.05)
    accident_weight = st.sidebar.slider("사고유형", 0.0, 1.0, enhanced_system.optimal_weights['accident_type'], 0.05)
    country_weight = st.sidebar.slider("수입국", 0.0, 1.0, enhanced_system.optimal_weights['country'], 0.05)
    amount_weight = st.sidebar.slider("금액대", 0.0, 1.0, enhanced_system.optimal_weights['amount_range'], 0.05)
    insurance_weight = st.sidebar.slider("보험종목", 0.0, 1.0, enhanced_system.optimal_weights['insurance_type'], 0.05)
    
    # 가중치 정규화
    total_weight = kosimcse_weight + accident_weight + country_weight + amount_weight + insurance_weight
    if total_weight > 0:
        enhanced_system.optimal_weights = {
            'kosimcse_similarity': kosimcse_weight / total_weight,
            'accident_type': accident_weight / total_weight,
            'country': country_weight / total_weight,
            'amount_range': amount_weight / total_weight,
            'insurance_type': insurance_weight / total_weight
        }
    
    # 현재 가중치 표시
    st.sidebar.write("**현재 가중치 분포:**")
    for key, value in enhanced_system.optimal_weights.items():
        st.sidebar.write(f"• {key}: {value:.1%}")
    
    # 메인 탭 구성
    tab1, tab2 = st.tabs([
        "📊 강화된 현황 분석", 
        "🤖 KoSimCSE 예측 시스템"
    ])
    
    with tab1:
        create_enhanced_overview_dashboard(df)
    
    with tab2:
        create_kosimcse_prediction_interface(enhanced_system, df)
    
    # 푸터
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🤖 KoSimCSE 강화 보험사고 예측 시스템</p>
        <small>한국어 문맥 이해 AI 모델을 활용한 정교한 유사사례 분석</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()