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
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import pickle
import warnings
import re
from datetime import datetime
warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(
    page_title="보험사고 판정 흐름 분석 및 예측 시스템",
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
    
    .similar-case {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: #fafafa;
    }
</style>
""", unsafe_allow_html=True)

class InsuranceAnalysisSystem:
    def __init__(self):
        self.model = None
        self.text_vectorizer = None
        self.label_encoders = {}
        self.feature_importance = None
        self.optimal_weights = {
            'text_similarity': 0.4,
            'accident_type': 0.25,
            'country': 0.15,
            'amount_range': 0.15,
            'insurance_type': 0.05
        }
    
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
    
    def train_prediction_model(self, df):
        """예측 모델 학습"""
        df_processed = self.preprocess_features(df)
        
        # 범주형 변수 인코딩
        categorical_features = ['수입국', '보험종목', '사고유형그룹', '금액구간', '상품분류그룹명']
        
        for feature in categorical_features:
            le = LabelEncoder()
            df_processed[f'{feature}_encoded'] = le.fit_transform(df_processed[feature].fillna('Unknown'))
            self.label_encoders[feature] = le
        
        # 특성 선택
        feature_columns = [f'{feature}_encoded' for feature in categorical_features]
        feature_columns.extend(['사고설명_길이', '원화사고금액'])
        
        X = df_processed[feature_columns].fillna(0)
        y = df_processed['판정구분']
        
        # 학습/검증 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 모델 학습
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_train, y_train)
        
        # 특성 중요도
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # 모델 성능 평가
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        return {
            'train_score': train_score,
            'test_score': test_score,
            'feature_importance': self.feature_importance
        }
    
    def calculate_similarity_score(self, query_case, candidate_case):
        """가중치 기반 유사도 점수 계산"""
        score = 0.0
        weights = self.optimal_weights
        
        # 1. 텍스트 유사도 (간단한 Jaccard 유사도)
        if 'text_similarity' in weights and '사고설명' in query_case and '사고설명' in candidate_case:
            query_text = str(query_case['사고설명']).lower()
            candidate_text = str(candidate_case['사고설명']).lower()
            
            query_words = set(query_text.split())
            candidate_words = set(candidate_text.split())
            
            if query_words and candidate_words:
                jaccard_sim = len(query_words.intersection(candidate_words)) / len(query_words.union(candidate_words))
                score += weights['text_similarity'] * jaccard_sim
        
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
        df = pd.read_csv('KoSimCSE/new/design.csv', encoding='cp949')
        
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

def create_overview_dashboard(df):
    """전체 현황 대시보드"""
    st.markdown('<div class="main-header"><h1>📊 보험사고 판정 흐름 분석 및 예측 시스템</h1></div>', unsafe_allow_html=True)
    
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
        avg_amount = df['원화사고금액'].mean() / 100000000
        st.markdown(f"""
        <div class="metric-card">
            <h3>평균 사고금액</h3>
            <h2>{avg_amount:.1f}억원</h2>
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
            <h3>유효 설명 비율</h3>
            <h2>{desc_rate:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 주요 차트들
    col1, col2 = st.columns(2)
    
    with col1:
        # 수입국별 상위 10개
        country_counts = df['수입국'].value_counts().head(10)
        fig1 = px.bar(
            x=country_counts.values,
            y=country_counts.index,
            orientation='h',
            title="🌍 상위 10개국 사고 발생 현황",
            labels={'x': '사고 건수', 'y': '수입국'},
            color=country_counts.values,
            color_continuous_scale='viridis'
        )
        fig1.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # 판정구분별 분포
        decision_counts = df['판정구분'].value_counts()
        colors = ['#2E8B57' if x == '지급' else '#DC143C' if x == '면책' else '#4682B4' 
                 for x in decision_counts.index]
        
        fig2 = px.pie(
            values=decision_counts.values,
            names=decision_counts.index,
            title="⚖️ 판정구분별 분포",
            color_discrete_sequence=colors
        )
        fig2.update_traces(textposition='inside', textinfo='percent+label')
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 사고유형별 상위 8개
        accident_types = df['사고유형명'].value_counts().head(8)
        fig3 = px.bar(
            x=accident_types.index,
            y=accident_types.values,
            title="⚠️ 주요 사고유형별 발생 현황",
            labels={'x': '사고유형', 'y': '건수'},
            color=accident_types.values,
            color_continuous_scale='plasma'
        )
        fig3.update_layout(height=400, xaxis_tickangle=-45, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        # 금액구간별 분포
        df_amount = df[df['원화사고금액'].notna() & (df['원화사고금액'] > 0)].copy()
        df_amount['금액구간'] = pd.cut(
            df_amount['원화사고금액'],
            bins=[0, 10000000, 50000000, 100000000, 500000000, 1000000000, float('inf')],
            labels=['1천만원 미만', '1천만-5천만원', '5천만-1억원', '1억-5억원', '5억-10억원', '10억원 이상']
        )
        
        amount_dist = df_amount['금액구간'].value_counts().sort_index()
        fig4 = px.bar(
            x=amount_dist.index,
            y=amount_dist.values,
            title="💰 사고금액 구간별 분포",
            labels={'x': '금액구간', 'y': '건수'},
            color=amount_dist.values,
            color_continuous_scale='blues'
        )
        fig4.update_layout(height=400, xaxis_tickangle=-45, showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

def create_process_flow_analysis(df):
    """보상 프로세스 흐름 분석"""
    st.subheader("🔄 보상 프로세스 흐름 분석")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 판정회차별 분포
        round_counts = df['판정회차'].value_counts().sort_index()
        fig1 = px.bar(
            x=round_counts.index,
            y=round_counts.values,
            title="판정회차별 사고 분포",
            labels={'x': '판정회차', 'y': '건수'},
            color=round_counts.values,
            color_continuous_scale='greens'
        )
        fig1.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)
        
        # 통계 정보
        st.write("**📊 판정회차 통계:**")
        st.write(f"• 평균 판정회차: {df['판정회차'].mean():.1f}회")
        st.write(f"• 최대 판정회차: {df['판정회차'].max()}회")
        st.write(f"• 1회차로 종료: {(df['판정회차'] == 1).sum():,}건 ({(df['판정회차'] == 1).mean()*100:.1f}%)")
        st.write(f"• 2회차 이상: {(df['판정회차'] >= 2).sum():,}건 ({(df['판정회차'] >= 2).mean()*100:.1f}%)")
    
    with col2:
        # 사고진행상태별 분포
        status_counts = df['사고진행상태'].value_counts().head(8)
        fig2 = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="사고진행상태별 분포 (상위 8개)",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig2.update_traces(textposition='inside', textinfo='percent+label')
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
    
    # 판정회차별 판정구분 변화 분석
    st.write("**🔄 판정회차별 판정구분 변화:**")
    
    # 사고번호별로 그룹화하여 판정회차별 변화 추적
    case_progression = df.groupby('사고번호').apply(
        lambda x: x.sort_values('판정회차')[['판정회차', '판정구분']].to_dict('records')
    )
    
    # 1차→2차 변화 패턴 분석
    transitions = []
    for case_data in case_progression:
        if len(case_data) >= 2:
            first_decision = case_data[0]['판정구분']
            second_decision = case_data[1]['판정구분']
            transitions.append(f"{first_decision} → {second_decision}")
    
    if transitions:
        transition_counts = Counter(transitions)
        transition_df = pd.DataFrame(list(transition_counts.items()), columns=['변화패턴', '건수'])
        transition_df = transition_df.sort_values('건수', ascending=False).head(10)
        
        fig3 = px.bar(
            transition_df,
            x='건수',
            y='변화패턴',
            orientation='h',
            title="1차→2차 판정구분 변화 패턴 (상위 10개)",
            color='건수',
            color_continuous_scale='viridis'
        )
        fig3.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

def create_prediction_interface(analysis_system, df):
    """예측 인터페이스"""
    st.subheader("🔮 사고 판정 예측 및 유사사례 검색")
    
    # 모델 학습 상태 확인
    if analysis_system.model is None:
        st.warning("⚠️ 예측 모델이 학습되지 않았습니다. 사이드바에서 '모델 학습 실행'을 클릭해주세요.")
        return
    
    st.write("**새로운 사고 정보를 입력하여 예상 판정과 유사사례를 확인해보세요.**")
    
    # 입력 폼
    with st.form("prediction_form"):
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
            
            input_product_group = st.selectbox(
                "상품분류그룹:",
                options=['선택안함'] + list(df['상품분류그룹명'].value_counts().head(15).index),
                help="수출품목의 분류그룹을 선택하세요"
            )
        
        with col2:
            st.write("**💰 금액 및 상세 정보**")
            
            input_amount = st.number_input(
                "사고금액 (원):",
                min_value=0,
                value=50000000,
                step=1000000,
                format="%d",
                help="사고로 인한 손실 금액을 입력하세요"
            )
            
            input_description = st.text_area(
                "사고설명:",
                placeholder="사고의 상세한 내용을 입력해주세요. 예: 수입자가 대금 지급을 지연하여 발생한 사고입니다...",
                height=120,
                help="사고의 구체적인 상황과 원인을 설명해주세요"
            )
        
        submitted = st.form_submit_button("🎯 예측 및 유사사례 검색", type="primary")
    
    if submitted:
        # 입력 데이터 구성
        case_data = {
            '수입국': input_country,
            '보험종목': input_insurance,
            '사고유형명': input_accident_type,
            '상품분류그룹명': input_product_group if input_product_group != '선택안함' else None,
            '원화사고금액': input_amount,
            '사고설명': input_description
        }
        
        with st.spinner("분석 중..."):
            # 유사사례 검색
            similarities = []
            for idx, row in df.iterrows():
                sim_score = analysis_system.calculate_similarity_score(case_data, row)
                similarities.append((sim_score, row))
            
            # 상위 유사사례
            similarities.sort(key=lambda x: x[0], reverse=True)
            top_similar = similarities[:10]
            
            # 예측 결과 (유사사례 기반)
            similar_decisions = [case[1]['판정구분'] for case in top_similar[:5]]
            decision_counts = Counter(similar_decisions)
            predicted_decision = decision_counts.most_common(1)[0][0]
            confidence = decision_counts[predicted_decision] / len(similar_decisions)
            
            # 결과 표시
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown(f"""
                <div class="prediction-result">
                    <h2>🎯 예측 결과</h2>
                    <h1>{predicted_decision}</h1>
                    <p>신뢰도: {confidence:.1%}</p>
                    <small>상위 5개 유사사례 기반 예측</small>
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
                # 유사도 분포
                sim_scores = [sim[0] for sim in top_similar]
                fig_sim = px.bar(
                    x=list(range(1, len(sim_scores) + 1)),
                    y=sim_scores,
                    title="상위 10개 유사사례 유사도 점수",
                    labels={'x': '순위', 'y': '유사도 점수'},
                    color=sim_scores,
                    color_continuous_scale='viridis'
                )
                fig_sim.update_layout(showlegend=False)
                st.plotly_chart(fig_sim, use_container_width=True)
            
            # 상세 유사사례
            st.subheader("📋 상위 유사사례 상세 정보")
            
            for i, (sim_score, similar_case) in enumerate(top_similar[:5]):
                with st.expander(f"#{i+1} 유사도 {sim_score:.3f} - {similar_case['판정구분']} ({similar_case['사고유형명']})"):
                    
                    # 유사도 진행바
                    st.progress(sim_score)
                    st.caption(f"유사도: {sim_score:.1%}")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**📋 사례 정보**")
                        st.write(f"• 보상파일번호: `{similar_case['보상파일번호']}`")
                        st.write(f"• 사고번호: `{similar_case['사고번호']}`")
                        st.write(f"• 수입국: **{similar_case['수입국']}**")
                        st.write(f"• 보험종목: {similar_case['보험종목']}")
                        st.write(f"• 상품분류: {similar_case['상품분류그룹명']}")
                        
                        if pd.notna(similar_case['원화사고금액']):
                            amount_str = f"{similar_case['원화사고금액']:,.0f}원"
                            if similar_case['원화사고금액'] >= 100000000:
                                amount_str += f" ({similar_case['원화사고금액']/100000000:.1f}억원)"
                            st.write(f"• 사고금액: **{amount_str}**")
                    
                    with col2:
                        st.write("**⚖️ 판정 정보**")
                        
                        # 판정구분 색상 표시
                        if similar_case['판정구분'] == '지급':
                            st.success(f"판정구분: {similar_case['판정구분']}")
                        elif similar_case['판정구분'] == '면책':
                            st.error(f"판정구분: {similar_case['판정구분']}")
                        else:
                            st.info(f"판정구분: {similar_case['판정구분']}")
                        
                        st.write(f"• 판정사유: **{similar_case['판정사유']}**")
                        st.write(f"• 판정회차: {similar_case['판정회차']}회")
                        st.write(f"• 사고진행상태: {similar_case['사고진행상태']}")
                        
                        if pd.notna(similar_case['향후결제전망']) and similar_case['향후결제전망'] != '판단불가':
                            st.write(f"• 향후결제전망: {similar_case['향후결제전망']}")
                    
                    if pd.notna(similar_case['사고설명']) and len(str(similar_case['사고설명'])) > 10:
                        st.write("**📝 사고설명**")
                        st.markdown(f"> {similar_case['사고설명']}")
                        
                        # 공통 키워드 하이라이팅
                        if input_description:
                            query_words = set(input_description.lower().split())
                            case_words = set(str(similar_case['사고설명']).lower().split())
                            common_words = query_words.intersection(case_words)
                            
                            if common_words and len(common_words) > 0:
                                meaningful_words = [word for word in common_words if len(word) > 2]
                                if meaningful_words:
                                    st.write("**🔑 공통 키워드:**")
                                    st.write(" • ".join([f"`{word}`" for word in meaningful_words[:10]]))

def create_analytics_dashboard(analysis_system, df):
    """분석 대시보드"""
    st.subheader("📊 시스템 분석 및 인사이트")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 가중치 설정 현황
        st.write("**⚖️ 현재 유사도 가중치 설정**")
        
        weights_df = pd.DataFrame(list(analysis_system.optimal_weights.items()), 
                                 columns=['특성', '가중치'])
        
        fig_weights = px.pie(
            weights_df,
            values='가중치',
            names='특성',
            title="유사도 계산 가중치 분포",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_weights, use_container_width=True)
        
        # 가중치 상세 정보
        for _, row in weights_df.iterrows():
            st.write(f"• {row['특성']}: {row['가중치']:.1%}")
    
    with col2:
        # 텍스트 데이터 품질 분석
        st.write("**📝 텍스트 데이터 품질 분석**")
        
        df_text = df[df['사고설명'].notna()].copy()
        df_text['설명_길이'] = df_text['사고설명'].str.len()
        
        # 길이별 분포
        length_bins = [0, 20, 50, 100, 200, 500, float('inf')]
        length_labels = ['20자 미만', '20-50자', '50-100자', '100-200자', '200-500자', '500자 이상']
        df_text['길이_구간'] = pd.cut(df_text['설명_길이'], bins=length_bins, labels=length_labels)
        
        length_dist = df_text['길이_구간'].value_counts()
        
        fig_length = px.bar(
            x=length_dist.values,
            y=length_dist.index,
            orientation='h',
            title="사고설명 길이별 분포",
            labels={'x': '건수', 'y': '길이 구간'},
            color=length_dist.values,
            color_continuous_scale='blues'
        )
        st.plotly_chart(fig_length, use_container_width=True)
    
    # 추가 분석
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # 수입국별 평균 사고금액
        country_avg_amount = df.groupby('수입국')['원화사고금액'].mean().sort_values(ascending=False).head(10)
        
        fig_country_amount = px.bar(
            x=country_avg_amount.values / 100000000,
            y=country_avg_amount.index,
            orientation='h',
            title="수입국별 평균 사고금액 (억원)",
            labels={'x': '평균 사고금액 (억원)', 'y': '수입국'},
            color=country_avg_amount.values,
            color_continuous_scale='reds'
        )
        fig_country_amount.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_country_amount, use_container_width=True)
    
    with col2:
        # 보험종목별 지급률
        insurance_payment_rate = df.groupby('보험종목').apply(
            lambda x: (x['판정구분'] == '지급').mean() * 100
        ).sort_values(ascending=False).head(10)
        
        fig_payment_rate = px.bar(
            x=insurance_payment_rate.values,
            y=insurance_payment_rate.index,
            orientation='h',
            title="보험종목별 지급률 (%)",
            labels={'x': '지급률 (%)', 'y': '보험종목'},
            color=insurance_payment_rate.values,
            color_continuous_scale='greens'
        )
        fig_payment_rate.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_payment_rate, use_container_width=True)
    
    with col3:
        # 사고유형별 평균 처리기간 (판정일 - 사고접수일자)
        df_with_dates = df[(df['판정일'].notna()) & (df['사고접수일자'].notna())].copy()
        df_with_dates['처리기간'] = (df_with_dates['판정일'] - df_with_dates['사고접수일자']).dt.days
        
        processing_time = df_with_dates.groupby('사고유형명')['처리기간'].mean().sort_values(ascending=False).head(10)
        
        fig_processing = px.bar(
            x=processing_time.values,
            y=processing_time.index,
            orientation='h',
            title="사고유형별 평균 처리기간 (일)",
            labels={'x': '평균 처리기간 (일)', 'y': '사고유형'},
            color=processing_time.values,
            color_continuous_scale='oranges'
        )
        fig_processing.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_processing, use_container_width=True)

def main():
    """메인 함수"""
    # 데이터 로드
    df = load_data()
    if df is None:
        st.error("데이터를 로드할 수 없습니다. 파일 경로를 확인해주세요.")
        return
    
    # 분석 시스템 초기화
    if 'analysis_system' not in st.session_state:
        st.session_state.analysis_system = InsuranceAnalysisSystem()
        st.session_state.model_trained = False
    
    analysis_system = st.session_state.analysis_system
    
    # 사이드바 설정
    st.sidebar.header("🔧 시스템 설정")
    
    # 모델 학습
    if st.sidebar.button("📚 예측 모델 학습 실행"):
        with st.spinner("모델을 학습하는 중..."):
            try:
                model_results = analysis_system.train_prediction_model(df)
                st.session_state.model_trained = True
                st.sidebar.success(f"모델 학습 완료! (정확도: {model_results['test_score']:.3f})")
            except Exception as e:
                st.sidebar.error(f"모델 학습 실패: {e}")
    
    # 가중치 조정
    st.sidebar.write("**⚖️ 유사도 가중치 조정**")
    st.sidebar.write("*가중치를 조정하여 유사도 계산을 최적화하세요*")
    
    text_weight = st.sidebar.slider("텍스트 유사도", 0.0, 1.0, analysis_system.optimal_weights['text_similarity'], 0.05)
    accident_weight = st.sidebar.slider("사고유형", 0.0, 1.0, analysis_system.optimal_weights['accident_type'], 0.05)
    country_weight = st.sidebar.slider("수입국", 0.0, 1.0, analysis_system.optimal_weights['country'], 0.05)
    amount_weight = st.sidebar.slider("금액대", 0.0, 1.0, analysis_system.optimal_weights['amount_range'], 0.05)
    insurance_weight = st.sidebar.slider("보험종목", 0.0, 1.0, analysis_system.optimal_weights['insurance_type'], 0.05)
    
    # 가중치 정규화
    total_weight = text_weight + accident_weight + country_weight + amount_weight + insurance_weight
    if total_weight > 0:
        analysis_system.optimal_weights = {
            'text_similarity': text_weight / total_weight,
            'accident_type': accident_weight / total_weight,
            'country': country_weight / total_weight,
            'amount_range': amount_weight / total_weight,
            'insurance_type': insurance_weight / total_weight
        }
    
    # 메인 탭 구성
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 전체 현황", 
        "🔄 프로세스 흐름", 
        "🔮 예측 및 검색", 
        "📈 분석 대시보드"
    ])
    
    with tab1:
        create_overview_dashboard(df)
    
    with tab2:
        create_process_flow_analysis(df)
    
    with tab3:
        create_prediction_interface(analysis_system, df)
    
    with tab4:
        create_analytics_dashboard(analysis_system, df)
    
    # 푸터
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🏢 보험사고 판정 흐름 분석 및 예측 시스템</p>
        <small>실무진의 의사결정 지원을 위한 AI 기반 분석 도구</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()