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
warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(
    page_title="보험사고 예측 및 유사사례 검색 시스템",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

class InsurancePredictionSystem:
    def __init__(self):
        self.model = None
        self.text_vectorizer = None
        self.label_encoders = {}
        self.feature_importance = None
        self.optimal_weights = {
            'text_similarity': 0.4,
            'accident_type': 0.2,
            'country': 0.15,
            'amount_range': 0.15,
            'insurance_type': 0.1
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
        
        # 사고유형 그룹화 (신용위험 계열 통합)
        df_processed['사고유형그룹'] = df_processed['사고유형명'].apply(self._group_accident_type)
        
        # 텍스트 특성 추출
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
        
        # 텍스트 벡터화 (의미있는 설명만)
        meaningful_texts = df_processed[df_processed['사고설명_유효']]['사고설명'].fillna('')
        if len(meaningful_texts) > 100:  # 최소 100개 이상의 의미있는 텍스트가 있을 때만
            self.text_vectorizer = TfidfVectorizer(
                max_features=200,
                ngram_range=(1, 2),
                min_df=2,
                stop_words=None
            )
            text_features = self.text_vectorizer.fit_transform(meaningful_texts)
            text_df = pd.DataFrame(
                text_features.toarray(), 
                columns=[f'text_feature_{i}' for i in range(text_features.shape[1])],
                index=meaningful_texts.index
            )
        else:
            text_df = pd.DataFrame()
        
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
        
        # 텍스트 특성 추가 (가능한 경우)
        if not text_df.empty:
            # 인덱스 맞추기
            X_with_text = X.join(text_df, how='left').fillna(0)
        else:
            X_with_text = X
        
        y = df_processed['판정구분']
        
        # 학습/검증 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X_with_text, y, test_size=0.2, random_state=42, stratify=y
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
            'feature': X_with_text.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # 모델 성능 평가
        y_pred = self.model.predict(X_test)
        
        return {
            'train_score': self.model.score(X_train, y_train),
            'test_score': self.model.score(X_test, y_test),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'feature_importance': self.feature_importance
        }
    
    def predict_case(self, case_data):
        """개별 사례 예측"""
        if self.model is None:
            return None
        
        # 특성 전처리
        processed_case = self.preprocess_features(pd.DataFrame([case_data]))
        
        # 범주형 변수 인코딩
        categorical_features = ['수입국', '보험종목', '사고유형그룹', '금액구간', '상품분류그룹명']
        
        for feature in categorical_features:
            if feature in self.label_encoders:
                try:
                    processed_case[f'{feature}_encoded'] = self.label_encoders[feature].transform(
                        processed_case[feature].fillna('Unknown')
                    )
                except ValueError:
                    # 새로운 카테고리인 경우 가장 빈번한 값으로 대체
                    processed_case[f'{feature}_encoded'] = 0
        
        # 특성 선택
        feature_columns = [f'{feature}_encoded' for feature in categorical_features]
        feature_columns.extend(['사고설명_길이', '원화사고금액'])
        
        X = processed_case[feature_columns].fillna(0)
        
        # 텍스트 특성 추가 (가능한 경우)
        if self.text_vectorizer and '사고설명' in case_data:
            try:
                text_features = self.text_vectorizer.transform([case_data['사고설명']])
                text_df = pd.DataFrame(
                    text_features.toarray(), 
                    columns=[f'text_feature_{i}' for i in range(text_features.shape[1])]
                )
                X = pd.concat([X.reset_index(drop=True), text_df], axis=1)
            except:
                pass
        
        # 예측
        prediction = self.model.predict(X)[0]
        prediction_proba = self.model.predict_proba(X)[0]
        
        # 클래스별 확률
        classes = self.model.classes_
        probabilities = {cls: prob for cls, prob in zip(classes, prediction_proba)}
        
        return {
            'prediction': prediction,
            'probabilities': probabilities,
            'confidence': max(prediction_proba)
        }
    
    def calculate_similarity_score(self, query_case, candidate_case, df_all):
        """가중치 기반 유사도 점수 계산"""
        score = 0.0
        weights = self.optimal_weights
        
        # 1. 텍스트 유사도 (KoSimCSE 또는 TF-IDF 기반)
        if 'text_similarity' in weights and '사고설명' in query_case and '사고설명' in candidate_case:
            # 간단한 텍스트 유사도 (실제로는 KoSimCSE 사용)
            query_words = set(str(query_case['사고설명']).split())
            candidate_words = set(str(candidate_case['사고설명']).split())
            
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

def create_prediction_interface(prediction_system, df):
    """예측 인터페이스"""
    st.subheader("🔮 사고 판정 예측")
    
    st.write("새로운 사고 정보를 입력하여 예상 판정을 확인해보세요.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**📋 기본 정보**")
        
        input_country = st.selectbox(
            "수입국:",
            options=df['수입국'].value_counts().head(20).index
        )
        
        input_insurance = st.selectbox(
            "보험종목:",
            options=df['보험종목'].value_counts().head(15).index
        )
        
        input_accident_type = st.selectbox(
            "사고유형:",
            options=df['사고유형명'].value_counts().head(15).index
        )
        
        input_product_group = st.selectbox(
            "상품분류그룹:",
            options=['선택안함'] + list(df['상품분류그룹명'].value_counts().head(15).index)
        )
    
    with col2:
        st.write("**💰 금액 정보**")
        
        input_amount = st.number_input(
            "사고금액 (원):",
            min_value=0,
            value=50000000,
            step=1000000,
            format="%d"
        )
        
        st.write("**📝 사고 설명**")
        input_description = st.text_area(
            "사고설명:",
            placeholder="사고의 상세한 내용을 입력해주세요...",
            height=100
        )
    
    if st.button("🎯 판정 예측 실행", type="primary"):
        # 입력 데이터 구성
        case_data = {
            '수입국': input_country,
            '보험종목': input_insurance,
            '사고유형명': input_accident_type,
            '상품분류그룹명': input_product_group if input_product_group != '선택안함' else None,
            '원화사고금액': input_amount,
            '사고설명': input_description
        }
        
        # 예측 실행
        if prediction_system.model is not None:
            prediction_result = prediction_system.predict_case(case_data)
            
            if prediction_result:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**🎯 예측 결과**")
                    
                    predicted_decision = prediction_result['prediction']
                    confidence = prediction_result['confidence']
                    
                    # 예측 결과에 따른 색상 표시
                    if predicted_decision == '지급':
                        st.success(f"예상 판정: **{predicted_decision}**")
                    elif predicted_decision == '면책':
                        st.error(f"예상 판정: **{predicted_decision}**")
                    else:
                        st.info(f"예상 판정: **{predicted_decision}**")
                    
                    st.write(f"신뢰도: **{confidence:.1%}**")
                    
                    # 신뢰도 시각적 표시
                    st.progress(confidence)
                
                with col2:
                    st.write("**📊 판정별 확률**")
                    
                    probabilities = prediction_result['probabilities']
                    prob_df = pd.DataFrame(list(probabilities.items()), columns=['판정구분', '확률'])
                    prob_df = prob_df.sort_values('확률', ascending=False)
                    
                    fig = px.bar(
                        prob_df,
                        x='확률',
                        y='판정구분',
                        orientation='h',
                        title="판정구분별 예측 확률"
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                # 유사사례 검색
                st.write("**🔍 유사한 사례들**")
                
                # 유사도 기반 상위 사례 검색
                similarities = []
                for idx, row in df.iterrows():
                    sim_score = prediction_system.calculate_similarity_score(case_data, row, df)
                    similarities.append((sim_score, row))
                
                # 상위 5개 유사사례
                similarities.sort(key=lambda x: x[0], reverse=True)
                top_similar = similarities[:5]
                
                for i, (sim_score, similar_case) in enumerate(top_similar):
                    with st.expander(f"#{i+1} 유사도 {sim_score:.3f} - {similar_case['판정구분']} ({similar_case['사고유형명']})"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**📋 사례 정보**")
                            st.write(f"• 수입국: {similar_case['수입국']}")
                            st.write(f"• 보험종목: {similar_case['보험종목']}")
                            st.write(f"• 사고금액: {similar_case['원화사고금액']:,.0f}원" if pd.notna(similar_case['원화사고금액']) else "• 사고금액: 정보없음")
                        
                        with col2:
                            st.write("**⚖️ 판정 결과**")
                            st.write(f"• 판정구분: **{similar_case['판정구분']}**")
                            st.write(f"• 판정사유: {similar_case['판정사유']}")
                            st.write(f"• 판정회차: {similar_case['판정회차']}회")
                        
                        if pd.notna(similar_case['사고설명']) and len(str(similar_case['사고설명'])) > 10:
                            st.write("**📝 사고설명**")
                            st.write(f"_{similar_case['사고설명']}_")

def create_model_analysis_dashboard(prediction_system, model_results):
    """모델 분석 대시보드"""
    st.subheader("📊 예측 모델 성능 분석")
    
    if model_results is None:
        st.warning("모델이 학습되지 않았습니다.")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("학습 정확도", f"{model_results['train_score']:.3f}")
    with col2:
        st.metric("검증 정확도", f"{model_results['test_score']:.3f}")
    with col3:
        overfitting = model_results['train_score'] - model_results['test_score']
        st.metric("과적합 정도", f"{overfitting:.3f}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 특성 중요도
        if prediction_system.feature_importance is not None:
            top_features = prediction_system.feature_importance.head(15)
            
            fig1 = px.bar(
                top_features,
                x='importance',
                y='feature',
                orientation='h',
                title="상위 15개 특성 중요도"
            )
            fig1.update_layout(height=500)
            st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # 분류 성능 히트맵
        report = model_results['classification_report']
        
        # precision, recall, f1-score 추출
        metrics_data = []
        for class_name, metrics in report.items():
            if isinstance(metrics, dict) and class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                metrics_data.append({
                    'Class': class_name,
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-Score': metrics['f1-score']
                })
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            
            fig2 = px.bar(
                metrics_df.melt(id_vars='Class', var_name='Metric', value_name='Score'),
                x='Class',
                y='Score',
                color='Metric',
                barmode='group',
                title="클래스별 성능 지표"
            )
            fig2.update_layout(height=500, xaxis_tickangle=-45)
            st.plotly_chart(fig2, use_container_width=True)
    
    # 가중치 최적화 결과
    st.write("**⚖️ 현재 유사도 가중치**")
    
    weights_df = pd.DataFrame(list(prediction_system.optimal_weights.items()), 
                             columns=['특성', '가중치'])
    
    fig3 = px.pie(
        weights_df,
        values='가중치',
        names='특성',
        title="유사도 계산 가중치 분포"
    )
    st.plotly_chart(fig3, use_container_width=True)

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

def main():
    st.title("🔮 보험사고 예측 및 유사사례 검색 시스템")
    st.markdown("""
    이 시스템은 머신러닝 모델을 활용하여 새로운 사고의 판정을 예측하고,
    가중치 기반 유사도 계산으로 관련 사례를 검색합니다.
    """)
    st.markdown("---")
    
    # 데이터 로드
    df = load_data()
    if df is None:
        return
    
    # 예측 시스템 초기화
    if 'prediction_system' not in st.session_state:
        st.session_state.prediction_system = InsurancePredictionSystem()
        st.session_state.model_results = None
    
    prediction_system = st.session_state.prediction_system
    
    # 사이드바 - 모델 설정
    st.sidebar.header("🔧 모델 설정")
    
    if st.sidebar.button("📚 모델 학습 실행"):
        with st.spinner("모델을 학습하는 중..."):
            model_results = prediction_system.train_prediction_model(df)
            st.session_state.model_results = model_results
        st.sidebar.success("모델 학습 완료!")
    
    # 가중치 조정
    st.sidebar.write("**⚖️ 유사도 가중치 조정**")
    
    text_weight = st.sidebar.slider("텍스트 유사도", 0.0, 1.0, prediction_system.optimal_weights['text_similarity'])
    accident_weight = st.sidebar.slider("사고유형", 0.0, 1.0, prediction_system.optimal_weights['accident_type'])
    country_weight = st.sidebar.slider("수입국", 0.0, 1.0, prediction_system.optimal_weights['country'])
    amount_weight = st.sidebar.slider("금액대", 0.0, 1.0, prediction_system.optimal_weights['amount_range'])
    insurance_weight = st.sidebar.slider("보험종목", 0.0, 1.0, prediction_system.optimal_weights['insurance_type'])
    
    # 가중치 정규화
    total_weight = text_weight + accident_weight + country_weight + amount_weight + insurance_weight
    if total_weight > 0:
        prediction_system.optimal_weights = {
            'text_similarity': text_weight / total_weight,
            'accident_type': accident_weight / total_weight,
            'country': country_weight / total_weight,
            'amount_range': amount_weight / total_weight,
            'insurance_type': insurance_weight / total_weight
        }
    
    # 탭 구성
    tab1, tab2 = st.tabs(["🔮 판정 예측", "📊 모델 분석"])
    
    with tab1:
        create_prediction_interface(prediction_system, df)
    
    with tab2:
        create_model_analysis_dashboard(prediction_system, st.session_state.model_results)

if __name__ == "__main__":
    main()