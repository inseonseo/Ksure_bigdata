import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
from typing import List, Tuple
import re

class KoSimCSESimilaritySearch:
    def __init__(self, model_name="BM-K/KoSimCSE-roberta-multitask"):
        """KoSimCSE 모델을 사용한 유사도 검색 클래스"""
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.embeddings_cache = {}
        self.cache_file = "kosimcse_embeddings_cache.pkl"
        
    def load_model(self):
        """KoSimCSE 모델 로드"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.eval()
            return True
        except Exception as e:
            st.error(f"모델 로드 실패: {e}")
            return False
    
    def preprocess_text(self, text: str) -> str:
        """텍스트 전처리"""
        if pd.isna(text) or text == '':
            return ""
        
        # 기본적인 전처리
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
    
    def get_embeddings(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """텍스트 리스트를 임베딩으로 변환"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("모델이 로드되지 않았습니다.")
        
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # 토큰화
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # 임베딩 생성
            with torch.no_grad():
                outputs = self.model(**inputs)
                # [CLS] 토큰의 임베딩 사용
                batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
                embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
    
    def load_embeddings_cache(self) -> bool:
        """캐시된 임베딩 로드"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    self.embeddings_cache = pickle.load(f)
                return True
        except Exception as e:
            st.warning(f"캐시 로드 실패: {e}")
        return False
    
    def save_embeddings_cache(self):
        """임베딩 캐시 저장"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.embeddings_cache, f)
        except Exception as e:
            st.warning(f"캐시 저장 실패: {e}")
    
    def compute_similarity(self, query_text: str, candidate_texts: List[str], 
                          candidate_data: pd.DataFrame) -> List[Tuple[float, pd.Series]]:
        """쿼리 텍스트와 후보 텍스트들 간의 유사도 계산"""
        
        # 텍스트 전처리
        query_processed = self.preprocess_text(query_text)
        if not query_processed:
            return []
        
        candidate_processed = [self.preprocess_text(text) for text in candidate_texts]
        valid_indices = [i for i, text in enumerate(candidate_processed) if text]
        
        if not valid_indices:
            return []
        
        # 유효한 텍스트들만 선택
        valid_texts = [candidate_processed[i] for i in valid_indices]
        valid_data = candidate_data.iloc[valid_indices]
        
        # 임베딩 계산
        try:
            query_embedding = self.get_embeddings([query_processed])
            candidate_embeddings = self.get_embeddings(valid_texts)
            
            # 코사인 유사도 계산
            similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]
            
            # 결과 정렬
            results = []
            for i, similarity in enumerate(similarities):
                results.append((similarity, valid_data.iloc[i]))
            
            results.sort(key=lambda x: x[0], reverse=True)
            return results
            
        except Exception as e:
            st.error(f"유사도 계산 오류: {e}")
            return []

def create_advanced_similarity_search(df, df_meaningful):
    """고급 유사도 검색 인터페이스"""
    st.subheader("🤖 KoSimCSE 기반 고급 유사사례 검색")
    
    # 모델 초기화
    if 'similarity_model' not in st.session_state:
        st.session_state.similarity_model = KoSimCSESimilaritySearch()
    
    similarity_model = st.session_state.similarity_model
    
    # 모델 로드 상태 확인
    if similarity_model.model is None:
        with st.spinner("KoSimCSE 모델을 로드하는 중..."):
            if not similarity_model.load_model():
                st.error("모델 로드에 실패했습니다. 일반 TF-IDF 기반 검색을 사용해주세요.")
                return
        st.success("모델 로드 완료!")
    
    # 검색 인터페이스
    col1, col2 = st.columns([2, 1])
    
    with col1:
        query_text = st.text_area(
            "사고설명을 입력하세요:",
            placeholder="예: 수입자가 대금 지급을 지연하여 발생한 사고입니다. L/C 조건이었으나 수입자의 재정상황 악화로...",
            height=120
        )
    
    with col2:
        st.write("**검색 옵션:**")
        
        # 추가 필터 조건
        filter_country = st.selectbox(
            "수입국 필터:",
            ['전체'] + list(df['수입국'].value_counts().head(15).index)
        )
        
        filter_accident_type = st.selectbox(
            "사고유형 필터:",
            ['전체'] + list(df['사고유형명'].value_counts().head(10).index)
        )
        
        filter_decision = st.selectbox(
            "판정구분 필터:",
            ['전체'] + list(df['판정구분'].unique())
        )
        
        max_results = st.slider("최대 결과 수:", 5, 20, 10)
    
    if query_text and st.button("🔍 고급 검색 실행", type="primary"):
        with st.spinner("유사사례를 검색하는 중..."):
            # 필터링 적용
            filtered_df = df_meaningful.copy()
            
            if filter_country != '전체':
                filtered_df = filtered_df[filtered_df['수입국'] == filter_country]
            if filter_accident_type != '전체':
                filtered_df = filtered_df[filtered_df['사고유형명'] == filter_accident_type]
            if filter_decision != '전체':
                filtered_df = filtered_df[filtered_df['판정구분'] == filter_decision]
            
            if len(filtered_df) == 0:
                st.warning("필터 조건에 맞는 데이터가 없습니다.")
                return
            
            # 유사도 계산
            candidate_texts = filtered_df['사고설명'].tolist()
            results = similarity_model.compute_similarity(query_text, candidate_texts, filtered_df)
            
            if not results:
                st.warning("유사한 사례를 찾을 수 없습니다.")
                return
            
            # 결과 표시
            st.write(f"**🎯 상위 {min(len(results), max_results)}개 유사사례:**")
            
            # 유사도 분포 시각화
            similarities = [result[0] for result in results[:max_results]]
            
            import plotly.express as px
            fig = px.bar(
                x=list(range(1, len(similarities) + 1)),
                y=similarities,
                title="유사도 점수 분포",
                labels={'x': '순위', 'y': '유사도 점수'}
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # 상세 결과
            for i, (similarity, case) in enumerate(results[:max_results]):
                with st.expander(f"#{i+1} 유사도 {similarity:.4f} - {case['판정구분']} ({case['사고유형명']})"):
                    
                    # 유사도 시각적 표시
                    similarity_pct = similarity * 100
                    st.progress(similarity_pct / 100)
                    st.caption(f"유사도: {similarity_pct:.1f}%")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**📋 기본정보**")
                        st.write(f"• 보상파일번호: `{case['보상파일번호']}`")
                        st.write(f"• 사고번호: `{case['사고번호']}`")
                        st.write(f"• 수입국: **{case['수입국']}**")
                        st.write(f"• 보험종목: {case['보험종목']}")
                        st.write(f"• 상품분류: {case['상품분류그룹명']}")
                        
                        if pd.notna(case['원화사고금액']):
                            amount_str = f"{case['원화사고금액']:,.0f}원"
                            if case['원화사고금액'] >= 100000000:
                                amount_str += f" ({case['원화사고금액']/100000000:.1f}억원)"
                            st.write(f"• 사고금액: **{amount_str}**")
                    
                    with col2:
                        st.write("**⚖️ 판정정보**")
                        
                        # 판정구분에 따른 색상 표시
                        if case['판정구분'] == '지급':
                            st.success(f"판정구분: {case['판정구분']}")
                        elif case['판정구분'] == '면책':
                            st.error(f"판정구분: {case['판정구분']}")
                        else:
                            st.info(f"판정구분: {case['판정구분']}")
                        
                        st.write(f"• 판정사유: **{case['판정사유']}**")
                        st.write(f"• 판정회차: {case['판정회차']}회")
                        st.write(f"• 사고진행상태: {case['사고진행상태']}")
                        
                        if pd.notna(case['향후결제전망']) and case['향후결제전망'] != '판단불가':
                            st.write(f"• 향후결제전망: {case['향후결제전망']}")
                    
                    st.write("**📝 사고설명**")
                    st.markdown(f"> {case['사고설명']}")
                    
                    # 키워드 하이라이팅 (간단한 방식)
                    query_words = set(query_text.split())
                    case_words = set(case['사고설명'].split())
                    common_words = query_words.intersection(case_words)
                    
                    if common_words:
                        st.write("**🔑 공통 키워드:**")
                        st.write(" • ".join([f"`{word}`" for word in common_words if len(word) > 1]))

def create_similarity_analysis_dashboard(df):
    """유사도 분석 대시보드"""
    st.subheader("📊 유사도 검색 분석 대시보드")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 사고설명 길이별 분포
        df_text = df[df['사고설명'].notna()].copy()
        df_text['설명_길이'] = df_text['사고설명'].str.len()
        
        length_bins = [0, 20, 50, 100, 200, 500, float('inf')]
        length_labels = ['20자 미만', '20-50자', '50-100자', '100-200자', '200-500자', '500자 이상']
        df_text['길이_구간'] = pd.cut(df_text['설명_길이'], bins=length_bins, labels=length_labels)
        
        length_dist = df_text['길이_구간'].value_counts()
        
        import plotly.express as px
        fig1 = px.bar(
            x=length_dist.index,
            y=length_dist.values,
            title="사고설명 길이별 분포",
            labels={'x': '길이 구간', 'y': '건수'}
        )
        fig1.update_layout(height=400)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # 판정구분별 평균 설명 길이
        avg_length_by_decision = df_text.groupby('판정구분')['설명_길이'].mean().sort_values(ascending=False)
        
        fig2 = px.bar(
            x=avg_length_by_decision.values,
            y=avg_length_by_decision.index,
            orientation='h',
            title="판정구분별 평균 설명 길이",
            labels={'x': '평균 길이 (자)', 'y': '판정구분'}
        )
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
    
    # 텍스트 품질 분석
    st.write("**📝 텍스트 데이터 품질 분석**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        meaningful_count = len(df_text[
            (df_text['설명_길이'] > 10) &
            (~df_text['사고설명'].str.contains('설명없음|첨부파일참고|해당없음', na=False, case=False))
        ])
        st.metric("의미있는 설명", f"{meaningful_count:,}건", f"{meaningful_count/len(df)*100:.1f}%")
    
    with col2:
        short_count = len(df_text[df_text['설명_길이'] <= 10])
        st.metric("짧은 설명", f"{short_count:,}건", f"{short_count/len(df)*100:.1f}%")
    
    with col3:
        long_count = len(df_text[df_text['설명_길이'] > 100])
        st.metric("상세한 설명", f"{long_count:,}건", f"{long_count/len(df)*100:.1f}%")
    
    with col4:
        avg_length = df_text['설명_길이'].mean()
        st.metric("평균 설명 길이", f"{avg_length:.1f}자")

def main():
    """메인 실행 함수"""
    st.title("🤖 KoSimCSE 기반 고급 유사사례 검색")
    
    # 데이터 로드
    @st.cache_data
    def load_data():
        try:
            df = pd.read_csv('KoSimCSE/new/design.csv', encoding='cp949')
            
            # 의미있는 사고설명만 필터링
            df_meaningful = df[
                (df['사고설명'].notna()) & 
                (df['사고설명'].str.len() > 10) &
                (~df['사고설명'].str.contains('설명없음|첨부파일참고|해당없음', na=False, case=False))
            ].copy()
            
            return df, df_meaningful
        except Exception as e:
            st.error(f"데이터 로드 오류: {e}")
            return None, None
    
    df, df_meaningful = load_data()
    if df is None:
        return
    
    # 탭 구성
    tab1, tab2 = st.tabs(["🔍 유사사례 검색", "📊 분석 대시보드"])
    
    with tab1:
        create_advanced_similarity_search(df, df_meaningful)
    
    with tab2:
        create_similarity_analysis_dashboard(df)

if __name__ == "__main__":
    main()