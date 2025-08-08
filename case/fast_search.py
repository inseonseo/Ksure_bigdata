import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import re
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict
import pickle
import os

class FastSimilarCaseSearch:
    def __init__(self, csv_path):
        """빠른 유사사례 검색기 초기화"""
        self.csv_path = csv_path
        self.cache_file = 'similarity_cache.pkl'
        self.load_or_create_cache()
    
    def load_or_create_cache(self):
        """캐시 파일이 있으면 로드, 없으면 새로 생성"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.df = cache_data['df']
                    self.vectorizer = cache_data['vectorizer']
                    self.tfidf_matrix = cache_data['tfidf_matrix']
                st.success("✅ 캐시에서 데이터 로드 완료!")
            except:
                self.create_cache()
        else:
            self.create_cache()
    
    def create_cache(self):
        """캐시 생성"""
        with st.spinner("데이터 로딩 중... (처음 실행시 시간이 걸립니다)"):
            # 데이터 로드 (low_memory=False로 경고 제거)
            self.df = pd.read_csv(self.csv_path, encoding='cp949', low_memory=False)
            self.preprocess_data()
            self.create_similarity_matrix()
            
            # 캐시 저장
            cache_data = {
                'df': self.df,
                'vectorizer': self.vectorizer,
                'tfidf_matrix': self.tfidf_matrix
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            st.success("✅ 캐시 생성 완료!")
    
    def preprocess_data(self):
        """데이터 전처리 (최적화)"""
        # 컬럼명 정리
        self.df.columns = self.df.columns.str.strip()
        
        # 결측값 처리
        self.df = self.df.fillna('')
        
        # 텍스트 데이터 정리 (필요한 컬럼만)
        text_columns = ['사고설명', '상품분류명', '사고유형명']
        for col in text_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.strip()
        
        # 금액 데이터 정리
        amount_columns = ['원화사고금액', '원화보험금액']
        for col in amount_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)
    
    def create_similarity_matrix(self):
        """유사도 행렬 생성 (개선된 버전)"""
        # 텍스트 데이터 결합 (더 풍부한 정보 포함)
        text_data = []
        for idx, row in self.df.iterrows():
            # 사고설명을 우선적으로, 상품분류와 사고유형도 포함
            description = str(row.get('사고설명', '')).strip()
            product_category = str(row.get('상품분류명', '')).strip()
            case_type = str(row.get('사고유형명', '')).strip()
            product_name = str(row.get('상품명', '')).strip()
            
            # 사고설명이 있는 경우 우선, 없으면 다른 정보 활용
            if description and description != 'nan':
                text = f"{description} {product_category} {case_type}"
            else:
                text = f"{product_category} {case_type} {product_name}"
            
            text_data.append(text)
        
        # TF-IDF 벡터화 (더 정확한 설정)
        self.vectorizer = TfidfVectorizer(
            max_features=2000,  # 특성 수 더 증가
            stop_words=None,
            ngram_range=(1, 4),  # n-gram 범위 더 증가
            min_df=1,  # 최소 문서 빈도 감소
            max_df=0.98,  # 더 관대하게
            analyzer='word',
            token_pattern=r'(?u)\b\w+\b'  # 한글 포함
        )
        
        self.tfidf_matrix = self.vectorizer.fit_transform(text_data)
    
    def search_similar_cases_by_category(self, query_text, currency_code=None, amount=None, 
                                       payment_method=None, top_k_per_category=1):
        """심사항목별로 유사사례 검색 (최적화)"""
        # 쿼리 텍스트 벡터화
        query_vector = self.vectorizer.transform([query_text])
        
        # 유사도 계산
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # 심사항목별로 그룹화
        category_results = defaultdict(list)
        
        # 상위 500개만 처리 (더 많은 후보)
        top_indices = np.argsort(similarities)[-500:][::-1]
        
        for idx in top_indices:
            row = self.df.iloc[idx]
            category = row.get('심사항목명', '기타')
            similarity = similarities[idx]
            
            # 필터링 조건 적용
            if currency_code and str(row.get('사고통화코드_ZZ067', '')).strip() != str(currency_code).strip():
                continue
                
            if amount and amount > 0:
                case_amount = row.get('원화사고금액', 0)
                if case_amount > 0:
                    # 금액 범위 필터링 (더 관대하게 - 100% 이내)
                    amount_diff = abs(case_amount - amount) / amount
                    if amount_diff > 1.0:
                        continue
            
            # 결제방법 필터링 제거
            pass
            
            # 사고설명 길이를 고려한 가중치 계산
            description = str(row.get('사고설명', '')).strip()
            description_length = len(description) if description and description != 'nan' else 0
            
            # 키워드 매칭 보너스 (쿼리와 사고설명에 공통 키워드가 있으면 보너스)
            query_lower = query_text.lower()
            desc_lower = description.lower()
            keyword_bonus = 0
            
            # 주요 키워드들로 매칭 확인
            keywords = query_lower.split()
            for keyword in keywords:
                if len(keyword) > 2 and keyword in desc_lower:  # 2글자 이상 키워드만
                    keyword_bonus += 0.05
            
            # 사고설명이 길수록 유사도에 보너스 점수 추가 (최대 0.1)
            description_bonus = min(description_length / 1000, 0.1) if description_length > 50 else 0
            adjusted_similarity = similarity + description_bonus + keyword_bonus
            
            result = {
                'index': idx,
                'similarity': similarity,
                'adjusted_similarity': adjusted_similarity,
                '보상파일번호': row.get('보상파일번호_x', ''),
                '사고번호': row.get('사고번호', ''),
                '심사항목명': category,
                '사고설명': description,
                '원화사고금액': row.get('원화사고금액', 0),
                '수출상품코드': row.get('수출보험계약번호', ''),
                '사고유형': row.get('사고유형명', ''),
                '업종한글명': row.get('상품분류명', ''),
                '사고통화코드': row.get('사고통화코드_ZZ067', ''),
                '결제방법코드': row.get('결제방법', ''),
                '원화보험금액': row.get('원화보험금액', 0),
                '부보율': row.get('부보율', ''),
                '보험성립일자': row.get('보험성립일자', ''),
                '사고접수일자': row.get('사고접수일자', '')
            }
            
            category_results[category].append(result)
        
        # 각 카테고리별로 조정된 유사도 순으로 정렬하고 상위 결과만 선택
        final_results = {}
        for category, results in category_results.items():
            results.sort(key=lambda x: x['adjusted_similarity'], reverse=True)
            # 유사도 임계값을 더 낮춤 (0.01 이상)
            filtered_results = [r for r in results if r['similarity'] >= 0.01]
            final_results[category] = filtered_results[:top_k_per_category]
        
        return final_results
    
    def get_statistics_by_category(self, category_column='심사항목명'):
        """심사항목별 통계"""
        if category_column not in self.df.columns:
            return None
        
        stats = self.df[category_column].value_counts()
        return stats
    
    def format_amount(self, amount):
        """금액 포맷팅"""
        if pd.isna(amount) or amount == 0:
            return "0원"
        return f"{amount:,.0f}원"
    
    def format_date(self, date_value):
        """날짜 포맷팅"""
        if pd.isna(date_value) or date_value == '':
            return "정보 없음"
        try:
            if isinstance(date_value, str):
                # YYYYMMDD 형식으로 가정
                if len(str(date_value)) == 8:
                    return f"{str(date_value)[:4]}-{str(date_value)[4:6]}-{str(date_value)[6:8]}"
            return str(date_value)
        except:
            return str(date_value)

def main():
    st.set_page_config(
        page_title="유사사례 검색",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 유사사례 검색 (빠른 버전)")
    st.markdown("---")
    
    # 데이터 로드
    try:
        search_engine = FastSimilarCaseSearch('Data/case.csv')
    except Exception as e:
        st.error(f"❌ 데이터 로드 실패: {str(e)}")
        return
    
    # 사이드바 - 검색 조건
    st.sidebar.header("🔍 검색 조건")
    
    # 텍스트 검색
    query_text = st.sidebar.text_area(
        "사고 설명 또는 키워드",
        placeholder="사고 내용을 자세히 입력해주세요...",
        height=100
    )
    
    # 통화 코드
    currency_options = [''] + sorted(list(search_engine.df['사고통화코드_ZZ067'].unique()))
    selected_currency = st.sidebar.selectbox("통화 코드", currency_options)
    
    # 금액 범위
    amount = st.sidebar.number_input(
        "사고 금액 (원)",
        min_value=0,
        value=0,
        step=1000000
    )
    
    # 결제 방법 (제거됨)
    selected_payment = None
    
    # 카테고리별 결과 수
    results_per_category = st.sidebar.slider(
        "심사항목별 결과 수",
        min_value=1,
        max_value=5,
        value=1
    )
    
    # 검색 버튼
    search_button = st.sidebar.button("🔍 유사사례 검색", type="primary")
    
    # 메인 영역
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 검색 결과 (심사항목별)")
        
        if search_button and query_text.strip():
            with st.spinner("검색 중..."):
                results = search_engine.search_similar_cases_by_category(
                    query_text=query_text,
                    currency_code=selected_currency if selected_currency else None,
                    amount=amount if amount > 0 else None,
                    payment_method=selected_payment if selected_payment else None,
                    top_k_per_category=results_per_category
                )
            
            if results:
                total_cases = sum(len(cases) for cases in results.values())
                st.success(f"✅ {len(results)}개 심사항목에서 총 {total_cases}개의 유사사례를 찾았습니다!")
                
                # 심사항목별로 탭 생성
                if len(results) > 1:
                    tab_names = list(results.keys())
                    tabs = st.tabs(tab_names)
                    
                    for i, (category, cases) in enumerate(results.items()):
                        with tabs[i]:
                            st.write(f"**{category}** 카테고리에서 찾은 유사사례:")
                            
                            for j, case in enumerate(cases, 1):
                                with st.expander(f"사례 {j}: 유사도 {case['adjusted_similarity']:.3f} (원본: {case['similarity']:.3f})"):
                                    col_a, col_b = st.columns(2)
                                    
                                    with col_a:
                                        st.write(f"**보상파일번호:** {case['보상파일번호']}")
                                        st.write(f"**사고번호:** {case['사고번호']}")
                                        st.write(f"**심사항목명:** {case['심사항목명']}")
                                        st.write(f"**사고금액:** {search_engine.format_amount(case['원화사고금액'])}")
                                        st.write(f"**보험금액:** {search_engine.format_amount(case['원화보험금액'])}")
                                        st.write(f"**통화코드:** {case['사고통화코드']}")
                                    
                                    with col_b:
                                        st.write(f"**수출상품코드:** {case['수출상품코드']}")
                                        st.write(f"**사고유형:** {case['사고유형']}")
                                        st.write(f"**업종:** {case['업종한글명']}")
                                        st.write(f"**결제방법:** {case['결제방법코드']}")
                                        st.write(f"**부보율:** {case['부보율']}%")
                                        st.write(f"**보험성립일:** {search_engine.format_date(case['보험성립일자'])}")
                                    
                                    st.write("**사고설명:**")
                                    st.text_area(
                                        f"사고설명_{category}_{j}",
                                        value=case['사고설명'],
                                        height=100,
                                        disabled=True,
                                        key=f"desc_{category}_{j}"
                                    )
                else:
                    # 단일 카테고리인 경우
                    category, cases = list(results.items())[0]
                    st.write(f"**{category}** 카테고리에서 찾은 유사사례:")
                    
                    for j, case in enumerate(cases, 1):
                        with st.expander(f"사례 {j}: 유사도 {case['adjusted_similarity']:.3f} (원본: {case['similarity']:.3f})"):
                            col_a, col_b = st.columns(2)
                            
                            with col_a:
                                st.write(f"**보상파일번호:** {case['보상파일번호']}")
                                st.write(f"**사고번호:** {case['사고번호']}")
                                st.write(f"**심사항목명:** {case['심사항목명']}")
                                st.write(f"**사고금액:** {search_engine.format_amount(case['원화사고금액'])}")
                                st.write(f"**보험금액:** {search_engine.format_amount(case['원화보험금액'])}")
                                st.write(f"**통화코드:** {case['사고통화코드']}")
                            
                            with col_b:
                                st.write(f"**수출상품코드:** {case['수출상품코드']}")
                                st.write(f"**사고유형:** {case['사고유형']}")
                                st.write(f"**업종:** {case['업종한글명']}")
                                st.write(f"**결제방법:** {case['결제방법코드']}")
                                st.write(f"**부보율:** {case['부보율']}%")
                                st.write(f"**보험성립일:** {search_engine.format_date(case['보험성립일자'])}")
                            
                            st.write("**사고설명:**")
                            st.text_area(
                                f"사고설명_{category}_{j}",
                                value=case['사고설명'],
                                height=100,
                                disabled=True,
                                key=f"desc_{category}_{j}"
                            )
            else:
                st.warning("⚠️ 검색 조건에 맞는 유사사례를 찾을 수 없습니다.")
        elif search_button:
            st.warning("⚠️ 검색할 텍스트를 입력해주세요.")
    
    with col2:
        st.subheader("📈 통계 정보")
        
        # 심사항목별 통계
        stats = search_engine.get_statistics_by_category()
        if stats is not None:
            st.write("**심사항목별 사례 수:**")
            for category, count in stats.head(10).items():
                st.write(f"• {category}: {count}건")
            
            # 차트 생성
            fig = px.pie(
                values=stats.head(10).values,
                names=stats.head(10).index,
                title="심사항목별 분포"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 전체 통계
        st.write("**전체 통계:**")
        st.write(f"• 총 사례 수: {len(search_engine.df):,}건")
        st.write(f"• 평균 사고금액: {search_engine.format_amount(search_engine.df['원화사고금액'].mean())}")
        st.write(f"• 최대 사고금액: {search_engine.format_amount(search_engine.df['원화사고금액'].max())}")
        
        # 통화별 통계
        currency_stats = search_engine.df['사고통화코드_ZZ067'].value_counts()
        st.write("**통화별 분포:**")
        for currency, count in currency_stats.head(5).items():
            st.write(f"• {currency}: {count}건")

if __name__ == "__main__":
    main() 