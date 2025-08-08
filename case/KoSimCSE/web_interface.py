import streamlit as st
import pandas as pd
import numpy as np
from case_similarity_search import CaseSimilaritySearch
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os # Added for cache check

# 페이지 설정
st.set_page_config(
    page_title="사고 유사도 검색 시스템",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .result-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .decision-process {
        background-color: #e8f4fd;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .metric-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #b0d4f1;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_search_engine():
    """검색 엔진 로드 (캐싱)"""
    try:
        search_engine = CaseSimilaritySearch(fast_mode=False)  # KoSimCSE 사용
        return search_engine
    except Exception as e:
        st.error(f"데이터 로드 중 오류가 발생했습니다: {str(e)}")
        return None

def create_input_form(search_engine):
    """입력 폼 생성"""
    st.markdown('<div class="sub-header">📝 검색 조건 입력</div>', unsafe_allow_html=True)
    
    # 탭 생성
    tab1, tab2, tab3 = st.tabs(["🔧 간단 입력", "📄 상세 입력", "📊 고급 검색"])
    
    with tab1:
        st.markdown("**기본 정보만 입력하여 빠르게 검색**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 드롭다운 필드들
            options = search_engine.get_available_options()
            
            사고유형명 = st.selectbox(
                "사고유형명",
                options=[''] + options.get('사고유형명', []),
                help="사고의 유형을 선택하세요"
            )
            
            수입국 = st.selectbox(
                "수입국",
                options=[''] + options.get('수입국', []),
                help="수입국을 선택하세요"
            )
            
            보험종목 = st.selectbox(
                "보험종목",
                options=[''] + options.get('보험종목', []),
                help="보험 종목을 선택하세요"
            )
        
        with col2:
            # 텍스트 입력 필드들
            수출자 = st.text_input(
                "수출자",
                help="수출자명을 입력하세요"
            )
            
            사고설명 = st.text_area(
                "사고설명 (간단)",
                height=100,
                help="사고 내용을 간단히 설명하세요"
            )
    
    with tab2:
        st.markdown("**상세한 정보를 입력하여 정확한 검색**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            사고유형명 = st.selectbox(
                "사고유형명",
                options=[''] + options.get('사고유형명', []),
                key="tab2_사고유형명"
            )
            
            수입국 = st.selectbox(
                "수입국",
                options=[''] + options.get('수입국', []),
                key="tab2_수입국"
            )
            
            보험종목 = st.selectbox(
                "보험종목",
                options=[''] + options.get('보험종목', []),
                key="tab2_보험종목"
            )
            
            수출자 = st.text_input(
                "수출자",
                key="tab2_수출자"
            )
        
        with col2:
            사고금액 = st.number_input(
                "사고금액 (원)",
                min_value=0,
                value=0,
                step=1000000,
                help="사고 금액을 입력하세요"
            )
            
            결제금액 = st.number_input(
                "결제금액 (원)",
                min_value=0,
                value=0,
                step=1000000,
                help="결제 금액을 입력하세요"
            )
        
        사고설명 = st.text_area(
            "사고설명 (상세)",
            height=150,
            key="tab2_사고설명",
            help="사고의 상세한 내용을 입력하세요"
        )
        
        사고경위 = st.text_area(
            "사고경위",
            height=100,
            help="사고가 발생한 경위를 설명하세요"
        )
    
    with tab3:
        st.markdown("**고급 검색 옵션**")
        
        # 검색 옵션
        col1, col2 = st.columns(2)
        
        with col1:
            top_k = st.slider(
                "검색 결과 수",
                min_value=1,
                max_value=20,
                value=5,
                help="검색할 유사 사례의 수를 선택하세요"
            )
            
            min_similarity = st.slider(
                "최소 유사도",
                min_value=0.0,
                max_value=1.0,
                value=0.1,  # 기본값을 0.3에서 0.1로 낮춤
                step=0.05,
                help="최소 유사도 임계값을 설정하세요 (낮을수록 더 많은 결과)"
            )
        
        with col2:
            # 필터 옵션
            st.markdown("**필터 옵션**")
            
            최소판정횟수 = st.number_input(
                "최소 판정 횟수",
                min_value=1,
                value=1,
                help="최소 판정 횟수를 설정하세요"
            )
    
    # 검색 버튼
    if st.button("🔍 유사 사례 검색", type="primary", use_container_width=True):
        # 쿼리 구성
        query = {}
        
        if 사고유형명:
            query['사고유형명'] = 사고유형명
        if 수입국:
            query['수입국'] = 수입국
        if 보험종목:
            query['보험종목'] = 보험종목
        if 수출자:
            query['수출자'] = 수출자
        if 사고설명:
            query['사고설명'] = 사고설명
        if 사고경위:
            query['사고경위'] = 사고경위
        if '사고금액' in locals() and 사고금액 > 0:
            query['사고금액'] = 사고금액
        if '결제금액' in locals() and 결제금액 > 0:
            query['결제금액'] = 결제금액
        
        return query, top_k, min_similarity, 최소판정횟수
    
    return None, 5, 0.1, 1

def display_results(results, search_engine):
    """검색 결과 표시"""
    if not results:
        st.warning("검색 결과가 없습니다. 검색 조건을 변경해보세요.")
        return
    
    st.markdown('<div class="sub-header">📊 검색 결과</div>', unsafe_allow_html=True)
    
    # 전체 통계
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("총 검색 결과", len(results))
    
    with col2:
        avg_similarity = np.mean([r['similarity'] for r in results])
        st.metric("평균 유사도", f"{avg_similarity:.3f}")
    
    with col3:
        # 패턴 분석은 제거 (아직 정교하지 않음)
        st.metric("최고 유사도", f"{max([r['similarity'] for r in results]):.3f}")
    
    with col4:
        avg_decisions = np.mean([r['판정횟수'] for r in results])
        st.metric("평균 판정 횟수", f"{avg_decisions:.1f}")
    
    # 유사도 분포 차트
    st.markdown("**📈 유사도 분포**")
    similarities = [r['similarity'] for r in results]
    
    fig = px.histogram(
        x=similarities,
        title="검색 결과 유사도 분포",
        labels={'x': '유사도', 'y': '빈도'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 개별 결과 표시
    for i, result in enumerate(results):
        with st.expander(f"🏆 {result['rank']}위 - {result['case_id']} (유사도: {result['similarity']:.3f})"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("**📋 사건 정보**")
                case_info = result['case_info']
                
                info_df = pd.DataFrame([
                    ["사고접수일자", case_info['사고접수일자']],
                    ["사고유형명", case_info['사고유형명']],
                    ["수입국", case_info['수입국']],
                    ["수출자", case_info['수출자']],
                    ["보험종목", case_info['보험종목']],
                    ["사고금액", f"{case_info['사고금액']:,.0f}원"],
                    ["결제금액", f"{case_info['결제금액']:,.0f}원"],
                    ["수입자명", case_info['수입자명']]
                ], columns=["항목", "내용"])
                
                st.dataframe(info_df, use_container_width=True)
            
            with col2:
                st.markdown("**🎯 예측 결과**")
                predicted = result.get('predicted_results', {})
                st.markdown(f"""
                - **판정구분**: {predicted.get('판정구분', 'N/A')}
                - **판정사유**: {predicted.get('판정사유', 'N/A')}
                - **판정횟수**: {result['판정횟수']}회
                """)
            
            # 판정 과정 표시
            st.markdown("**🔄 판정 과정**")
            for j, decision in enumerate(result['decision_process']):
                with st.container():
                    st.markdown(f"""
                    <div class="decision-process">
                        <strong>{j+1}차 판정</strong><br>
                        📅 날짜: {decision['날짜']}<br>
                        🎯 판정: {decision['판정구분']}<br>
                        💰 금액: {decision['판정금액']:,.0f}원<br>
                        📝 사유: {decision['판정사유']}<br>
                        📊 상태: {decision['진행상태']}
                    </div>
                    """, unsafe_allow_html=True)

def main():
    st.title("🔍 사고 유사도 검색기")
    st.markdown("KoSimCSE 기반 텍스트 유사도 분석으로 정확한 유사 사례를 찾아보세요")
    
    # 사이드바에 시스템 정보 표시
    with st.sidebar:
        st.markdown("### 📊 시스템 정보")
        st.markdown("**KoSimCSE 모델**: 사고설명 맥락 분석")
        st.markdown("**데이터**: testy.csv")
        st.markdown("**검색 방식**: 다중 특성 유사도")
        
        # 캐시 상태 확인
        cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kosimcse_embeddings_cache.pkl')
        cache_exists = os.path.exists(cache_path)
        if cache_exists:
            st.success("✅ 임베딩 캐시 준비됨")
        else:
            st.warning("⚠️ 첫 실행 시 임베딩 생성 필요")
    
    # 검색기 로드 (캐싱 사용)
    search_engine = load_search_engine()
    if search_engine is None:
        st.error("❌ 검색기 초기화 실패")
        return
    else:
        st.success("✅ 검색기 초기화 완료")
    
    # 입력 폼
    search_params = create_input_form(search_engine)
    
    if search_params[0] is not None:
        query, top_k, min_similarity, 최소판정횟수 = search_params
        
        # 검색 실행
        with st.spinner("🔍 유사 사례를 검색하고 있습니다..."):
            results = search_engine.search_similar_cases(query, top_k=top_k, verbose=False)
        
        # 필터 적용 (패턴 필터 제거)
        results = [r for r in results if r['판정횟수'] >= 최소판정횟수]
        results = [r for r in results if r['similarity'] >= min_similarity]
        
        # 결과 표시
        display_results(results, search_engine)

if __name__ == "__main__":
    main() 