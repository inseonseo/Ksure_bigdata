import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from comprehensive_similarity_search import ComprehensiveSimilaritySearch
import time

# 페이지 설정
st.set_page_config(
    page_title="종합 사고 유사도 검색기",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 제목
st.title("🔍 종합 사고 유사도 검색기")
st.markdown("모든 특성을 활용한 순수 유사도 검색 시스템")

# 사이드바
st.sidebar.header("🔧 검색 설정")

# 검색기 초기화
@st.cache_resource
def load_search_engine():
    """검색기 로드 (캐시)"""
    with st.spinner("🔧 검색기 초기화 중..."):
        try:
            engine = ComprehensiveSimilaritySearch()
            return engine
        except Exception as e:
            st.error(f"검색기 초기화 실패: {str(e)}")
            return None

# 검색기 로드
search_engine = load_search_engine()

if search_engine is None:
    st.error("❌ 검색기를 로드할 수 없습니다. 데이터 파일과 모델을 확인해주세요.")
    st.stop()

# 사용 가능한 옵션 가져오기
@st.cache_data
def get_available_options():
    """사용 가능한 옵션 가져오기 (캐시)"""
    return search_engine.get_available_options()

options = get_available_options()

# 검색 폼
st.sidebar.subheader("📝 검색 조건")

# 기본 입력 필드
query = {}

# 드롭다운 필드들
dropdown_fields = ['사고유형명', '수입국', '보험종목']
for field in dropdown_fields:
    if field in options and options[field]:
        selected = st.sidebar.selectbox(
            f"{field}",
            ["선택하세요"] + options[field],
            key=f"dropdown_{field}"
        )
        if selected != "선택하세요":
            query[field] = selected

# 텍스트 입력 필드들
text_fields = ['수출자', '수입자명']
for field in text_fields:
    value = st.sidebar.text_input(f"{field}", key=f"text_{field}")
    if value.strip():
        query[field] = value

# 수치 입력 필드들
numerical_fields = ['사고금액', '결제금액', '수출보증금액', '판정금액']
for field in numerical_fields:
    value = st.sidebar.number_input(
        f"{field}",
        min_value=0.0,
        value=0.0,
        step=1000.0,
        key=f"num_{field}"
    )
    if value > 0:
        query[field] = value

# 범주형 필드들 (추가)
categorical_fields = [
    '판정진행상태', '사고진행상태', '수출보증금액통화', 
    '결제금액통화', '사고금액통화', '결제방법', 
    '결제방법설명', '결제조건', '향후결제전망',
    '판정구분', '판정사유'
]

for field in categorical_fields:
    if field in options and options[field]:
        selected = st.sidebar.selectbox(
            f"{field}",
            ["선택하세요"] + options[field],
            key=f"cat_{field}"
        )
        if selected != "선택하세요":
            query[field] = selected

# 텍스트 영역
st.sidebar.subheader("📄 상세 설명")
사고설명 = st.sidebar.text_area(
    "사고설명",
    height=100,
    placeholder="사고에 대한 상세한 설명을 입력하세요..."
)

if 사고설명.strip():
    query['사고설명'] = 사고설명

# 검색 버튼
search_button = st.sidebar.button("🔍 검색", type="primary")

# 메인 영역
if search_button:
    if not query:
        st.warning("⚠️ 검색 조건을 하나 이상 입력해주세요.")
    else:
        with st.spinner("🔍 유사 사례 검색 중..."):
            try:
                # 검색 실행
                results = search_engine.search_similar_cases(query, top_k=5, verbose=False)
                
                if not results:
                    st.warning("⚠️ 검색 결과가 없습니다. 검색 조건을 변경해보세요.")
                else:
                    # 결과 표시
                    st.success(f"✅ {len(results)}개의 유사 사례를 찾았습니다.")
                    
                    # 결과 요약
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        avg_similarity = np.mean([r['similarity'] for r in results])
                        st.metric("평균 유사도", f"{avg_similarity:.3f}")
                    
                    with col2:
                        max_similarity = max([r['similarity'] for r in results])
                        st.metric("최고 유사도", f"{max_similarity:.3f}")
                    
                    with col3:
                        min_similarity = min([r['similarity'] for r in results])
                        st.metric("최저 유사도", f"{min_similarity:.3f}")
                    
                    # 유사도 분포 차트
                    similarities = [r['similarity'] for r in results]
                    ranks = [r['rank'] for r in results]
                    
                    fig = px.bar(
                        x=ranks,
                        y=similarities,
                        title="유사도 분포",
                        labels={'x': '순위', 'y': '유사도'},
                        color=similarities,
                        color_continuous_scale='viridis'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 상세 결과
                    st.subheader("📋 상세 검색 결과")
                    
                    for i, result in enumerate(results):
                        with st.expander(f"🏆 {result['rank']}위 - 유사도: {result['similarity']:.3f}"):
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.markdown(f"**사건 ID:** {result['case_id']}")
                                st.markdown(f"**판정 요약:** {result['판정요약']}")
                                st.markdown(f"**판정 횟수:** {result['판정횟수']}회")
                                
                                # 사건 정보
                                case_info = result['case_info']
                                st.markdown("**📋 사건 정보:**")
                                info_text = f"""
                                - 사고접수일자: {case_info['사고접수일자']}
                                - 사고유형: {case_info['사고유형명']}
                                - 수입국: {case_info['수입국']}
                                - 수출자: {case_info['수출자']}
                                - 보험종목: {case_info['보험종목']}
                                """
                                st.markdown(info_text)
                            
                            with col2:
                                # 금액 정보
                                case_info = result['case_info']
                                st.markdown("**💰 금액 정보:**")
                                amount_text = f"""
                                - 사고금액: {case_info['사고금액']:,.0f}
                                - 결제금액: {case_info['결제금액']:,.0f}
                                - 수출보증금액: {case_info['수출보증금액']:,.0f}
                                - 판정금액: {case_info['판정금액']:,.0f}
                                """
                                st.markdown(amount_text)
                            
                            # 판정 과정
                            st.markdown("**🔄 판정 과정:**")
                            for j, decision in enumerate(result['decision_process']):
                                st.markdown(f"""
                                **{j+1}차 판정:**
                                - 날짜: {decision['날짜']}
                                - 판정: {decision['판정구분']}
                                - 금액: {decision['판정금액']:,.0f}
                                - 사유: {decision['판정사유']}
                                - 상태: {decision['진행상태']}
                                """)
                            
                            st.divider()
                
            except Exception as e:
                st.error(f"❌ 검색 중 오류가 발생했습니다: {str(e)}")

# 검색 히스토리 (세션 상태 사용)
if 'search_history' not in st.session_state:
    st.session_state.search_history = []

# 검색 히스토리 표시
if st.session_state.search_history:
    st.sidebar.subheader("📚 검색 히스토리")
    for i, (timestamp, query_text) in enumerate(st.session_state.search_history[-5:]):
        st.sidebar.text(f"{timestamp}: {query_text[:30]}...")

# 정보 섹션
with st.expander("ℹ️ 시스템 정보"):
    st.markdown("""
    ### 🔍 종합 유사도 검색기
    
    **특징:**
    - 모든 특성을 활용한 순수 유사도 검색
    - KoSimCSE 기반 텍스트 유사도
    - 판정구분, 판정사유 포함한 종합 검색
    - 사건별 그룹화 및 판정 과정 분석
    
    **사용된 특성:**
    - **범주형:** 판정진행상태, 사고진행상태, 수출자, 수입자, 통화, 사고유형명, 수입국, 보험종목, 결제방법, 판정구분, 판정사유 등
    - **수치형:** 사고금액, 결제금액, 수출보증금액, 판정금액
    - **텍스트:** 사고설명, 수출자, 수입자명 (KoSimCSE 임베딩)
    
    **유사도 계산:**
    - 텍스트 유사도: 60% (KoSimCSE)
    - 범주형 유사도: 30%
    - 수치형 유사도: 10%
    """)

# 푸터
st.markdown("---")
st.markdown("🔍 종합 사고 유사도 검색기 | 모든 특성을 활용한 순수 검색 시스템")
