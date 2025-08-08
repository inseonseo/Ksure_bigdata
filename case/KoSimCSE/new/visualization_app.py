import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import pycountry

# 페이지 설정
st.set_page_config(
    page_title="보상 프로세스 분석",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 캐시된 데이터 로드 함수
@st.cache_data
def load_data():
    """데이터 로드 및 전처리"""
    try:
        df = pd.read_csv('data/design.csv', encoding='cp949')
        
        # 날짜 컬럼 처리
        date_columns = ['판정일', '판정결재일', '사고접수일자', '보험금청구일']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # 금액 컬럼을 숫자로 변환
        amount_columns = ['원화사고금액', '원화판정금액', '미화사고금액', '미화판정금액', '수출보증금액']
        for col in amount_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"데이터 로드 오류: {e}")
        return None

def create_country_analysis(df, analysis_info=None):
    """수입국별 사고 분석"""
    st.subheader("🌍 수입국별 사고 현황")
    
    # 분석 기준 정보 표시
    if analysis_info and analysis_info.get('date_basis') and analysis_info.get('date_range'):
        st.caption(f"📅 분석 기준일자: `{analysis_info['date_basis']}` / 범위: {analysis_info['date_range'][0]} ~ {analysis_info['date_range'][1]}")
    
    # 탭 생성
    tab1, tab2, tab3 = st.tabs(["📊 차트 분석", "🗺️ 세계지도", "📋 상세 통계"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # 상위 15개 수입국별 사고 건수
            country_counts = df['수입국'].value_counts().head(15)
            fig1 = px.bar(
                x=country_counts.values,
                y=country_counts.index,
                orientation='h',
                title="상위 15개국 사고 발생 건수",
                labels={'x': '사고 건수', 'y': '수입국'},
                color=country_counts.values,
                color_continuous_scale='viridis'
            )
            fig1.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # 수입국별 판정구분 비율
            country_decision = df.groupby('수입국')['판정구분'].value_counts().unstack(fill_value=0)
            top_countries = df['수입국'].value_counts().head(10).index
            country_decision_top = country_decision.loc[top_countries]
            
            fig2 = px.bar(
                country_decision_top,
                title="상위 10개국 판정구분 현황",
                labels={'value': '건수', 'index': '수입국'},
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig2.update_layout(height=500, xaxis_tickangle=-45)
            st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        create_world_map(df)
    
    with tab3:
        create_country_detailed_stats(df)

def create_world_map(df):
    """세계지도 시각화"""
    st.subheader("🗺️ 세계지도 - 수입국별 사고 현황")
    
    # 국가명 매핑 함수
    def get_country_code(country_name):
        """국가명을 ISO 코드로 변환"""
        country_mapping = {
            '미국': 'USA', '중국': 'CHN', '일본': 'JPN', '독일': 'DEU', '영국': 'GBR',
            '프랑스': 'FRA', '이탈리아': 'ITA', '캐나다': 'CAN', '호주': 'AUS', '한국': 'KOR',
            '인도': 'IND', '브라질': 'BRA', '러시아': 'RUS', '스페인': 'ESP', '멕시코': 'MEX',
            '네덜란드': 'NLD', '스위스': 'CHE', '스웨덴': 'SWE', '벨기에': 'BEL', '노르웨이': 'NOR',
            '덴마크': 'DNK', '폴란드': 'POL', '오스트리아': 'AUT', '핀란드': 'FIN', '포르투갈': 'PRT',
            '그리스': 'GRC', '체코': 'CZE', '헝가리': 'HUN', '아일랜드': 'IRL', '뉴질랜드': 'NZL',
            '싱가포르': 'SGP', '홍콩': 'HKG', '대만': 'TWN', '태국': 'THA', '말레이시아': 'MYS',
            '인도네시아': 'IDN', '필리핀': 'PHL', '베트남': 'VNM', '터키': 'TUR', '이스라엘': 'ISR',
            '아랍에미리트': 'ARE', '사우디아라비아': 'SAU', '이집트': 'EGY', '남아프리카': 'ZAF',
            '아르헨티나': 'ARG', '칠레': 'CHL', '콜롬비아': 'COL', '페루': 'PER', '베네수엘라': 'VEN'
        }
        return country_mapping.get(country_name, country_name)
    
    # 국가별 통계 계산
    country_stats = df.groupby('수입국').agg({
        '보상파일번호': 'count',  # 사고 건수
        '원화사고금액': 'sum',    # 총 사고금액
        '판정구분': lambda x: (x == '지급').sum()  # 지급 건수
    }).reset_index()
    
    country_stats.columns = ['국가', '사고건수', '총사고금액', '지급건수']
    country_stats['지급비율'] = (country_stats['지급건수'] / country_stats['사고건수'] * 100).round(1)
    country_stats['국가코드'] = country_stats['국가'].apply(get_country_code)
    
    # 지도 시각화 옵션
    map_option = st.selectbox(
        "지도에 표시할 지표 선택",
        ["사고건수", "총사고금액", "지급비율"],
        help="지도에서 확인하고 싶은 지표를 선택하세요"
    )
    
    if map_option == "사고건수":
        color_column = '사고건수'
        title = "수입국별 사고 발생 건수"
        color_scale = 'viridis'
    elif map_option == "총사고금액":
        color_column = '총사고금액'
        title = "수입국별 총 사고금액 (원화)"
        color_scale = 'reds'
    else:  # 지급비율
        color_column = '지급비율'
        title = "수입국별 지급 비율 (%)"
        color_scale = 'blues'
    
    # 세계지도 생성
    fig = px.choropleth(
        country_stats,
        locations='국가코드',
        color=color_column,
        hover_name='국가',
        hover_data={
            '사고건수': True,
            '총사고금액': True,
            '지급건수': True,
            '지급비율': True,
            '국가코드': False
        },
        title=title,
        color_continuous_scale=color_scale,
        projection='natural earth'
    )
    
    fig.update_layout(
        height=600,
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor='lightgray',
            showland=True,
            landcolor='lightgray',
            showocean=True,
            oceancolor='lightblue',
            showlakes=True,
            lakecolor='lightblue',
            showrivers=True,
            rivercolor='lightblue'
        ),
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 지도 상호작용 안내
    st.info("💡 **지도 사용법**: 국가를 클릭하면 상세 정보를 확인할 수 있습니다. 마우스 휠로 확대/축소가 가능합니다.")
    
    # 상위 국가 요약
    st.subheader("📊 상위 10개국 요약")
    top_countries = country_stats.nlargest(10, color_column)
    st.dataframe(
        top_countries[['국가', '사고건수', '총사고금액', '지급건수', '지급비율']].round(2),
        use_container_width=True
    )

def create_country_detailed_stats(df):
    """국가별 상세 통계"""
    st.subheader("📋 국가별 상세 통계")
    
    # 국가별 상세 분석
    country_detailed = df.groupby('수입국').agg({
        '보상파일번호': 'count',
        '원화사고금액': ['sum', 'mean', 'std'],
        '판정구분': lambda x: (x == '지급').sum(),
        '사고유형명': 'nunique',
        '보험종목': 'nunique'
    }).round(2)
    
    country_detailed.columns = ['사고건수', '총사고금액', '평균사고금액', '사고금액표준편차', '지급건수', '사고유형수', '보험종목수']
    country_detailed['지급비율'] = (country_detailed['지급건수'] / country_detailed['사고건수'] * 100).round(1)
    country_detailed = country_detailed.sort_values('사고건수', ascending=False)
    
    # 필터링 옵션
    min_cases = st.slider("최소 사고건수", 1, int(country_detailed['사고건수'].max()), 1)
    filtered_stats = country_detailed[country_detailed['사고건수'] >= min_cases]
    
    st.dataframe(filtered_stats, use_container_width=True)
    
    # 국가별 사고유형 분석
    st.subheader("🔍 국가별 주요 사고유형")
    top_countries = df['수입국'].value_counts().head(5).index
    
    for country in top_countries:
        country_data = df[df['수입국'] == country]
        accident_types = country_data['사고유형명'].value_counts().head(3)
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric(f"{country} 총 사고건수", f"{len(country_data):,}건")
        with col2:
            st.write("**주요 사고유형:**")
            for acc_type, count in accident_types.items():
                st.write(f"• {acc_type}: {count}건")
        st.divider()

def create_accident_type_analysis(df, analysis_info=None):
    """사고유형별 분석"""
    st.subheader("⚠️ 사고유형별 현황")
    
    # 분석 기준 정보 표시
    if analysis_info and analysis_info.get('date_basis') and analysis_info.get('date_range'):
        st.caption(f"📅 분석 기준일자: `{analysis_info['date_basis']}` / 범위: {analysis_info['date_range'][0]} ~ {analysis_info['date_range'][1]}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 사고유형별 건수 (상위 10개)
        accident_types = df['사고유형명'].value_counts().head(10)
        fig1 = px.pie(
            values=accident_types.values,
            names=accident_types.index,
            title="주요 사고유형 분포",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig1.update_traces(textposition='inside', textinfo='percent+label')
        fig1.update_layout(height=500)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # 사고유형별 판정구분 히트맵
        accident_decision = df.groupby(['사고유형명', '판정구분']).size().unstack(fill_value=0)
        top_accidents = df['사고유형명'].value_counts().head(8).index
        accident_decision_top = accident_decision.loc[top_accidents]
        
        # 비율로 변환
        accident_decision_pct = accident_decision_top.div(accident_decision_top.sum(axis=1), axis=0) * 100
        
        fig2 = px.imshow(
            accident_decision_pct.values,
            x=accident_decision_pct.columns,
            y=accident_decision_pct.index,
            title="사고유형별 판정구분 비율 (%)",
            color_continuous_scale='RdYlBu_r',
            aspect='auto'
        )
        fig2.update_layout(height=500, xaxis_tickangle=-45)
        st.plotly_chart(fig2, use_container_width=True)

def create_amount_analysis(df, analysis_info=None):
    """금액별 분석"""
    st.subheader("💰 사고금액별 현황")
    
    # 분석 기준 정보 표시
    if analysis_info and analysis_info.get('date_basis') and analysis_info.get('date_range'):
        st.caption(f"📅 분석 기준일자: `{analysis_info['date_basis']}` / 범위: {analysis_info['date_range'][0]} ~ {analysis_info['date_range'][1]}")
    
    # 원화사고금액 기준으로 분석
    df_amount = df[df['원화사고금액'].notna() & (df['원화사고금액'] > 0)].copy()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 사고금액 구간별 분포
        df_amount['금액구간'] = pd.cut(
            df_amount['원화사고금액'],
            bins=[0, 10000000, 50000000, 100000000, 500000000, 1000000000, float('inf')],
            labels=['1천만원 미만', '1천만-5천만원', '5천만-1억원', '1억-5억원', '5억-10억원', '10억원 이상']
        )
        
        amount_dist = df_amount['금액구간'].value_counts().sort_index()
        fig1 = px.bar(
            x=amount_dist.index,
            y=amount_dist.values,
            title="사고금액 구간별 분포",
            labels={'x': '금액구간', 'y': '건수'},
            color=amount_dist.values,
            color_continuous_scale='blues'
        )
        fig1.update_layout(height=400, xaxis_tickangle=-45, showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # 금액구간별 판정구분
        amount_decision = df_amount.groupby(['금액구간', '판정구분']).size().unstack(fill_value=0)
        
        fig2 = px.bar(
            amount_decision,
            title="금액구간별 판정구분 현황",
            labels={'value': '건수', 'index': '금액구간'},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig2.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig2, use_container_width=True)
    
    # 금액 통계 요약
    col3, col4, col5, col6 = st.columns(4)
    with col3:
        st.metric("전체 사고건수", f"{len(df_amount):,}건")
    with col4:
        st.metric("평균 사고금액", f"{df_amount['원화사고금액'].mean()/100000000:.1f}억원")
    with col5:
        st.metric("최대 사고금액", f"{df_amount['원화사고금액'].max()/100000000:.1f}억원")
    with col6:
        st.metric("총 사고금액", f"{df_amount['원화사고금액'].sum()/1000000000000:.1f}조원")

def create_insurance_product_analysis(df, analysis_info=None):
    """보험종목별 분석"""
    st.subheader("📋 보험종목별 현황")
    
    # 분석 기준 정보 표시
    if analysis_info and analysis_info.get('date_basis') and analysis_info.get('date_range'):
        st.caption(f"📅 분석 기준일자: `{analysis_info['date_basis']}` / 범위: {analysis_info['date_range'][0]} ~ {analysis_info['date_range'][1]}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 보험종목별 건수
        insurance_counts = df['보험종목'].value_counts().head(10)
        fig1 = px.bar(
            x=insurance_counts.values,
            y=insurance_counts.index,
            orientation='h',
            title="보험종목별 사고 건수",
            labels={'x': '사고 건수', 'y': '보험종목'},
            color=insurance_counts.values,
            color_continuous_scale='plasma'
        )
        fig1.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # 보험종목별 평균 사고금액
        df_amount = df[df['원화사고금액'].notna() & (df['원화사고금액'] > 0)]
        avg_amount_by_insurance = df_amount.groupby('보험종목')['원화사고금액'].mean().sort_values(ascending=False).head(10)
        
        fig2 = px.bar(
            x=avg_amount_by_insurance.values / 100000000,  # 억원 단위
            y=avg_amount_by_insurance.index,
            orientation='h',
            title="보험종목별 평균 사고금액 (억원)",
            labels={'x': '평균 사고금액 (억원)', 'y': '보험종목'},
            color=avg_amount_by_insurance.values,
            color_continuous_scale='reds'
        )
        fig2.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

def create_process_flow_analysis(df, analysis_info=None):
    """보상 프로세스 흐름 분석"""
    st.subheader("🔄 보상 프로세스 흐름")
    
    # 분석 기준 정보 표시
    if analysis_info and analysis_info.get('date_basis') and analysis_info.get('date_range'):
        st.caption(f"📅 분석 기준일자: `{analysis_info['date_basis']}` / 범위: {analysis_info['date_range'][0]} ~ {analysis_info['date_range'][1]}")
    
    # 탭 생성
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ 프로세스 여정맵", "🌊 Sankey 흐름도", "📊 단계별 통계", "🔄 상태 전환 분석"])
    
    with tab1:
        create_journey_map(df, analysis_info)
    
    with tab2:
        create_sankey_diagram(df, analysis_info)
    
    with tab3:
        create_step_statistics(df, analysis_info)
    
    with tab4:
        create_status_transition_analysis(df, analysis_info)

def create_journey_map(df, analysis_info=None):
    """보상 프로세스 여정맵 생성"""
    st.subheader("🗺️ 보상 프로세스 여정맵")
    
    # 분석 기준 설명
    st.info("💡 **분석 기준**: 선택된 기간 내 판정이 진행된 모든 케이스를 기준으로 계산됩니다. 결재 대기 중인 케이스는 '최종결재' 단계에서 제외됩니다.")
    
    # 실제 프로세스 흐름에 따른 단계별 건수 계산
    step_data = []
    total_cases = len(df)
    
    # 1. 사고접수 (시작점)
    if '사고접수일자' in df.columns:
        accident_reported = len(df[df['사고접수일자'].notna()])
        step_data.append({
            '단계': '사고접수',
            '건수': accident_reported,
            '비율': accident_reported / total_cases * 100,
            '설명': '사고 접수 완료'
        })
    else:
        step_data.append({
            '단계': '사고접수',
            '건수': total_cases,
            '비율': 100.0,
            '설명': '전체 사고'
        })
    
    # 2. 보험금청구
    if '보험금청구일' in df.columns:
        claim_filed = len(df[df['보험금청구일'].notna()])
        step_data.append({
            '단계': '보험금청구',
            '건수': claim_filed,
            '비율': claim_filed / total_cases * 100,
            '설명': '보험금 청구 완료'
        })
    
    # 3. 판정 단계별 진행률 (실제 프로세스 흐름에 따른 누적 진행률)
    # 1차판정까지 진행된 건수
    first_round_total = len(df[df['판정회차'] >= 1])
    step_data.append({
        '단계': '1차판정',
        '건수': first_round_total,
        '비율': first_round_total / total_cases * 100,
        '설명': '1차판정까지 진행'
    })
    
    # 2차판정까지 진행된 건수
    second_round_total = len(df[df['판정회차'] >= 2])
    step_data.append({
        '단계': '2차판정',
        '건수': second_round_total,
        '비율': second_round_total / total_cases * 100,
        '설명': '2차판정까지 진행'
    })
    
    # 3차판정까지 진행된 건수
    third_round_total = len(df[df['판정회차'] >= 3])
    step_data.append({
        '단계': '3차판정',
        '건수': third_round_total,
        '비율': third_round_total / total_cases * 100,
        '설명': '3차판정까지 진행'
    })
    
    # 4. 최종결재 (실제 결재 완료된 건수)
    if '판정결재일' in df.columns:
        # 결재일이 있는 경우만 최종결재로 계산
        final_decision = len(df[df['판정결재일'].notna()])
        step_data.append({
            '단계': '최종결재',
            '건수': final_decision,
            '비율': final_decision / total_cases * 100,
            '설명': '결재 완료'
        })
    else:
        # 판정결재일 컬럼이 없는 경우 판정구분으로 추정
        final_decision = len(df[df['판정구분'].isin(['지급', '면책'])])
        step_data.append({
            '단계': '최종결재',
            '건수': final_decision,
            '비율': final_decision / total_cases * 100,
            '설명': '판정 완료'
        })
    
    # 5. 지급완료 (실제 지급된 건수)
    payment_completed = len(df[df['판정구분'] == '지급'])
    step_data.append({
        '단계': '지급완료',
        '건수': payment_completed,
        '비율': payment_completed / total_cases * 100,
        '설명': '지급 완료'
    })
    
    step_df = pd.DataFrame(step_data)
    
    # 여정맵 시각화
    fig = go.Figure()
    
    # 부드러운 색상 팔레트 (원색 제거)
    colors = {
        '사고접수': '#6c757d',      # 회색
        '보험금청구': '#495057',     # 진한 회색
        '1차판정': '#28a745',       # 부드러운 초록
        '2차판정': '#ffc107',       # 부드러운 노랑
        '3차판정': '#17a2b8',       # 부드러운 파랑
        '최종결재': '#6f42c1',      # 부드러운 보라
        '지급완료': '#20c997'       # 부드러운 청록
    }
    
    # 단계별 박스 그리기
    for i, (idx, row) in enumerate(step_df.iterrows()):
        # 박스 크기 (건수에 비례하되 최소 크기 보장)
        box_width = 1.0
        box_height = max(0.4, min(row['비율'] / 12, 0.9))  # 비율에 비례하되 최소 0.4, 최대 0.9
        
        # 박스 위치
        x_center = i
        y_center = 0
        
        # 박스 그리기 (둥근 모서리 효과)
        fig.add_shape(
            type="rect",
            x0=x_center - box_width/2,
            y0=y_center - box_height/2,
            x1=x_center + box_width/2,
            y1=y_center + box_height/2,
            fillcolor=colors.get(row['단계'], '#95a5a6'),
            opacity=0.9,
            line=dict(color="white", width=3),
            layer="below"
        )
        
        # 박스 그림자 효과 (약간 오른쪽 아래로)
        fig.add_shape(
            type="rect",
            x0=x_center - box_width/2 + 0.05,
            y0=y_center - box_height/2 - 0.05,
            x1=x_center + box_width/2 + 0.05,
            y1=y_center + box_height/2 - 0.05,
            fillcolor="rgba(0,0,0,0.1)",
            opacity=0.3,
            line=dict(width=0),
            layer="below"
        )
        
        # 단계명 텍스트
        fig.add_annotation(
            x=x_center,
            y=y_center + 0.15,
            text=f"<b>{row['단계']}</b><br>{row['건수']:,}건",
            showarrow=False,
            font=dict(size=12, color="white", family="Arial Black"),
            bgcolor="rgba(0,0,0,0.3)",
            bordercolor="white",
            borderwidth=1
        )
        
        # 비율 텍스트 (더 큰 폰트)
        fig.add_annotation(
            x=x_center,
            y=y_center - 0.15,
            text=f"<b>{row['비율']:.1f}%</b>",
            showarrow=False,
            font=dict(size=14, color="white", weight="bold", family="Arial Black"),
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="white",
            borderwidth=2
        )
        
        # 설명 텍스트
        fig.add_annotation(
            x=x_center,
            y=y_center - 0.6,
            text=row['설명'],
            showarrow=False,
            font=dict(size=10, color="#2c3e50", family="Arial"),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#bdc3c7",
            borderwidth=1
        )
        
        # 간단한 방향 표시 (화살표 대신)
        if i < len(step_df) - 1:
            # 단순한 선으로 방향만 표시
            fig.add_shape(
                type="line",
                x0=x_center + box_width/2 + 0.1,
                y0=y_center,
                x1=x_center + box_width/2 + 0.4,
                y1=y_center,
                line=dict(color="#dee2e6", width=2, dash="dot")
            )
    
    # 레이아웃 설정
    fig.update_layout(
        title=dict(
            text="보상 프로세스 여정맵 - 단계별 진행률",
            font=dict(size=20, color="#2c3e50", family="Arial Black"),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            range=[-0.5, len(step_df) - 0.5],
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            showline=False
        ),
        yaxis=dict(
            range=[-1.2, 1.2],
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            showline=False
        ),
        height=600,
        showlegend=False,
        plot_bgcolor='#f8f9fa',
        paper_bgcolor='#f8f9fa',
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 프로세스 분석 요약
    st.subheader("📊 프로세스 분석 요약")
    
    # 단계별 진행률을 카드 형태로 표시
    st.write("**단계별 진행률:**")
    
    # 3열로 배치
    cols = st.columns(3)
    
    for i, (_, row) in enumerate(step_df.iterrows()):
        col_idx = i % 3
        
        with cols[col_idx]:
            # 진행률에 따른 색상 결정 (부드러운 색상)
            if row['비율'] > 90:
                color = "#28a745"  # 부드러운 초록
                emoji = "🟢"
            elif row['비율'] > 70:
                color = "#20c997"  # 부드러운 청록
                emoji = "🟢"
            elif row['비율'] > 50:
                color = "#ffc107"  # 부드러운 노랑
                emoji = "🟡"
            else:
                color = "#dc3545"  # 부드러운 빨강
                emoji = "🔴"
            
            # 카드 스타일로 표시
            st.markdown(f"""
            <div style="
                background-color: {color};
                color: white;
                padding: 15px;
                border-radius: 10px;
                margin: 5px 0;
                text-align: center;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            ">
                <h4 style="margin: 0; font-size: 16px;">{emoji} {row['단계']}</h4>
                <p style="margin: 5px 0; font-size: 18px; font-weight: bold;">{row['건수']:,}건</p>
                <p style="margin: 0; font-size: 14px;">{row['비율']:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 프로세스 병목 지점 분석
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**🔍 프로세스 병목 지점:**")
        
        # 가장 큰 감소율 찾기
        max_drop = 0
        bottleneck_step = ""
        
        for i in range(len(step_df) - 1):
            current_ratio = step_df.iloc[i]['비율']
            next_ratio = step_df.iloc[i + 1]['비율']
            drop = current_ratio - next_ratio
            
            if drop > max_drop:
                max_drop = drop
                bottleneck_step = step_df.iloc[i]['단계']
        
        if max_drop > 0:
            st.error(f"**주요 병목**: {bottleneck_step} → {max_drop:.1f}% 감소")
        
        # 미완료 건수
        if '판정결재일' in df.columns:
            pending_cases = len(df[df['판정결재일'].isna()])
            if pending_cases > 0:
                st.warning(f"**결재 대기**: {pending_cases:,}건 ({pending_cases/len(df)*100:.1f}%)")
            else:
                st.success("**모든 케이스 결재 완료**")
    
    with col2:
        st.write("**📈 주요 지표:**")
        
        # 지급률
        payment_rate = len(df[df['판정구분'] == '지급']) / len(df) * 100
        st.success(f"**지급률**: {payment_rate:.1f}%")
        
        # 평균 판정회차
        avg_rounds = df['판정회차'].mean()
        st.info(f"**평균 판정회차**: {avg_rounds:.1f}회")
        
        # 처리완료율
        if '판정결재일' in df.columns:
            completion_rate = len(df[df['판정결재일'].notna()]) / len(df) * 100
            st.info(f"**처리완료율**: {completion_rate:.1f}%")
    
    # 프로세스 효율성 지표
    st.subheader("📈 프로세스 효율성 지표")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # 평균 처리 시간 계산
        if '사고접수일자' in df.columns and '판정결재일' in df.columns:
            df_time = df[df['사고접수일자'].notna() & df['판정결재일'].notna()].copy()
            df_time['처리일수'] = (df_time['판정결재일'] - df_time['사고접수일자']).dt.days
            avg_days = df_time['처리일수'].mean()
            st.metric("평균 처리일수", f"{avg_days:.1f}일")
        else:
            st.metric("평균 처리일수", "N/A")
    
    with col2:
        # 지급률
        payment_rate = len(df[df['판정구분'] == '지급']) / len(df) * 100
        st.metric("지급률", f"{payment_rate:.1f}%")
    
    with col3:
        # 평균 판정회차
        avg_rounds = df['판정회차'].mean()
        st.metric("평균 판정회차", f"{avg_rounds:.1f}회")
    
    with col4:
        # 처리완료율
        completion_rate = len(df[df['판정결재일'].notna()]) / len(df) * 100
        st.metric("처리완료율", f"{completion_rate:.1f}%")

def create_sankey_diagram(df, analysis_info=None):
    """Sankey 다이어그램으로 프로세스 흐름 시각화"""
    st.subheader("🌊 Sankey 다이어그램 - 프로세스 흐름")
    
    # 노드 정의 (각 단계)
    nodes = [
        "사고접수", "보험금청구", "1차판정", "2차판정", "3차판정", "최종결재", "지급", "면책"
    ]
    
    # 링크 데이터 생성
    links = []
    
    # 사고접수 -> 보험금청구
    if '사고접수일자' in df.columns and '보험금청구일' in df.columns:
        count = len(df[df['사고접수일자'].notna() & df['보험금청구일'].notna()])
        links.append([0, 1, count])  # 사고접수 -> 보험금청구
    
    # 보험금청구 -> 1차판정
    if '보험금청구일' in df.columns:
        count = len(df[df['판정회차'] >= 1])
        links.append([1, 2, count])  # 보험금청구 -> 1차판정
    else:
        # 보험금청구일이 없는 경우 사고접수에서 직접 1차판정으로
        count = len(df[df['판정회차'] >= 1])
        links.append([0, 2, count])  # 사고접수 -> 1차판정
    
    # 1차판정에서 종료되는 경우 -> 최종결재
    first_round_only = len(df[df['판정회차'] == 1])
    links.append([2, 5, first_round_only])  # 1차판정 -> 최종결재
    
    # 1차판정 -> 2차판정
    count_2nd = len(df[df['판정회차'] >= 2])
    links.append([2, 3, count_2nd])  # 1차판정 -> 2차판정
    
    # 2차판정에서 종료되는 경우 -> 최종결재
    second_round_only = len(df[df['판정회차'] == 2])
    links.append([3, 5, second_round_only])  # 2차판정 -> 최종결재
    
    # 2차판정 -> 3차판정
    count_3rd = len(df[df['판정회차'] >= 3])
    links.append([3, 4, count_3rd])  # 2차판정 -> 3차판정
    
    # 3차판정 -> 최종결재
    count_3rd_final = len(df[df['판정회차'] >= 3])
    links.append([4, 5, count_3rd_final])  # 3차판정 -> 최종결재
    
    # 최종결재 -> 지급/면책
    count_payment = len(df[df['판정구분'] == '지급'])
    count_rejection = len(df[df['판정구분'] == '면책'])
    links.append([5, 6, count_payment])  # 최종결재 -> 지급
    links.append([5, 7, count_rejection])  # 최종결재 -> 면책
    
    # Sankey 다이어그램 생성
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes,
            color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
        ),
        link=dict(
            source=[link[0] for link in links],
            target=[link[1] for link in links],
            value=[link[2] for link in links],
            color=["rgba(0,0,0,0.2)"] * len(links)
        )
    )])
    
    fig.update_layout(
        title_text="보상 프로세스 흐름도 - Sankey 다이어그램",
        font_size=12,
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 흐름도 해석 가이드
    st.info("""
    💡 **Sankey 다이어그램 해석 가이드**:
    - 링크의 두께는 해당 경로를 통과하는 건수를 나타냅니다
    - 각 노드는 프로세스의 단계를 나타냅니다
    - 지급과 면책으로 나뉘는 분기점을 통해 최종 결과를 확인할 수 있습니다
    """)
    
    # 주요 지표 요약
    st.subheader("📊 주요 프로세스 지표")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # 전체 처리율
        total_processed = len(df[df['판정결재일'].notna()])
        total_rate = total_processed / len(df) * 100
        st.metric("전체 처리율", f"{total_rate:.1f}%")
    
    with col2:
        # 지급률
        payment_rate = len(df[df['판정구분'] == '지급']) / len(df) * 100
        st.metric("지급률", f"{payment_rate:.1f}%")
    
    with col3:
        # 평균 판정회차
        avg_rounds = df['판정회차'].mean()
        st.metric("평균 판정회차", f"{avg_rounds:.1f}회")

def create_step_statistics(df, analysis_info=None):
    """단계별 상세 통계"""
    st.subheader("📊 단계별 상세 통계")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 판정회차별 분포
        round_counts = df['판정회차'].value_counts().sort_index()
        fig1 = px.bar(
            x=round_counts.index,
            y=round_counts.values,
            title="판정회차별 분포",
            labels={'x': '판정회차', 'y': '건수'},
            color=round_counts.values,
            color_continuous_scale='greens'
        )
        fig1.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # 사고진행상태별 분포
        status_counts = df['사고진행상태'].value_counts().head(8)
        fig2 = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="사고진행상태 분포",
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        fig2.update_traces(textposition='inside', textinfo='percent+label')
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
    
    # 단계별 처리 시간 분석
    if '사고접수일자' in df.columns and '판정결재일' in df.columns:
        st.subheader("⏱️ 단계별 처리 시간 분석")
        
        df_time = df[df['사고접수일자'].notna() & df['판정결재일'].notna()].copy()
        df_time['처리일수'] = (df_time['판정결재일'] - df_time['사고접수일자']).dt.days
        
        # 데이터 검증 및 디버깅 정보
        st.write("**🔍 데이터 검증:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("총 케이스 수", len(df_time))
        
        with col2:
            st.metric("평균 처리일수", f"{df_time['처리일수'].mean():.1f}일")
        
        with col3:
            st.metric("최대 처리일수", f"{df_time['처리일수'].max()}일")
        
        # 처리일수 분포 확인
        st.write("**📊 처리일수 분포:**")
        col1, col2 = st.columns(2)
        
        with col1:
            # 처리일수 히스토그램
            fig_hist = px.histogram(
                df_time, 
                x='처리일수',
                title="처리일수 분포",
                nbins=20,
                labels={'처리일수': '처리일수', 'count': '케이스 수'}
            )
            fig_hist.update_layout(height=300)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # 이상치 확인
            q75 = df_time['처리일수'].quantile(0.75)
            q25 = df_time['처리일수'].quantile(0.25)
            iqr = q75 - q25
            outlier_threshold = q75 + 1.5 * iqr
            
            normal_cases = df_time[df_time['처리일수'] <= outlier_threshold]
            outlier_cases = df_time[df_time['처리일수'] > outlier_threshold]
            
            st.write(f"**정상 범위**: {q25:.0f}일 ~ {q75:.0f}일")
            st.write(f"**이상치 기준**: {outlier_threshold:.0f}일 초과")
            st.write(f"**이상치 케이스**: {len(outlier_cases)}건 ({len(outlier_cases)/len(df_time)*100:.1f}%)")
            
            if len(outlier_cases) > 0:
                st.write("**🔍 이상치 케이스 샘플:**")
                sample_outliers = outlier_cases[['사고접수일자', '판정결재일', '처리일수']].head(3)
                st.dataframe(sample_outliers, use_container_width=True)
        
        # 판정회차별 평균 처리일수 (이상치 제외)
        st.write("**📈 판정회차별 평균 처리일수 (이상치 제외):**")
        normal_avg_days = normal_cases.groupby('판정회차')['처리일수'].agg(['mean', 'count']).reset_index()
        normal_avg_days.columns = ['판정회차', '평균처리일수', '건수']
        
        fig3 = px.bar(
            normal_avg_days,
            x='판정회차',
            y='평균처리일수',
            title="판정회차별 평균 처리일수 (이상치 제외)",
            labels={'평균처리일수': '평균 처리일수', '판정회차': '판정회차'},
            color='건수',
            color_continuous_scale='viridis'
        )
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)
        
        # 원본 데이터 (이상치 포함)도 표시
        st.write("**📈 판정회차별 평균 처리일수 (전체):**")
        avg_days_by_round = df_time.groupby('판정회차')['처리일수'].agg(['mean', 'count']).reset_index()
        avg_days_by_round.columns = ['판정회차', '평균처리일수', '건수']
        
        fig4 = px.bar(
            avg_days_by_round,
            x='판정회차',
            y='평균처리일수',
            title="판정회차별 평균 처리일수 (전체)",
            labels={'평균처리일수': '평균 처리일수', '판정회차': '판정회차'},
            color='건수',
            color_continuous_scale='viridis'
        )
        fig4.update_layout(height=400)
        st.plotly_chart(fig4, use_container_width=True)

def create_status_transition_analysis(df, analysis_info=None):
    """상태 전환 분석"""
    st.subheader("🔄 상태 전환 분석")
    
    # 판정구분별 사고진행상태 분석
    if '사고진행상태' in df.columns and '판정구분' in df.columns:
        status_decision = df.groupby(['사고진행상태', '판정구분']).size().unstack(fill_value=0)
        
        # 비율로 변환
        status_decision_pct = status_decision.div(status_decision.sum(axis=1), axis=0) * 100
        
        fig = px.imshow(
            status_decision_pct.values,
            x=status_decision_pct.columns,
            y=status_decision_pct.index,
            title="사고진행상태별 판정구분 비율 (%)",
            color_continuous_scale='RdYlBu_r',
            aspect='auto',
            text_auto=True
        )
        fig.update_layout(height=500, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    # 시간별 처리 패턴 분석
    if '판정일' in df.columns:
        st.subheader("📅 처리 패턴")
        
        df_time = df[df['판정일'].notna()].copy()
        df_time['판정월'] = df_time['판정일'].dt.to_period('M')
        monthly_counts = df_time['판정월'].value_counts().sort_index()
        
        fig2 = px.line(
            x=monthly_counts.index.astype(str),
            y=monthly_counts.values,
            title="월별 판정 건수 추이",
            labels={'x': '월', 'y': '판정 건수'}
        )
        fig2.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig2, use_container_width=True)

def main():
    st.title("📊 보험사고 판정 현황")
    st.markdown("---")
    
    # 데이터 로드
    df = load_data()
    if df is None:
        st.error("데이터를 로드할 수 없습니다.")
        return
    
    # 사이드바 - 필터 옵션
    st.sidebar.header("📋 분석 옵션")
    
    # 기준일자 선택
    available_date_columns = [col for col in ['판정일', '사고접수일자', '보험금청구일', '판정결재일'] if col in df.columns]
    date_basis = st.sidebar.selectbox(
        "분석 기준일자 선택",
        options=available_date_columns,
        index=0  # 기본값은 판정일
    )
    
    # 기간 필터
    if date_basis in df.columns:
        min_date = df[date_basis].min()
        max_date = df[date_basis].max()
        if pd.notna(min_date) and pd.notna(max_date):
            date_range = st.sidebar.date_input(
                "분석 기간 선택",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            if len(date_range) == 2:
                # 선택된 기준일자로 필터링
                df = df[(df[date_basis] >= pd.Timestamp(date_range[0])) & 
                       (df[date_basis] <= pd.Timestamp(date_range[1]))]
                
                # 결재 대기 중인 케이스 정보 표시
                if '판정결재일' in df.columns:
                    pending_cases = len(df[df['판정결재일'].isna()])
                    if pending_cases > 0:
                        st.sidebar.info(f"⚠️ 결재 대기 중: {pending_cases:,}건")
    
    # 분석 기준 정보 저장
    analysis_info = {
        'date_basis': date_basis,
        'date_range': date_range if 'date_range' in locals() and len(date_range) == 2 else None
    }
    
    # 판정구분 필터
    decision_filter = st.sidebar.multiselect(
        "판정구분 선택",
        options=df['판정구분'].unique(),
        default=df['판정구분'].unique()
    )
    df_filtered = df[df['판정구분'].isin(decision_filter)]
    
    # 전체 통계 요약
    st.subheader("📈 전체 현황 요약")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("전체 사고건수", f"{len(df_filtered):,}건")
    with col2:
        st.metric("지급 비율", f"{(df_filtered['판정구분'] == '지급').mean() * 100:.1f}%")
    with col3:
        st.metric("면책 비율", f"{(df_filtered['판정구분'] == '면책').mean() * 100:.1f}%")
    with col4:
        st.metric("보상파일 수", f"{df_filtered['보상파일번호'].nunique():,}건")
    
    st.markdown("---")
    
    # 각 분석 섹션
    create_country_analysis(df_filtered, analysis_info)
    st.markdown("---")
    
    create_accident_type_analysis(df_filtered, analysis_info)
    st.markdown("---")
    
    create_amount_analysis(df_filtered, analysis_info)
    st.markdown("---")
    
    create_insurance_product_analysis(df_filtered, analysis_info)
    st.markdown("---")
    
    create_process_flow_analysis(df_filtered, analysis_info)
    
    # 데이터 테이블 (선택사항)
    if st.sidebar.checkbox("원본 데이터 보기"):
        st.subheader("📋 원본 데이터")
        st.dataframe(df_filtered.head(100))

if __name__ == "__main__":
    main()