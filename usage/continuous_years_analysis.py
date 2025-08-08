import pandas as pd
import numpy as np

def load_data():
    """데이터 로드"""
    df = pd.read_csv('data/이용현황.csv', encoding='cp949', index_col=0)
    ds = pd.read_csv('data/인수내용.csv', encoding='cp949', index_col=0)
    
    # 이용여부 컬럼 추가
    df['이용여부'] = df['사업자등록번호'].isin(ds['사업자등록번호']).map({True: '이용', False: '미이용'})
    
    return df, ds

def analyze_continuous_years(df):
    """연속 2년 이상의 정보가 있는 기업 분석"""
    
    # 기업별 연도별 데이터 확인
    company_years = df.groupby('사업자등록번호')['기준연도'].apply(list).reset_index()
    company_years['연도수'] = company_years['기준연도'].apply(len)
    company_years['연도정렬'] = company_years['기준연도'].apply(sorted)
    
    # 연속성 체크 함수
    def check_consecutive(years):
        if len(years) < 2:
            return False, 0, []
        
        years = sorted(years)
        consecutive_count = 1
        max_consecutive = 1
        consecutive_periods = []
        current_start = years[0]
        
        for i in range(1, len(years)):
            if years[i] == years[i-1] + 1:
                consecutive_count += 1
            else:
                if consecutive_count >= 2:
                    consecutive_periods.append((current_start, years[i-1]))
                max_consecutive = max(max_consecutive, consecutive_count)
                consecutive_count = 1
                current_start = years[i]
        
        # 마지막 연속 구간 처리
        if consecutive_count >= 2:
            consecutive_periods.append((current_start, years[-1]))
        max_consecutive = max(max_consecutive, consecutive_count)
        
        return max_consecutive >= 2, max_consecutive, consecutive_periods
    
    # 연속성 분석
    company_years[['연속2년이상', '최대연속년수', '연속기간']] = company_years['기준연도'].apply(
        lambda x: pd.Series(check_consecutive(x))
    )
    
    return company_years

def analyze_previous_years(df, target_year=2025):
    """전년도 기준 분석 (방안 1)"""
    
    # 기준연도별 기업 목록
    yearly_companies = df.groupby('기준연도')['사업자등록번호'].apply(set).to_dict()
    
    # 전년도 정보가 있는 기업 분석
    previous_year_analysis = {}
    
    for year in range(2020, target_year + 1):
        current_companies = yearly_companies.get(year, set())
        previous_companies = yearly_companies.get(year - 1, set())
        
        # 전년도 정보가 있는 기업
        with_previous = current_companies & previous_companies
        
        # 2년전 정보가 있는 기업
        two_years_ago_companies = yearly_companies.get(year - 2, set())
        with_two_years_ago = current_companies & two_years_ago_companies
        
        # 3년전 정보가 있는 기업
        three_years_ago_companies = yearly_companies.get(year - 3, set())
        with_three_years_ago = current_companies & three_years_ago_companies
        
        previous_year_analysis[year] = {
            '전체기업수': len(current_companies),
            '전년도정보있음': len(with_previous),
            '2년전정보있음': len(with_two_years_ago),
            '3년전정보있음': len(with_three_years_ago),
            '전년도비율(%)': round(len(with_previous) / len(current_companies) * 100, 2) if current_companies else 0,
            '2년전비율(%)': round(len(with_two_years_ago) / len(current_companies) * 100, 2) if current_companies else 0,
            '3년전비율(%)': round(len(with_three_years_ago) / len(current_companies) * 100, 2) if current_companies else 0
        }
    
    return pd.DataFrame(previous_year_analysis).T

def analyze_continuous_periods(df):
    """연속 기간별 분석 (방안 2)"""
    
    company_years = analyze_continuous_years(df)
    
    # 연속 기간별 분류
    continuous_analysis = {}
    
    # 2년 연속
    two_consecutive = company_years[company_years['최대연속년수'] == 2]
    
    # 3년 연속
    three_consecutive = company_years[company_years['최대연속년수'] == 3]
    
    # 4년 연속
    four_consecutive = company_years[company_years['최대연속년수'] == 4]
    
    # 5년 이상 연속
    five_plus_consecutive = company_years[company_years['최대연속년수'] >= 5]
    
    # 연속 2년 이상 (전체)
    any_consecutive = company_years[company_years['연속2년이상'] == True]
    
    continuous_analysis = {
        '연속2년': len(two_consecutive),
        '연속3년': len(three_consecutive),
        '연속4년': len(four_consecutive),
        '연속5년이상': len(five_plus_consecutive),
        '연속2년이상(전체)': len(any_consecutive),
        '전체기업수': len(company_years),
        '연속2년비율(%)': round(len(two_consecutive) / len(company_years) * 100, 2),
        '연속3년비율(%)': round(len(three_consecutive) / len(company_years) * 100, 2),
        '연속4년비율(%)': round(len(four_consecutive) / len(company_years) * 100, 2),
        '연속5년이상비율(%)': round(len(five_plus_consecutive) / len(company_years) * 100, 2),
        '연속2년이상비율(%)': round(len(any_consecutive) / len(company_years) * 100, 2)
    }
    
    return continuous_analysis, company_years

def get_companies_by_continuous_period(df, min_consecutive=2):
    """특정 연속 기간 이상의 기업 목록 반환"""
    
    company_years = analyze_continuous_years(df)
    
    # 연속 2년 이상 기업 필터링
    continuous_companies = company_years[company_years['최대연속년수'] >= min_consecutive]
    
    # 해당 기업들의 상세 정보
    continuous_data = df[df['사업자등록번호'].isin(continuous_companies['사업자등록번호'])]
    
    return continuous_data, continuous_companies

def main():
    """메인 분석 실행"""
    
    print("=" * 80)
    print("📊 연속 2년 이상 기업 정보 분석")
    print("=" * 80)
    
    # 데이터 로드
    df, ds = load_data()
    
    print(f"📈 전체 데이터: {len(df):,}건")
    print(f"📅 연도 범위: {df['기준연도'].min()}년 ~ {df['기준연도'].max()}년")
    print(f"🏢 고유 기업수: {df['사업자등록번호'].nunique():,}개사")
    print()
    
    # 방안 1: 전년도 기준 분석
    print("🔍 방안 1: 전년도 기준 분석")
    print("-" * 50)
    previous_analysis = analyze_previous_years(df)
    print(previous_analysis)
    print()
    
    # 방안 2: 연속 기간별 분석
    print("🔍 방안 2: 연속 기간별 분석")
    print("-" * 50)
    continuous_analysis, company_years = analyze_continuous_periods(df)
    
    for key, value in continuous_analysis.items():
        if '비율' in key:
            print(f"{key}: {value}%")
        else:
            print(f"{key}: {value:,}")
    
    print()
    
    # 연속 2년 이상 기업 상세 정보
    print("🔍 연속 2년 이상 기업 상세 분석")
    print("-" * 50)
    
    continuous_data, continuous_companies = get_companies_by_continuous_period(df, min_consecutive=2)
    
    print(f"연속 2년 이상 기업수: {len(continuous_companies):,}개사")
    print(f"연속 2년 이상 기업 데이터: {len(continuous_data):,}건")
    
    # 연속 기간별 분포
    consecutive_dist = continuous_companies['최대연속년수'].value_counts().sort_index()
    print("\n연속 기간별 분포:")
    for years, count in consecutive_dist.items():
        print(f"  {years}년 연속: {count:,}개사")
    
    # 지역별 분석
    print("\n지역별 연속 2년 이상 기업 분포:")
    regional_continuous = continuous_data.groupby('소재지')['사업자등록번호'].nunique().sort_values(ascending=False)
    for region, count in regional_continuous.items():
        print(f"  {region}: {count:,}개사")
    
    return df, continuous_data, continuous_companies

if __name__ == "__main__":
    df, continuous_data, continuous_companies = main() 