import pandas as pd

def add_previous_year_column(df):
    """
    전년도 정보가 있는 기업만 표시하는 컬럼을 추가하는 함수
    
    Parameters:
    df: DataFrame - 기준연도와 사업자등록번호가 포함된 데이터프레임
    
    Returns:
    DataFrame - '전년도정보있음' 컬럼이 추가된 데이터프레임
    """
    
    # 기준연도별 기업 목록 생성
    yearly_companies = df.groupby('기준연도')['사업자등록번호'].apply(set).to_dict()
    
    # 전년도 정보 보유 여부 확인 함수
    def has_previous_year(row):
        current_year = row['기준연도']
        current_company = row['사업자등록번호']
        
        # 전년도 기업 목록 가져오기
        previous_companies = yearly_companies.get(current_year - 1, set())
        
        # 현재 기업이 전년도에도 있는지 확인
        return current_company in previous_companies
    
    # 전년도 정보 있음 컬럼 추가
    df['전년도정보있음'] = df.apply(has_previous_year, axis=1)
    
    return df

# 사용 예시
if __name__ == "__main__":
    # 데이터 로드
    df = pd.read_csv('data/이용현황.csv', encoding='cp949', index_col=0)
    ds = pd.read_csv('data/인수내용.csv', encoding='cp949', index_col=0)
    
    # 이용여부 컬럼 추가
    df['이용여부'] = df['사업자등록번호'].isin(ds['사업자등록번호']).map({True: '이용', False: '미이용'})
    
    # 전년도 정보 있음 컬럼 추가
    df = add_previous_year_column(df)
    
    # 결과 확인
    print("전년도 정보 보유 현황:")
    print(df['전년도정보있음'].value_counts())
    
    print("\n연도별 전년도 정보 보유 현황:")
    print(df.groupby('기준연도')['전년도정보있음'].value_counts().unstack(fill_value=0))
    
    # 결과 저장
    df.to_csv('전년도정보_추가된_데이터.csv', encoding='utf-8-sig', index=False)
    print("\n✅ 전년도 정보 컬럼이 추가된 데이터가 저장되었습니다.") 