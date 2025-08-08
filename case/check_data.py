import pandas as pd

# 데이터 로드
df = pd.read_csv('Data/casestudy.csv', encoding='cp949')

print("=== 데이터 기본 정보 ===")
print(f"행 수: {len(df)}")
print(f"열 수: {len(df.columns)}")
print(f"열 이름: {list(df.columns)}")

print("\n=== 처음 5행 ===")
print(df.head())

print("\n=== 데이터 타입 ===")
print(df.dtypes)

print("\n=== 결측값 확인 ===")
print(df.isnull().sum())

print("\n=== 사고유형 분포 ===")
if '사고유형명' in df.columns:
    print(df['사고유형명'].value_counts().head(10))
elif '사고유형' in df.columns:
    print(df['사고유형'].value_counts().head(10))

print("\n=== 사고설명 예시 ===")
if '사고설명' in df.columns:
    print(df['사고설명'].iloc[0])
    print(df['사고설명'].iloc[1])
    print(df['사고설명'].iloc[2]) 