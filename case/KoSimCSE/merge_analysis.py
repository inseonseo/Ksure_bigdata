import pandas as pd
import numpy as np

# 데이터 로드
print("📊 데이터 로드 중...")
df = pd.read_csv('data/case.csv', encoding='cp949', index_col=0)
지급 = pd.read_csv('data/보상판정_지급.csv', encoding='utf-8')
합본 = pd.read_csv('data/보상판정_합본.csv', encoding='cp949')

# 컬럼명 통일
df.rename(columns={'보상파일번호_x': '보상파일번호'}, inplace=True)

# dd 데이터프레임 생성 및 전처리
dd = pd.concat([지급, 합본])
dd['보상파일번호'] = dd['보상파일번호'].astype(str).str.replace('-', '')
dd['사고번호'] = dd['사고번호'].astype(str).str.replace('-', '')

# df의 보상파일번호와 사고번호도 문자열로 변환
df['보상파일번호'] = df['보상파일번호'].astype(str)
df['사고번호'] = df['사고번호'].astype(str)

print(f"📊 데이터 크기:")
print(f"   - df: {df.shape}")
print(f"   - dd: {dd.shape}")

# 🔍 1단계: 키 값 확인 및 정리
print("\n🔍 1단계: 키 값 확인 및 정리")

# 각 데이터프레임의 고유 키 개수 확인
df_unique_keys = df[['보상파일번호', '사고번호']].drop_duplicates()
dd_unique_keys = dd[['보상파일번호', '사고번호']].drop_duplicates()

print(f"📊 고유 키 개수:")
print(f"   - df 고유 키: {len(df_unique_keys)}개")
print(f"   - dd 고유 키: {len(dd_unique_keys)}개")

# 키 값의 데이터 타입 및 형식 확인
print(f"\n📊 키 값 샘플:")
print(f"   - df 보상파일번호 샘플: {df['보상파일번호'].head().tolist()}")
print(f"   - dd 보상파일번호 샘플: {dd['보상파일번호'].head().tolist()}")
print(f"   - df 사고번호 샘플: {df['사고번호'].head().tolist()}")
print(f"   - dd 사고번호 샘플: {dd['사고번호'].head().tolist()}")

# 🔍 2단계: 교집합 및 차집합 분석
print("\n🔍 2단계: 교집합 및 차집합 분석")

# 교집합 (양쪽 모두에 있는 키)
common_keys = pd.merge(df_unique_keys, dd_unique_keys, on=['보상파일번호', '사고번호'], how='inner')
print(f"📊 교집합 (양쪽 모두에 있는 키): {len(common_keys)}개")

# df에만 있는 키
df_only_keys = pd.merge(df_unique_keys, dd_unique_keys, on=['보상파일번호', '사고번호'], how='left', indicator=True)
df_only_keys = df_only_keys[df_only_keys['_merge'] == 'left_only'][['보상파일번호', '사고번호']]
print(f"📊 df에만 있는 키: {len(df_only_keys)}개")

# dd에만 있는 키
dd_only_keys = pd.merge(dd_unique_keys, df_unique_keys, on=['보상파일번호', '사고번호'], how='left', indicator=True)
dd_only_keys = dd_only_keys[dd_only_keys['_merge'] == 'left_only'][['보상파일번호', '사고번호']]
print(f"📊 dd에만 있는 키: {len(dd_only_keys)}개")

# 🔍 3단계: Merge 전략 제안
print("\n🔍 3단계: Merge 전략 제안")

# LEFT JOIN (df 기준) - df의 모든 데이터 보존
print("🔄 LEFT JOIN (df 기준) 실행...")
merged_left = pd.merge(df, dd, on=['보상파일번호', '사고번호'], how='left', suffixes=('_df', '_dd'))

print(f"📊 LEFT JOIN 결과:")
print(f"   - 원본 df: {len(df)}행")
print(f"   - LEFT JOIN 후: {len(merged_left)}행")
print(f"   - 매칭된 행: {len(merged_left.dropna(subset=['판정일']))}행")
print(f"   - 매칭되지 않은 행: {len(merged_left[merged_left['판정일'].isna()])}행")

# 매칭률 계산
matching_rate = len(merged_left.dropna(subset=['판정일'])) / len(df) * 100
print(f"   - 매칭률: {matching_rate:.2f}%")

# 🔍 4단계: 매칭되지 않은 데이터 샘플 확인
print("\n🔍 4단계: 매칭되지 않은 데이터 샘플 확인")

# df에만 있는 데이터 샘플
if len(df_only_keys) > 0:
    print(f"📊 df에만 있는 데이터 샘플 (상위 3개):")
    sample_df_only = df.merge(df_only_keys.head(3), on=['보상파일번호', '사고번호'], how='inner')
    print(sample_df_only[['보상파일번호', '사고번호', '사고접수일자', '사고설명']].to_string())

# dd에만 있는 데이터 샘플
if len(dd_only_keys) > 0:
    print(f"\n📊 dd에만 있는 데이터 샘플 (상위 3개):")
    sample_dd_only = dd.merge(dd_only_keys.head(3), on=['보상파일번호', '사고번호'], how='inner')
    print(sample_dd_only[['보상파일번호', '사고번호', '사고접수일', '판정결재일']].to_string())

# 🔍 5단계: 최종 Merge 실행 및 저장
print("\n🔍 5단계: 최종 Merge 실행 및 저장")

# 최종 merged 데이터프레임 생성
final_merged = merged_left.copy()

# 매칭되지 않은 행에 대한 처리
unmatched_count = len(final_merged[final_merged['판정일'].isna()])
print(f"📊 최종 결과:")
print(f"   - 총 행 수: {len(final_merged)}")
print(f"   - 매칭된 행: {len(final_merged) - unmatched_count}")
print(f"   - 매칭되지 않은 행: {unmatched_count}")
print(f"   - 매칭률: {((len(final_merged) - unmatched_count) / len(final_merged) * 100):.2f}%")

# 결과 저장
final_merged.to_csv('data/merged_case_data.csv', index=False, encoding='utf-8-sig')
print(f"💾 결과가 'data/merged_case_data.csv'에 저장되었습니다.")

# 🔍 6단계: 추가 분석 및 권장사항
print("\n🔍 6단계: 추가 분석 및 권장사항")

if unmatched_count > 0:
    print(f"⚠️  매칭되지 않은 {unmatched_count}개 행이 있습니다.")
    print(f"   - 이는 정상적인 상황일 수 있습니다 (데이터 수집 시점 차이 등)")
    print(f"   - 필요시 추가 전처리나 수동 확인이 필요할 수 있습니다")

print(f"✅ LEFT JOIN을 사용하여 df의 모든 데이터를 보존하면서 dd의 정보를 추가했습니다.")
print(f"✅ 약 {len(final_merged) - unmatched_count:,}개 행이 성공적으로 매칭되었습니다.") 