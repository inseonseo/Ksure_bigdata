import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def demonstrate_data_leakage():
    """데이터 누출 문제를 시연하는 함수"""
    print("🚨 데이터 누출 문제 시연")
    print("=" * 50)
    
    # 가상의 데이터 생성
    np.random.seed(42)
    n_samples = 1000
    
    # 범주형 변수 (심사항목명)
    categories = ['화재', '도난', '교통사고', '자연재해', '기타']
    category_data = np.random.choice(categories, n_samples)
    
    # 숫자형 변수 (사고금액)
    amount_data = np.random.normal(1000000, 300000, n_samples)
    
    # 텍스트 변수 (사고설명)
    text_data = [f"사고유형_{cat}_케이스_{i}" for i, cat in enumerate(category_data)]
    
    # 데이터프레임 생성
    df = pd.DataFrame({
        '사고설명': text_data,
        '심사항목명': category_data,
        '사고금액': amount_data
    })
    
    print(f"📊 원본 데이터: {len(df)}행")
    print(f"📋 심사항목명 분포:")
    print(df['심사항목명'].value_counts())
    
    # 🚨 문제가 되는 방식 (데이터 누출)
    print(f"\n🚨 문제가 되는 방식 (데이터 누출):")
    print("1. 전체 데이터에 대해 전처리 모델 학습")
    
    # 전체 데이터에 대해 LabelEncoder 학습
    le_wrong = LabelEncoder()
    df['심사항목명_encoded_wrong'] = le_wrong.fit_transform(df['심사항목명'])
    
    # 전체 데이터에 대해 StandardScaler 학습
    scaler_wrong = StandardScaler()
    df['사고금액_scaled_wrong'] = scaler_wrong.fit_transform(df[['사고금액']])
    
    # 그 다음에 데이터 분할
    train_wrong, test_wrong = train_test_split(df, test_size=0.3, random_state=42)
    
    print(f"   - 학습 데이터: {len(train_wrong)}행")
    print(f"   - 테스트 데이터: {len(test_wrong)}행")
    print(f"   - LabelEncoder가 본 고유 카테고리: {len(le_wrong.classes_)}개")
    print(f"   - StandardScaler가 본 평균: {scaler_wrong.mean_[0]:.2f}")
    
    # ✅ 올바른 방식 (데이터 누출 방지)
    print(f"\n✅ 올바른 방식 (데이터 누출 방지):")
    print("1. 먼저 데이터 분할")
    
    # 먼저 데이터 분할
    train_correct, test_correct = train_test_split(df, test_size=0.3, random_state=42)
    
    # 학습 데이터만으로 LabelEncoder 학습
    le_correct = LabelEncoder()
    le_correct.fit(train_correct['심사항목명'])
    train_correct['심사항목명_encoded_correct'] = le_correct.transform(train_correct['심사항목명'])
    test_correct['심사항목명_encoded_correct'] = le_correct.transform(test_correct['심사항목명'])
    
    # 학습 데이터만으로 StandardScaler 학습
    scaler_correct = StandardScaler()
    scaler_correct.fit(train_correct[['사고금액']])
    train_correct['사고금액_scaled_correct'] = scaler_correct.transform(train_correct[['사고금액']])
    test_correct['사고금액_scaled_correct'] = scaler_correct.transform(test_correct[['사고금액']])
    
    print(f"   - 학습 데이터: {len(train_correct)}행")
    print(f"   - 테스트 데이터: {len(test_correct)}행")
    print(f"   - LabelEncoder가 본 고유 카테고리: {len(le_correct.classes_)}개")
    print(f"   - StandardScaler가 본 평균: {scaler_correct.mean_[0]:.2f}")
    
    # 🚨 데이터 누출의 영향 분석
    print(f"\n🚨 데이터 누출의 영향 분석:")
    
    # 테스트 데이터에만 있는 새로운 카테고리가 있다면?
    test_unique_categories = set(test_correct['심사항목명']) - set(train_correct['심사항목명'])
    if test_unique_categories:
        print(f"   ⚠️ 테스트 데이터에만 있는 카테고리: {test_unique_categories}")
        print(f"   🚨 문제가 되는 방식: 이 정보를 미리 알게 됨!")
        print(f"   ✅ 올바른 방식: 테스트 시점에만 알게 됨")
    else:
        print(f"   ✅ 테스트 데이터에 새로운 카테고리 없음")
    
    # 숫자형 변수의 분포 차이
    train_mean = train_correct['사고금액'].mean()
    test_mean = test_correct['사고금액'].mean()
    print(f"   📊 학습 데이터 평균: {train_mean:.2f}")
    print(f"   📊 테스트 데이터 평균: {test_mean:.2f}")
    print(f"   📊 차이: {abs(train_mean - test_mean):.2f}")
    
    if abs(train_mean - test_mean) > 50000:
        print(f"   ⚠️ 학습/테스트 데이터 분포가 다름!")
        print(f"   🚨 문제가 되는 방식: 테스트 분포 정보가 미리 반영됨")
        print(f"   ✅ 올바른 방식: 학습 분포만으로 정규화")
    
    print(f"\n🎯 결론:")
    print(f"   - 데이터 누출은 테스트 데이터의 정보가 모델 학습 과정에 미리 반영되는 것")
    print(f"   - 이는 과도하게 높은 성능을 야기할 수 있음")
    print(f"   - 실제 운영 환경에서는 새로운 데이터에 대한 성능이 떨어질 수 있음")

if __name__ == "__main__":
    demonstrate_data_leakage() 