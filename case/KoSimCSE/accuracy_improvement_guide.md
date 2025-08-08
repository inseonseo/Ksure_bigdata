# 🎯 유사 사례 검색 정확도 개선 가이드

## 📊 현재 문제점들

### 1. **사고금액이 -0으로 표시되는 문제**
- 원인: StandardScaler로 정규화된 값이 출력됨
- 해결: 원본 값 별도 저장 후 출력시 사용

### 2. **범주형 매칭이 모두 실패하는 문제**
- 원인: 쿼리 인코딩과 데이터 인코딩 불일치
- 해결: LabelEncoder의 unknown 값 처리 개선

### 3. **텍스트 유사도만 높고 범주형이 0인 문제**
- 원인: 정확히 일치하는 값만 매칭으로 인정
- 해결: 유사 매칭 알고리즘 도입

## 🚀 정확도 개선 방법들

### A. 즉시 적용 가능한 개선사항

#### 1. **가중치 조정**
```python
# 현재 가중치
final_similarity = (
    0.6 * text_similarity +      # 텍스트 (KoSimCSE)
    0.3 * categorical_similarity + # 범주형
    0.1 * numerical_similarity    # 수치형
)

# 개선된 가중치 (도메인 특성 고려)
final_similarity = (
    0.4 * text_similarity +      # 텍스트
    0.5 * categorical_similarity + # 범주형 (중요도 증가)
    0.1 * numerical_similarity    # 수치형
)
```

#### 2. **범주형 유사도 개선**
```python
# 현재: 정확히 일치만 인정
if query_value == case_value:
    match = True

# 개선: 부분 매칭 허용
def calculate_categorical_similarity(query_value, case_value, field_type):
    if query_value == case_value:
        return 1.0
    
    # 국가명 유사도 (예: 미국 vs 캐나다)
    if field_type == '수입국':
        return country_similarity(query_value, case_value)
    
    # 사고유형 유사도 (예: 지급거절 vs 연체)
    if field_type == '사고유형명':
        return accident_type_similarity(query_value, case_value)
    
    return 0.0
```

#### 3. **수치형 유사도 개선**
```python
# 현재: 유클리드 거리
distance = np.sqrt(np.sum((query_numerical - case_numerical) ** 2))
similarity = 1 / (1 + distance)

# 개선: 상대적 차이 고려
def calculate_numerical_similarity(query_amount, case_amount):
    if query_amount == 0 and case_amount == 0:
        return 1.0
    
    if query_amount == 0 or case_amount == 0:
        return 0.0
    
    # 상대적 차이 계산 (10% 이내면 높은 유사도)
    relative_diff = abs(query_amount - case_amount) / max(query_amount, case_amount)
    return max(0, 1 - relative_diff)
```

### B. 고급 개선사항

#### 1. **의미적 범주형 유사도**
```python
# 사고유형별 유사도 매트릭스
accident_similarity_matrix = {
    '지급거절': {'연체': 0.8, '파산': 0.6, '부도': 0.7},
    '연체': {'지급거절': 0.8, '파산': 0.5, '부도': 0.6},
    '파산': {'부도': 0.9, '지급거절': 0.6, '연체': 0.5}
}
```

#### 2. **컨텍스트 기반 가중치**
```python
def get_context_weights(query):
    # 사고금액이 큰 경우 수치형 가중치 증가
    if query.get('사고금액', 0) > 100000000:  # 1억 이상
        return {'text': 0.3, 'categorical': 0.4, 'numerical': 0.3}
    
    # 텍스트 설명이 상세한 경우 텍스트 가중치 증가
    if len(query.get('사고설명', '')) > 50:
        return {'text': 0.6, 'categorical': 0.3, 'numerical': 0.1}
    
    # 기본 가중치
    return {'text': 0.4, 'categorical': 0.5, 'numerical': 0.1}
```

#### 3. **앙상블 방법**
```python
def ensemble_similarity(query, case):
    # 여러 방법의 유사도 계산
    kosimcse_sim = calculate_kosimcse_similarity(query, case)
    tfidf_sim = calculate_tfidf_similarity(query, case)
    categorical_sim = calculate_categorical_similarity(query, case)
    numerical_sim = calculate_numerical_similarity(query, case)
    
    # 앙상블 가중치
    final_sim = (
        0.4 * kosimcse_sim +
        0.2 * tfidf_sim +
        0.3 * categorical_sim +
        0.1 * numerical_sim
    )
    
    return final_sim
```

## 🌐 웹 인터페이스 구현

### Streamlit 기반 웹앱
```python
# 실행 명령어
streamlit run comprehensive_app.py --server.port 8501
```

### 주요 기능들
1. **실시간 검색**: 조건 입력시 즉시 결과 표시
2. **시각화**: 유사도 분포 차트
3. **상세 분석**: 클릭시 상세 정보 표시
4. **필터링**: 결과 필터링 및 정렬
5. **내보내기**: 결과를 CSV/Excel로 저장

## 📈 성능 최적화

### 1. **인덱싱 도입**
```python
# FAISS 인덱스 사용
import faiss

# 임베딩 인덱스 생성
index = faiss.IndexFlatIP(embedding_dim)
index.add(embeddings.astype('float32'))

# 빠른 검색
scores, indices = index.search(query_embedding, top_k)
```

### 2. **캐시 전략**
```python
# 검색 결과 캐시
@st.cache_data(ttl=3600)  # 1시간 캐시
def search_with_cache(query_hash, query):
    return search_engine.search_similar_cases(query)
```

## 🎯 테스트 케이스

### 고품질 테스트 쿼리들
```python
test_queries = [
    {
        "name": "지급거절_미국_상세설명",
        "query": {
            "사고유형명": "지급거절",
            "수입국": "미국",
            "보험종목": "단기수출보험",
            "사고설명": "수입자가 계약 위반을 이유로 지급을 거절하였으며, 협상을 통한 해결을 시도했으나 실패함",
            "사고금액": 50000000
        },
        "expected_features": ["지급거절", "미국", "계약위반"]
    },
    {
        "name": "파산_중국_제조업",
        "query": {
            "사고유형명": "파산",
            "수입국": "중국",
            "보험종목": "중장기수출신용보험",
            "사고설명": "중국 제조업체가 코로나19 여파로 경영난을 겪다가 파산 절차에 들어감",
            "사고금액": 200000000
        },
        "expected_features": ["파산", "중국", "제조업", "코로나"]
    },
    {
        "name": "연체_베트남_소액",
        "query": {
            "사고유형명": "연체",
            "수입국": "베트남",
            "보험종목": "단기수출보험",
            "사고설명": "베트남 바이어의 일시적 자금난으로 대금 지급이 지연됨",
            "사고금액": 10000000
        },
        "expected_features": ["연체", "베트남", "자금난"]
    }
]
```

## 📊 정확도 평가 방법

### 1. **정성적 평가**
- 전문가가 검색 결과의 적절성 평가
- 1-5점 척도로 유사도 점수 부여
- 상위 5개 결과 중 적절한 결과 비율 계산

### 2. **정량적 평가**
```python
def evaluate_search_accuracy(test_queries, search_engine):
    results = []
    
    for test_query in test_queries:
        # 검색 실행
        search_results = search_engine.search_similar_cases(
            test_query["query"], top_k=10
        )
        
        # 기대 특성과의 매칭 점수 계산
        relevance_scores = []
        for result in search_results[:5]:  # 상위 5개만 평가
            score = calculate_relevance_score(
                result, test_query["expected_features"]
            )
            relevance_scores.append(score)
        
        # 평균 정확도 계산
        avg_accuracy = np.mean(relevance_scores)
        results.append({
            "query_name": test_query["name"],
            "accuracy": avg_accuracy,
            "top1_accuracy": relevance_scores[0] if relevance_scores else 0
        })
    
    return results
```

## 💡 권장사항

### 단계별 개선 순서
1. **즉시 개선**: 결과 표시 오류 수정
2. **1주차**: 범주형 유사도 알고리즘 개선
3. **2주차**: 가중치 튜닝 및 컨텍스트 기반 조정
4. **3주차**: 웹 인터페이스 고도화
5. **4주차**: 성능 최적화 및 인덱싱 도입

### 즉시 적용 가능한 개선
- 결과 표시시 원본 금액 값 사용
- 범주형 매칭 로직 개선
- 테스트 케이스로 정확도 검증