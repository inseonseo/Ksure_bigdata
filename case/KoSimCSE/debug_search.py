#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
검색 디버깅 스크립트
"""

from case_similarity_search import CaseSimilaritySearch
import time

def debug_search():
    print("🔍 검색 디버깅 시작")
    
    # 검색기 초기화
    search_engine = CaseSimilaritySearch(fast_mode=True)  # 빠른 모드로 테스트
    
    # 사용 가능한 옵션 확인
    options = search_engine.get_available_options()
    print(f"📋 사용 가능한 옵션:")
    for field, values in options.items():
        print(f"   {field}: {len(values)}개")
        if len(values) > 0:
            print(f"      예시: {values[:3]}")
    
    # 간단한 검색 쿼리 테스트
    test_queries = [
        # 쿼리 1: 최소한의 정보
        {
            '사고설명': '수입자가 지급을 거절함'
        },
        # 쿼리 2: 사고유형명만
        {
            '사고유형명': options.get('사고유형명', ['지급거절'])[0] if options.get('사고유형명') else '지급거절'
        },
        # 쿼리 3: 수입국만
        {
            '수입국': options.get('수입국', ['미국'])[0] if options.get('수입국') else '미국'
        },
        # 쿼리 4: 보험종목만
        {
            '보험종목': options.get('보험종목', ['단기수출보험'])[0] if options.get('보험종목') else '단기수출보험'
        },
        # 쿼리 5: 조합
        {
            '사고유형명': options.get('사고유형명', ['지급거절'])[0] if options.get('사고유형명') else '지급거절',
            '수입국': options.get('수입국', ['미국'])[0] if options.get('수입국') else '미국',
            '사고설명': '수입자가 지급을 거절함'
        }
    ]
    
    for i, query in enumerate(test_queries):
        print(f"\n🔍 테스트 쿼리 {i+1}: {query}")
        
        # 검색 실행
        start_time = time.time()
        results = search_engine.search_similar_cases(query, top_k=5, verbose=False)
        search_time = time.time() - start_time
        
        print(f"   검색 시간: {search_time:.2f}초")
        print(f"   결과 수: {len(results)}개")
        
        if results:
            print(f"   유사도 범위: {min([r['similarity'] for r in results]):.3f} ~ {max([r['similarity'] for r in results]):.3f}")
            print(f"   첫 번째 결과:")
            top_result = results[0]
            print(f"     - 사건 ID: {top_result['case_id']}")
            print(f"     - 유사도: {top_result['similarity']:.3f}")
            print(f"     - 사고유형명: {top_result['case_info']['사고유형명']}")
            print(f"     - 수입국: {top_result['case_info']['수입국']}")
        else:
            print("   ❌ 검색 결과 없음")
    
    # 필터링 테스트
    print(f"\n🔍 필터링 테스트")
    test_query = {
        '사고설명': '수입자가 지급을 거절함'
    }
    
    results = search_engine.search_similar_cases(test_query, top_k=10, verbose=False)
    print(f"필터링 전 결과 수: {len(results)}개")
    
    if results:
        similarities = [r['similarity'] for r in results]
        print(f"유사도 분포:")
        print(f"   최소: {min(similarities):.3f}")
        print(f"   최대: {max(similarities):.3f}")
        print(f"   평균: {sum(similarities)/len(similarities):.3f}")
        
        # 다양한 임계값으로 필터링 테스트
        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
            filtered = [r for r in results if r['similarity'] >= threshold]
            print(f"   임계값 {threshold}: {len(filtered)}개 결과")

if __name__ == "__main__":
    debug_search() 