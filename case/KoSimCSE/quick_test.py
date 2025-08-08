#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
빠른 테스트용 스크립트
KoSimCSE 임베딩 없이 기본 기능만 테스트
"""

from case_similarity_search import CaseSimilaritySearch
import time

def quick_test():
    print("🚀 빠른 테스트 시작 (KoSimCSE 임베딩 생략)")
    start_time = time.time()
    
    # 빠른 모드로 검색기 초기화
    search_engine = CaseSimilaritySearch(fast_mode=True)
    
    init_time = time.time() - start_time
    print(f"✅ 초기화 완료: {init_time:.2f}초")
    
    # 사용 가능한 옵션 확인
    options = search_engine.get_available_options()
    print("\n📋 사용 가능한 옵션:")
    for field, values in options.items():
        print(f"   {field}: {len(values)}개 옵션")
        if len(values) > 0:
            print(f"      예시: {values[:3]}")  # 처음 3개만 표시
    
    # 간단한 검색 테스트
    print("\n🔍 검색 테스트:")
    if '사고유형명' in options and len(options['사고유형명']) > 0:
        test_query = {
            '사고유형명': options['사고유형명'][0],  # 첫 번째 옵션 사용
            '수입국': options['수입국'][0] if '수입국' in options and len(options['수입국']) > 0 else '미국',
            '보험종목': options['보험종목'][0] if '보험종목' in options and len(options['보험종목']) > 0 else '단기수출보험'
        }
        
        print(f"   테스트 쿼리: {test_query}")
        
        search_start = time.time()
        results = search_engine.search_similar_cases(test_query, top_k=3, verbose=False)
        search_time = time.time() - search_start
        
        print(f"   검색 완료: {search_time:.2f}초")
        print(f"   결과 수: {len(results)}개")
        
        if results:
            print(f"   최고 유사도: {results[0]['similarity']:.3f}")
            print(f"   최저 유사도: {results[-1]['similarity']:.3f}")
    
    total_time = time.time() - start_time
    print(f"\n✅ 전체 테스트 완료: {total_time:.2f}초")

if __name__ == "__main__":
    quick_test() 