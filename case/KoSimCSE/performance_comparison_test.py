"""
성능 비교 테스트 스크립트
원본 comprehensive_similarity_search.py vs 최적화된 optimized_comprehensive_similarity_search.py
"""

import time
import os
import sys
from typing import Dict, List

# 현재 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def test_original_version():
    """원본 버전 테스트"""
    print("🔄 원본 버전 테스트 시작...")
    start_time = time.time()
    
    try:
        from comprehensive_similarity_search import ComprehensiveSimilaritySearch
        
        # 초기화 시간 측정
        init_start = time.time()
        search_engine = ComprehensiveSimilaritySearch()
        init_time = time.time() - init_start
        
        # 검색 시간 측정
        query = {
            '사고유형명': '지급거절',
            '수입국': '미국',
            '보험종목': '단기수출보험',
            '사고설명': '수입자가 지급을 거절함'
        }
        
        search_start = time.time()
        results = search_engine.search_similar_cases(query, top_k=3, verbose=False)
        search_time = time.time() - search_start
        
        total_time = time.time() - start_time
        
        return {
            'success': True,
            'init_time': init_time,
            'search_time': search_time,
            'total_time': total_time,
            'results_count': len(results),
            'has_embeddings': search_engine.text_embeddings is not None
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'total_time': time.time() - start_time
        }

def test_optimized_version():
    """최적화된 버전 테스트"""
    print("🚀 최적화된 버전 테스트 시작...")
    start_time = time.time()
    
    try:
        from optimized_comprehensive_similarity_search import OptimizedComprehensiveSimilaritySearch
        
        # 초기화 시간 측정
        init_start = time.time()
        search_engine = OptimizedComprehensiveSimilaritySearch()
        init_time = time.time() - init_start
        
        # 검색 시간 측정
        query = {
            '사고유형명': '지급거절',
            '수입국': '미국',
            '보험종목': '단기수출보험',
            '사고설명': '수입자가 지급을 거절함'
        }
        
        search_start = time.time()
        results = search_engine.search_similar_cases(query, top_k=3, verbose=False)
        search_time = time.time() - search_start
        
        total_time = time.time() - start_time
        
        return {
            'success': True,
            'init_time': init_time,
            'search_time': search_time,
            'total_time': total_time,
            'results_count': len(results),
            'has_embeddings': search_engine.text_embeddings is not None,
            'performance_stats': search_engine.performance_stats
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'total_time': time.time() - start_time
        }

def run_multiple_searches(search_engine, num_searches=5):
    """여러 번 검색 실행하여 평균 시간 측정"""
    queries = [
        {
            '사고유형명': '지급거절',
            '수입국': '미국',
            '보험종목': '단기수출보험',
            '사고설명': '수입자가 지급을 거절함'
        },
        {
            '사고유형명': '파산',
            '수입국': '중국',
            '보험종목': '중장기수출신용보험',
            '사고설명': '수입자 회사가 파산함'
        },
        {
            '사고유형명': '연체',
            '수입국': '독일',
            '보험종목': '단기수출보험',
            '사고설명': '대금 지급이 연체됨'
        },
        {
            '사고유형명': '지급거절',
            '수입국': '일본',
            '보험종목': '해외투자보험',
            '사고설명': '계약 위반으로 지급 거절'
        },
        {
            '사고유형명': '부도',
            '수입국': '베트남',
            '보험종목': '단기수출보험',
            '사고설명': '수입업체 부도'
        }
    ]
    
    search_times = []
    
    for i in range(num_searches):
        query = queries[i % len(queries)]
        
        start_time = time.time()
        results = search_engine.search_similar_cases(query, top_k=3, verbose=False)
        search_time = time.time() - start_time
        
        search_times.append(search_time)
    
    return {
        'avg_search_time': sum(search_times) / len(search_times),
        'min_search_time': min(search_times),
        'max_search_time': max(search_times),
        'total_searches': num_searches
    }

def print_comparison_results(original_result, optimized_result):
    """결과 비교 출력"""
    print("\n" + "="*80)
    print("📊 성능 비교 결과")
    print("="*80)
    
    if original_result['success'] and optimized_result['success']:
        print(f"📈 초기화 시간:")
        print(f"   원본: {original_result['init_time']:.2f}초")
        print(f"   최적화: {optimized_result['init_time']:.2f}초")
        
        if original_result['init_time'] > 0:
            improvement = ((original_result['init_time'] - optimized_result['init_time']) / original_result['init_time']) * 100
            print(f"   개선율: {improvement:.1f}%")
        
        print(f"\n🔍 단일 검색 시간:")
        print(f"   원본: {original_result['search_time']:.2f}초")
        print(f"   최적화: {optimized_result['search_time']:.2f}초")
        
        if original_result['search_time'] > 0:
            improvement = ((original_result['search_time'] - optimized_result['search_time']) / original_result['search_time']) * 100
            print(f"   개선율: {improvement:.1f}%")
        
        print(f"\n⏱️  전체 실행 시간:")
        print(f"   원본: {original_result['total_time']:.2f}초")
        print(f"   최적화: {optimized_result['total_time']:.2f}초")
        
        if original_result['total_time'] > 0:
            improvement = ((original_result['total_time'] - optimized_result['total_time']) / original_result['total_time']) * 100
            print(f"   개선율: {improvement:.1f}%")
        
        print(f"\n📋 기능 비교:")
        print(f"   임베딩 사용: 원본({original_result['has_embeddings']}) vs 최적화({optimized_result['has_embeddings']})")
        print(f"   결과 개수: 원본({original_result['results_count']}) vs 최적화({optimized_result['results_count']})")
        
        if 'performance_stats' in optimized_result:
            print(f"\n📊 최적화 버전 상세 통계:")
            stats = optimized_result['performance_stats']
            for key, value in stats.items():
                print(f"   {key}: {value:.2f}초")
    
    else:
        print("⚠️ 일부 테스트 실패:")
        if not original_result['success']:
            print(f"   원본 오류: {original_result.get('error', 'Unknown')}")
        if not optimized_result['success']:
            print(f"   최적화 오류: {optimized_result.get('error', 'Unknown')}")

def main():
    """메인 테스트 함수"""
    print("🧪 성능 비교 테스트 시작")
    print("="*80)
    
    # 첫 번째 실행 (초기화 포함)
    print("\n1️⃣ 첫 번째 실행 테스트 (초기화 포함)")
    original_result = test_original_version()
    optimized_result = test_optimized_version()
    
    print_comparison_results(original_result, optimized_result)
    
    # 두 번째 실행 (캐시 활용)
    print("\n\n2️⃣ 두 번째 실행 테스트 (캐시 활용)")
    print("🔄 최적화된 버전 재실행 (캐시 활용 테스트)...")
    
    optimized_result_cached = test_optimized_version()
    
    if optimized_result_cached['success'] and optimized_result['success']:
        print(f"\n📊 캐시 효과:")
        print(f"   첫 번째 실행: {optimized_result['total_time']:.2f}초")
        print(f"   두 번째 실행: {optimized_result_cached['total_time']:.2f}초")
        
        if optimized_result['total_time'] > 0:
            cache_improvement = ((optimized_result['total_time'] - optimized_result_cached['total_time']) / optimized_result['total_time']) * 100
            print(f"   캐시로 인한 개선율: {cache_improvement:.1f}%")
    
    # 반복 검색 테스트
    if optimized_result_cached['success']:
        print("\n\n3️⃣ 반복 검색 성능 테스트")
        
        try:
            from optimized_comprehensive_similarity_search import OptimizedComprehensiveSimilaritySearch
            search_engine = OptimizedComprehensiveSimilaritySearch()
            
            print("🔄 5회 연속 검색 테스트...")
            multiple_search_result = run_multiple_searches(search_engine, 5)
            
            print(f"\n📊 반복 검색 결과:")
            print(f"   평균 검색 시간: {multiple_search_result['avg_search_time']:.3f}초")
            print(f"   최소 검색 시간: {multiple_search_result['min_search_time']:.3f}초")
            print(f"   최대 검색 시간: {multiple_search_result['max_search_time']:.3f}초")
            
        except Exception as e:
            print(f"⚠️ 반복 검색 테스트 실패: {str(e)}")
    
    print(f"\n{'='*80}")
    print("✅ 성능 비교 테스트 완료")
    print("="*80)
    
    # 권장사항 출력
    print(f"\n💡 권장사항:")
    print(f"   1. 첫 실행은 시간이 걸리지만, 임베딩 캐시 생성 후 빠른 검색 가능")
    print(f"   2. 캐시 파일(optimized_comprehensive_embeddings_cache.pkl) 보관 권장")
    print(f"   3. 데이터 변경 시 캐시가 자동으로 무효화되어 새로 생성됨")
    print(f"   4. GPU 사용 시 더 빠른 임베딩 생성 가능")

if __name__ == "__main__":
    main()