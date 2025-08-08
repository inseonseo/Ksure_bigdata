#!/usr/bin/env python3
"""
기존 캐시를 활용한 빠른 테스트
"""

import os
import sys
import time
import shutil

# 현재 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def test_with_existing_cache():
    """기존 캐시를 활용한 테스트"""
    
    # 기존 캐시 파일들 확인
    cache_files = [
        'kosimcse_embeddings_cache.pkl',
        'optimized_comprehensive_embeddings_cache.pkl'
    ]
    
    print("📂 캐시 파일 상태:")
    for cache_file in cache_files:
        if os.path.exists(cache_file):
            size_mb = os.path.getsize(cache_file) / 1024 / 1024
            print(f"   ✅ {cache_file}: {size_mb:.1f}MB")
        else:
            print(f"   ❌ {cache_file}: 없음")
    
    # 기존 캐시가 있는 case_similarity_search 사용
    if os.path.exists('kosimcse_embeddings_cache.pkl'):
        print("\n🚀 기존 캐시를 활용한 빠른 검색 테스트")
        
        try:
            from case_similarity_search import CaseSimilaritySearch
            
            start_time = time.time()
            search_engine = CaseSimilaritySearch()
            init_time = time.time() - start_time
            
            print(f"✅ 초기화 완료: {init_time:.2f}초")
            
            # 테스트 쿼리
            query = {
                '사고유형명': '지급거절',
                '수입국': '미국',
                '보험종목': '단기수출보험',
                '사고설명': '수입자가 지급을 거절함'
            }
            
            print(f"\n🔍 검색 테스트:")
            print(f"   쿼리: {query}")
            
            search_start = time.time()
            results = search_engine.search_similar_cases(query, top_k=3)
            search_time = time.time() - search_start
            
            print(f"✅ 검색 완료: {search_time:.2f}초")
            print(f"📊 결과 수: {len(results)}개")
            
            # 첫 번째 결과만 간단히 표시
            if results:
                result = results[0]
                print(f"\n🏆 1순위 결과:")
                print(f"   유사도: {result['similarity']:.3f}")
                print(f"   사건ID: {result['case_id']}")
                case_info = result['case_info']
                print(f"   사고유형: {case_info['사고유형명']}")
                print(f"   수입국: {case_info['수입국']}")
            
            return True
            
        except Exception as e:
            print(f"❌ 오류 발생: {str(e)}")
            return False
    
    else:
        print("❌ 사용 가능한 캐시가 없습니다.")
        return False

def copy_cache_for_optimized():
    """기존 캐시를 최적화된 버전용으로 복사"""
    source = 'kosimcse_embeddings_cache.pkl'
    target = 'optimized_comprehensive_embeddings_cache.pkl'
    
    if os.path.exists(source) and not os.path.exists(target):
        print(f"📋 캐시 파일 복사: {source} → {target}")
        try:
            shutil.copy2(source, target)
            print("✅ 캐시 복사 완료")
            return True
        except Exception as e:
            print(f"❌ 캐시 복사 실패: {str(e)}")
            return False
    
    return False

if __name__ == "__main__":
    print("🧪 캐시 활용 빠른 테스트")
    print("=" * 50)
    
    # 기존 캐시로 테스트
    success = test_with_existing_cache()
    
    if success:
        print("\n✅ 캐시가 정상 작동합니다!")
        print("💡 이제 웹앱이나 다른 인터페이스를 사용할 수 있습니다.")
    else:
        print("\n❌ 캐시 테스트 실패")
        print("💡 새로 임베딩을 생성해야 할 수 있습니다.")