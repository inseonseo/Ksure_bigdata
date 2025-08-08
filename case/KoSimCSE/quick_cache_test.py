#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
캐시 시스템 빠른 테스트
"""

from case_similarity_search import CaseSimilaritySearch
import time
import os

def quick_cache_test():
    print("🔍 캐시 시스템 테스트")
    
    # 캐시 경로 확인
    search_engine = CaseSimilaritySearch.__new__(CaseSimilaritySearch)
    search_engine.embedding_cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kosimcse_embeddings_cache.pkl')
    
    print(f"📁 캐시 파일 경로: {search_engine.embedding_cache_path}")
    print(f"📁 캐시 파일 존재: {os.path.exists(search_engine.embedding_cache_path)}")
    
    if not os.path.exists(search_engine.embedding_cache_path):
        # 기존 improved_embeddings_cache.pkl이 있는지 확인
        old_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'improved_embeddings_cache.pkl')
        if os.path.exists(old_cache):
            print(f"📂 기존 캐시 파일 발견: {old_cache}")
            print("💡 이 파일을 새 캐시로 복사할 수 있습니다.")
            
            # 사용자에게 물어보기
            response = input("기존 캐시 파일을 사용하시겠습니까? (y/n): ")
            if response.lower() == 'y':
                import shutil
                shutil.copy2(old_cache, search_engine.embedding_cache_path)
                print("✅ 캐시 파일 복사 완료!")
            else:
                print("⚠️ 새로운 임베딩을 생성해야 합니다 (약 30분 소요)")
        else:
            print("⚠️ 캐시 파일이 없습니다. 첫 실행 시 임베딩 생성이 필요합니다.")
    else:
        print("✅ 캐시 파일이 존재합니다!")
    
    # 빠른 초기화 테스트
    print("\n🚀 빠른 초기화 테스트...")
    start_time = time.time()
    
    try:
        search_engine = CaseSimilaritySearch(fast_mode=True)  # 빠른 모드로 테스트
        init_time = time.time() - start_time
        print(f"✅ 빠른 모드 초기화 완료: {init_time:.2f}초")
        
        # 옵션 확인
        options = search_engine.get_available_options()
        print(f"📋 사용 가능한 옵션: {len(options)}개")
        for field, values in options.items():
            print(f"   {field}: {len(values)}개")
            
    except Exception as e:
        print(f"❌ 초기화 실패: {str(e)}")
    
    print(f"\n💡 다음 단계:")
    print(f"   1. 캐시가 있다면: 웹 인터페이스 실행 (streamlit run web_interface.py)")
    print(f"   2. 캐시가 없다면: 첫 실행으로 임베딩 생성 필요")

if __name__ == "__main__":
    quick_cache_test()