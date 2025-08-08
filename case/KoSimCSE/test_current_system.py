#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
현재 시스템 테스트 스크립트
- 사고유형명 드롭다운 확인
- KoSimCSE 모델 로드 확인
- 검색 기능 테스트
"""

from case_similarity_search import CaseSimilaritySearch
import time

def test_current_system():
    print("🚀 현재 시스템 테스트 시작")
    start_time = time.time()
    
    # 1. 검색기 초기화 (fast_mode=False로 KoSimCSE 사용)
    print("\n1️⃣ 검색기 초기화...")
    try:
        search_engine = CaseSimilaritySearch(fast_mode=False)
        print("✅ 검색기 초기화 성공")
    except Exception as e:
        print(f"❌ 검색기 초기화 실패: {str(e)}")
        return
    
    # 2. 사용 가능한 옵션 확인
    print("\n2️⃣ 사용 가능한 옵션 확인...")
    options = search_engine.get_available_options()
    
    print("📋 드롭다운 옵션:")
    for field, values in options.items():
        print(f"   {field}: {len(values)}개 옵션")
        if len(values) > 0:
            print(f"      예시: {values[:3]}")
    
    # 3. 사고유형명 드롭다운 확인
    print("\n3️⃣ 사고유형명 드롭다운 확인...")
    if '사고유형명' in options:
        accident_types = options['사고유형명']
        print(f"✅ 사고유형명 드롭다운: {len(accident_types)}개 옵션")
        if len(accident_types) > 0:
            print(f"   첫 번째 옵션: {accident_types[0]}")
        else:
            print("⚠️ 사고유형명 옵션이 비어있습니다")
    else:
        print("❌ 사고유형명 필드를 찾을 수 없습니다")
    
    # 4. KoSimCSE 모델 확인
    print("\n4️⃣ KoSimCSE 모델 확인...")
    if search_engine.kosimcse_model is not None:
        print("✅ KoSimCSE 모델 로드됨")
        if search_engine.text_embeddings is not None:
            print(f"✅ 텍스트 임베딩 생성됨: {search_engine.text_embeddings.shape}")
        else:
            print("⚠️ 텍스트 임베딩이 생성되지 않았습니다")
    else:
        print("❌ KoSimCSE 모델이 로드되지 않았습니다")
    
    # 5. 실제 검색 테스트
    print("\n5️⃣ 실제 검색 테스트...")
    
    # 검색 가능한 옵션이 있는지 확인
    if '사고유형명' in options and len(options['사고유형명']) > 0:
        sample_accident_type = options['사고유형명'][0]
        sample_country = options['수입국'][0] if '수입국' in options and len(options['수입국']) > 0 else '미국'
        sample_insurance = options['보험종목'][0] if '보험종목' in options and len(options['보험종목']) > 0 else '단기수출보험'
        
        test_query = {
            '사고유형명': sample_accident_type,
            '수입국': sample_country,
            '보험종목': sample_insurance,
            '사고설명': '수입자가 품질 문제를 이유로 지급을 거절함. 신생 업체의 첫 수출 건에서 발생한 문제입니다.',
            '수출자': '신생 전자제품 수출업체'
        }
        
        print(f"🔍 검색 쿼리:")
        for key, value in test_query.items():
            print(f"   {key}: {value}")
        
        # 검색 실행
        search_start = time.time()
        results = search_engine.search_similar_cases(test_query, top_k=3, verbose=False)
        search_time = time.time() - search_start
        
        print(f"✅ 검색 완료: {search_time:.2f}초")
        print(f"📊 결과 수: {len(results)}개")
        
        if results:
            print(f"\n🏆 검색 결과 요약:")
            print(f"   최고 유사도: {results[0]['similarity']:.3f}")
            print(f"   최저 유사도: {results[-1]['similarity']:.3f}")
            
            # 첫 번째 결과 상세 표시
            top_result = results[0]
            print(f"\n🥇 최고 유사 사례:")
            print(f"   사건 ID: {top_result['case_id']}")
            print(f"   유사도: {top_result['similarity']:.3f}")
            print(f"   사고유형명: {top_result['case_info']['사고유형명']}")
            print(f"   수입국: {top_result['case_info']['수입국']}")
            print(f"   예측 판정구분: {top_result['predicted_results']['판정구분']}")
            print(f"   예측 판정사유: {top_result['predicted_results']['판정사유']}")
        else:
            print("❌ 검색 결과가 없습니다")
    else:
        print("⚠️ 검색 테스트를 위한 충분한 옵션이 없습니다")
    
    # 6. 전체 시스템 상태 요약
    total_time = time.time() - start_time
    print(f"\n✅ 전체 시스템 테스트 완료: {total_time:.2f}초")
    
    print(f"\n📊 시스템 상태 요약:")
    print(f"   - 사고유형명 드롭다운: {'✅' if '사고유형명' in options and len(options['사고유형명']) > 0 else '❌'}")
    print(f"   - KoSimCSE 모델: {'✅' if search_engine.kosimcse_model is not None else '❌'}")
    print(f"   - 텍스트 임베딩: {'✅' if search_engine.text_embeddings is not None else '❌'}")
    print(f"   - 검색 기능: {'✅' if 'results' in locals() and len(results) > 0 else '❌'}")
    
    print(f"\n💡 다음 단계:")
    print(f"   - 웹 인터페이스 실행: streamlit run web_interface.py")
    print(f"   - 샘플 테스트: python sample_test.py")
    print(f"   - 최적화된 검색: python optimized_search.py")

if __name__ == "__main__":
    test_current_system() 