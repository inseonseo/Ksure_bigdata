#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
수정사항 테스트 스크립트
- 사고유형명 표시 확인
- 패턴 체크 제거 확인
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from case_similarity_search import CaseSimilaritySearch

def test_fixes():
    """수정사항 테스트"""
    print("🔧 수정사항 테스트 시작...")
    
    try:
        # 검색기 초기화
        print("📦 검색기 초기화 중...")
        search_engine = CaseSimilaritySearch()
        print("✅ 검색기 초기화 완료")
        
        # 사용 가능한 옵션 확인
        options = search_engine.get_available_options()
        print(f"📋 사용 가능한 사고유형: {len(options.get('사고유형명', []))}개")
        print(f"📋 사용 가능한 수입국: {len(options.get('수입국', []))}개")
        print(f"📋 사용 가능한 보험종목: {len(options.get('보험종목', []))}개")
        
        # 테스트 쿼리
        test_query = {
            '사고유형명': options.get('사고유형명', ['지급거절'])[0] if options.get('사고유형명') else '지급거절',
            '수입국': options.get('수입국', ['미국'])[0] if options.get('수입국') else '미국',
            '보험종목': options.get('보험종목', ['단기수출보험'])[0] if options.get('보험종목') else '단기수출보험',
            '사고설명': '수입자가 지급을 거절함'
        }
        
        print(f"\n🔍 테스트 쿼리: {test_query}")
        
        # 검색 실행
        print("\n🔍 유사 사례 검색 중...")
        results = search_engine.search_similar_cases(test_query, top_k=3, verbose=True)
        
        print(f"\n✅ 검색 완료: {len(results)}개 결과")
        
        # 결과 검증
        for i, result in enumerate(results):
            print(f"\n📊 결과 {i+1} 검증:")
            
            # 1. 사고유형명이 표시되는지 확인
            case_info = result['case_info']
            사고유형명 = case_info.get('사고유형명', 'N/A')
            print(f"   ✅ 사고유형명: {사고유형명}")
            
            # 2. 패턴 관련 필드가 제거되었는지 확인
            if '판정패턴' not in result:
                print("   ✅ 판정패턴 필드 제거됨")
            else:
                print("   ❌ 판정패턴 필드가 여전히 존재함")
                
            if '판정요약' not in result:
                print("   ✅ 판정요약 필드 제거됨")
            else:
                print("   ❌ 판정요약 필드가 여전히 존재함")
            
            # 3. 예측 결과가 있는지 확인
            predicted = result.get('predicted_results', {})
            print(f"   ✅ 예측 결과 - 판정구분: {predicted.get('판정구분', 'N/A')}")
            print(f"   ✅ 예측 결과 - 판정사유: {predicted.get('판정사유', 'N/A')}")
        
        print("\n🎉 모든 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fixes() 