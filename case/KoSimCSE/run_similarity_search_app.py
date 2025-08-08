#!/usr/bin/env python3
"""
유사 사례 검색 앱 실행기
- 최적화된 임베딩 캐시 시스템
- 상세한 진행률 표시 (10%마다)
- 인터랙티브 검색 인터페이스
"""

import os
import sys
import time
import torch
from typing import Dict, List

# 현재 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def print_banner():
    """앱 시작 배너"""
    print("=" * 80)
    print("🔍 KoSimCSE 유사 사례 검색 시스템")
    print("=" * 80)
    print("📋 약 17,848개 사례에서 유사한 사고 사례를 찾아드립니다")
    print("⚡ 첫 실행시 임베딩 생성으로 시간이 소요됩니다 (3-12분)")
    print("🚀 두 번째부터는 캐시를 사용하여 빠른 검색이 가능합니다")
    print("=" * 80)

def check_system_info():
    """시스템 정보 확인"""
    print("\n🖥️  시스템 정보 확인 중...")
    
    # GPU 확인
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   ✅ GPU 사용 가능: {gpu_name} ({gpu_memory:.1f}GB)")
        print(f"   ⚡ 예상 임베딩 시간: 3-5분")
    else:
        print(f"   ⚠️  GPU 사용 불가 - CPU 사용")
        print(f"   ⏱️  예상 임베딩 시간: 8-12분")
    
    # 메모리 확인
    import psutil
    memory_gb = psutil.virtual_memory().total / 1024**3
    print(f"   💾 시스템 메모리: {memory_gb:.1f}GB")
    
    if memory_gb < 8:
        print(f"   ⚠️  메모리가 부족할 수 있습니다 (8GB+ 권장)")
    
    return gpu_available

def initialize_search_engine():
    """검색 엔진 초기화"""
    print("\n🚀 검색 엔진 초기화 중...")
    
    try:
        from optimized_comprehensive_similarity_search import OptimizedComprehensiveSimilaritySearch
        
        start_time = time.time()
        search_engine = OptimizedComprehensiveSimilaritySearch()
        init_time = time.time() - start_time
        
        print(f"\n✅ 검색 엔진 초기화 완료 ({init_time:.1f}초)")
        
        # 캐시 상태 확인
        cache_file = search_engine.embedding_cache_path
        if os.path.exists(cache_file):
            cache_size = os.path.getsize(cache_file) / 1024 / 1024
            print(f"📂 캐시 파일 발견: {cache_size:.1f}MB")
        else:
            print(f"📂 캐시 파일 없음 - 임베딩 새로 생성됨")
        
        return search_engine
        
    except Exception as e:
        print(f"❌ 검색 엔진 초기화 실패: {str(e)}")
        return None

def get_available_options(search_engine):
    """사용 가능한 검색 옵션 확인"""
    print("\n📋 사용 가능한 검색 옵션 확인 중...")
    
    options = search_engine.get_available_options()
    
    for field, values in options.items():
        print(f"   {field}: {len(values)}개 옵션")
        if len(values) <= 10:
            print(f"      예시: {', '.join(values[:5])}")
        else:
            print(f"      예시: {', '.join(values[:5])}... (총 {len(values)}개)")
    
    return options

def create_sample_queries():
    """샘플 검색 쿼리들"""
    return [
        {
            'name': '지급거절 사례 (미국)',
            'query': {
                '사고유형명': '지급거절',
                '수입국': '미국',
                '보험종목': '단기수출보험',
                '사고설명': '수입자가 지급을 거절함'
            }
        },
        {
            'name': '파산 사례 (중국)',
            'query': {
                '사고유형명': '파산',
                '수입국': '중국',
                '보험종목': '중장기수출신용보험',
                '사고설명': '수입자 회사가 파산함'
            }
        },
        {
            'name': '연체 사례 (독일)',
            'query': {
                '사고유형명': '연체',
                '수입국': '독일',
                '보험종목': '단기수출보험',
                '사고설명': '대금 지급이 연체됨'
            }
        },
        {
            'name': '부도 사례 (베트남)',
            'query': {
                '사고유형명': '부도',
                '수입국': '베트남',
                '보험종목': '단기수출보험',
                '사고설명': '수입업체 부도로 인한 손실'
            }
        }
    ]

def show_sample_queries():
    """샘플 쿼리 표시"""
    samples = create_sample_queries()
    
    print("\n🎯 샘플 검색 쿼리:")
    for i, sample in enumerate(samples, 1):
        print(f"   {i}. {sample['name']}")
        for key, value in sample['query'].items():
            print(f"      {key}: {value}")
        print()
    
    return samples

def get_user_choice():
    """사용자 선택 받기"""
    print("📝 검색 방법을 선택하세요:")
    print("   1. 샘플 쿼리 사용")
    print("   2. 직접 입력")
    print("   3. 종료")
    
    while True:
        try:
            choice = input("\n선택 (1-3): ").strip()
            if choice in ['1', '2', '3']:
                return int(choice)
            else:
                print("⚠️ 1, 2, 3 중에서 선택해주세요.")
        except KeyboardInterrupt:
            print("\n👋 프로그램을 종료합니다.")
            return 3

def select_sample_query(samples):
    """샘플 쿼리 선택"""
    print("\n🎯 샘플 쿼리 선택:")
    for i, sample in enumerate(samples, 1):
        print(f"   {i}. {sample['name']}")
    
    while True:
        try:
            choice = input(f"\n선택 (1-{len(samples)}): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(samples):
                return samples[int(choice) - 1]['query']
            else:
                print(f"⚠️ 1부터 {len(samples)} 사이의 숫자를 입력해주세요.")
        except KeyboardInterrupt:
            return None

def create_custom_query(options):
    """커스텀 쿼리 생성"""
    print("\n✏️  직접 검색 조건을 입력해주세요 (Enter로 건너뛰기 가능):")
    
    query = {}
    
    # 필수 필드들
    required_fields = ['사고유형명', '수입국', '보험종목']
    
    for field in required_fields:
        if field in options:
            print(f"\n📋 {field} (선택사항):")
            print(f"   사용 가능한 옵션: {', '.join(options[field][:10])}")
            if len(options[field]) > 10:
                print(f"   ... 총 {len(options[field])}개 옵션")
            
            value = input(f"   입력: ").strip()
            if value:
                query[field] = value
    
    # 사고설명
    print(f"\n📝 사고설명 (자유 입력):")
    description = input(f"   입력: ").strip()
    if description:
        query['사고설명'] = description
    
    return query if query else None

def run_search(search_engine, query, top_k=5):
    """검색 실행"""
    print(f"\n🔍 유사 사례 검색 중...")
    print(f"📝 검색 조건:")
    for key, value in query.items():
        print(f"   {key}: {value}")
    
    start_time = time.time()
    results = search_engine.search_similar_cases(query, top_k=top_k, verbose=True)
    search_time = time.time() - start_time
    
    print(f"\n⏱️ 검색 완료 ({search_time:.2f}초)")
    
    return results

def display_results(search_engine, results):
    """검색 결과 표시"""
    if not results:
        print("❌ 검색 결과가 없습니다.")
        return
    
    print(f"\n📊 검색 결과 ({len(results)}개):")
    
    for i, result in enumerate(results, 1):
        print(f"\n{'='*60}")
        print(f"🏆 {i}순위 - 유사도: {result['similarity']:.3f}")
        print(f"📋 사건 ID: {result['case_id']}")
        
        # 유사도 상세 분석
        if 'similarity_details' in result:
            details = result['similarity_details']
            print(f"\n📊 유사도 분석:")
            print(f"   📝 텍스트: {details['text_similarity']:.3f}")
            print(f"   🏷️  범주형: {details['categorical_similarity']:.3f}")
            print(f"   🔢 수치형: {details['numerical_similarity']:.3f}")
        
        # 사건 정보
        case_info = result['case_info']
        print(f"\n📋 사건 정보:")
        print(f"   사고유형: {case_info['사고유형명']}")
        print(f"   수입국: {case_info['수입국']}")
        print(f"   보험종목: {case_info['보험종목']}")
        print(f"   사고금액: {case_info['사고금액']:,.0f}원")
        
        # 간단한 판정 정보
        if result['decision_process']:
            first_decision = result['decision_process'][0]
            print(f"   판정: {first_decision['판정구분']}")
            if len(result['decision_process']) > 1:
                print(f"   (총 {len(result['decision_process'])}회 판정)")
    
    # 상세 결과 표시 옵션
    print(f"\n💡 상세 결과를 보시겠습니까?")
    detailed = input("상세 보기 (y/N): ").strip().lower()
    
    if detailed in ['y', 'yes']:
        for result in results:
            search_engine.print_detailed_result(result)

def main():
    """메인 함수"""
    print_banner()
    
    # 시스템 정보 확인
    gpu_available = check_system_info()
    
    input("\nEnter를 눌러 계속 진행하세요... ")
    
    # 검색 엔진 초기화
    search_engine = initialize_search_engine()
    if search_engine is None:
        print("❌ 검색 엔진 초기화에 실패했습니다. 프로그램을 종료합니다.")
        return
    
    # 사용 가능한 옵션 확인
    options = get_available_options(search_engine)
    
    # 샘플 쿼리 표시
    samples = show_sample_queries()
    
    # 메인 루프
    while True:
        print("\n" + "="*80)
        choice = get_user_choice()
        
        if choice == 3:  # 종료
            print("👋 프로그램을 종료합니다.")
            break
        
        elif choice == 1:  # 샘플 쿼리
            query = select_sample_query(samples)
            if query is None:
                continue
        
        elif choice == 2:  # 직접 입력
            query = create_custom_query(options)
            if query is None:
                print("⚠️ 검색 조건이 입력되지 않았습니다.")
                continue
        
        # 검색 실행
        try:
            results = run_search(search_engine, query)
            display_results(search_engine, results)
            
        except Exception as e:
            print(f"❌ 검색 중 오류 발생: {str(e)}")
        
        # 계속할지 묻기
        continue_search = input("\n🔄 다른 검색을 수행하시겠습니까? (Y/n): ").strip().lower()
        if continue_search in ['n', 'no']:
            print("👋 프로그램을 종료합니다.")
            break

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 사용자가 프로그램을 중단했습니다.")
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류 발생: {str(e)}")
        print("🔧 문제가 지속되면 관리자에게 문의하세요.")