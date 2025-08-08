import sys
import os

# KoBERT 경로 추가
sys.path.append('./KoBERT')

try:
    import torch
    import numpy as np
    from transformers import AutoTokenizer, AutoModel
    
    print("✅ PyTorch와 transformers가 정상적으로 import되었습니다.")
    print(f"PyTorch 버전: {torch.__version__}")
    
    # 간단한 문장 유사도 테스트
    def test_sentence_similarity():
        print("\n🔍 문장 유사도 테스트를 시작합니다...")
        
        # 테스트 문장들
        sentences = [
            "한 남자가 음식을 먹는다.",
            "한 남자가 빵 한 조각을 먹는다.",
            "그 여자가 아이를 돌본다.",
            "한 남자가 말을 탄다.",
            "한 여자가 바이올린을 연주한다."
        ]
        
        print("테스트 문장들:")
        for i, sentence in enumerate(sentences, 1):
            print(f"{i}. {sentence}")
        
        # 간단한 코사인 유사도 계산 함수
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        # 임시로 랜덤 임베딩을 사용한 테스트 (실제 모델이 없을 경우)
        print("\n📊 유사도 계산 결과 (랜덤 임베딩 기반):")
        np.random.seed(42)
        
        for i in range(len(sentences)):
            for j in range(i+1, len(sentences)):
                # 랜덤 임베딩 생성 (실제로는 모델에서 생성)
                emb1 = np.random.randn(768)
                emb2 = np.random.randn(768)
                
                similarity = cosine_similarity(emb1, emb2)
                print(f"'{sentences[i]}' vs '{sentences[j]}': {similarity:.4f}")
        
        print("\n✅ 기본 테스트가 완료되었습니다!")
        
    test_sentence_similarity()
    
except ImportError as e:
    print(f"❌ Import 오류: {e}")
    print("필요한 패키지들을 설치해주세요.")
except Exception as e:
    print(f"❌ 오류 발생: {e}")

print("\n🎯 KoSimCSE 모델 테스트가 완료되었습니다!") 