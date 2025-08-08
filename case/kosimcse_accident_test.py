import torch
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

def cal_score(a, b):
    if len(a.shape) == 1: a = a.unsqueeze(0)
    if len(b.shape) == 1: b = b.unsqueeze(0)

    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    return torch.mm(a_norm, b_norm.transpose(0, 1)) * 100

class KoSimCSEAccidentTest:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 디바이스: {self.device}")
        
        print("📥 KoSimCSE 모델 로딩 중...")
        self.model = AutoModel.from_pretrained('BM-K/KoSimCSE-roberta')
        self.tokenizer = AutoTokenizer.from_pretrained('BM-K/KoSimCSE-roberta')
        self.model.to(self.device)
        self.model.eval()
        print("✅ KoSimCSE 모델 로드 완료!")
        
        self.embedding_cache = {}
    
    def load_data(self, file_path, test_size=0.2, random_state=42):
        """데이터 로드 및 train/test 분할"""
        df = pd.read_csv(file_path, encoding='cp949')
        df = df.dropna(subset=['사고설명'])
        df = df.drop_duplicates()
        print(f"📊 전체 데이터 로드: {len(df)}개 사고")
        
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state
        )
        print(f"📚 학습 데이터: {len(train_df)}개 사고")
        print(f"🧪 테스트 데이터: {len(test_df)}개 사고")
        return train_df, test_df
    
    def encode_text(self, text):
        """텍스트를 임베딩으로 변환"""
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        inputs = self.tokenizer(
            [text], padding=True, truncation=True, max_length=128, return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            embeddings, _ = self.model(**inputs, return_dict=False)
            embedding = embeddings[0][0].cpu().numpy()  # KoSimCSE는 첫 번째 토큰 사용
            self.embedding_cache[text] = embedding
            return embedding
    
    def encode_batch(self, texts, batch_size=16):
        """배치로 텍스트들을 임베딩으로 변환"""
        embeddings = []
        uncached_texts = []
        
        for text in texts:
            if text not in self.embedding_cache:
                uncached_texts.append(text)
            else:
                embeddings.append(self.embedding_cache[text])
        
        if uncached_texts:
            print(f"    📥 {len(uncached_texts)}개 새로운 텍스트 임베딩 생성 중...")
            total_batches = (len(uncached_texts) + batch_size - 1) // batch_size
            
            for i in range(0, len(uncached_texts), batch_size):
                batch_texts = uncached_texts[i:i+batch_size]
                inputs = self.tokenizer(
                    batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    batch_embeddings, _ = self.model(**inputs, return_dict=False)
                    batch_embeddings = batch_embeddings[0].cpu().numpy()
                    
                    for text, embedding in zip(batch_texts, batch_embeddings):
                        self.embedding_cache[text] = embedding
                        embeddings.append(embedding)
                
                current_batch = i // batch_size + 1
                if current_batch % max(1, total_batches // 10) == 0 or current_batch == total_batches:
                    progress = min((i + batch_size) / len(uncached_texts) * 100, 100)
                    print(f"    임베딩 진행률: {progress:.1f}% ({min(i + batch_size, len(uncached_texts))}/{len(uncached_texts)})")
        else:
            print(f"    ✅ 모든 텍스트가 캐시되어 있어서 즉시 완료!")
        
        return np.array(embeddings)
    
    def test_single_query(self, query, train_df, top_k=5):
        """단일 쿼리로 유사한 사고 찾기"""
        print(f"\n🔍 쿼리: {query}")
        
        # 쿼리 임베딩
        query_embedding = self.encode_text(query)
        
        # 학습 데이터 임베딩
        train_embeddings = self.encode_batch(train_df['사고설명'].tolist())
        
        # 유사도 계산
        similarities = cosine_similarity([query_embedding], train_embeddings)[0]
        
        # 상위 K개 결과
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        print(f"\n🎯 상위 {top_k}개 유사 사고:")
        for i, idx in enumerate(top_indices, 1):
            similarity = similarities[idx]
            accident_type = train_df.iloc[idx]['사고유형명']
            description = train_df.iloc[idx]['사고설명']
            print(f"\n{i}. 유사도: {similarity:.4f}")
            print(f"   사고유형: {accident_type}")
            print(f"   사고설명: {description}")
    
    def evaluate_performance(self, test_df, train_df, top_k=5):
        """모델 성능 평가"""
        print(f"\n📊 모델 성능 평가 (상위 {top_k}개 기준)")
        
        test_embeddings = self.encode_batch(test_df['사고설명'].tolist())
        train_embeddings = self.encode_batch(train_df['사고설명'].tolist())
        
        # 유사도 계산
        similarities = cosine_similarity(test_embeddings, train_embeddings)
        
        correct_count = 0
        total_similarities = []
        
        for i, test_row in test_df.iterrows():
            test_idx = test_df.index.get_loc(i)
            test_accident_type = test_row['사고유형명']
            
            # 상위 K개 인덱스
            top_indices = np.argsort(similarities[test_idx])[::-1][:top_k]
            
            # 정확도 계산
            for idx in top_indices:
                if train_df.iloc[idx]['사고유형명'] == test_accident_type:
                    correct_count += 1
                    break
            
            # 평균 유사도
            top_similarities = similarities[test_idx][top_indices]
            total_similarities.extend(top_similarities)
        
        accuracy = correct_count / len(test_df) * 100
        avg_similarity = np.mean(total_similarities)
        
        print(f"✅ 정확도: {accuracy:.2f}%")
        print(f"📈 평균 유사도: {avg_similarity:.4f}")
        
        return accuracy, avg_similarity

def main():
    # KoSimCSE 테스터 초기화
    tester = KoSimCSEAccidentTest()
    
    # 데이터 로드
    train_df, test_df = tester.load_data('Data/casestudy.csv')
    
    # 성능 평가
    accuracy, avg_similarity = tester.evaluate_performance(test_df, train_df)
    
    # 개별 쿼리 테스트
    print(f"\n🔍 개별 쿼리 테스트")
    test_samples = []
    for accident_type in test_df['사고유형명'].unique():
        type_samples = test_df[test_df['사고유형명'] == accident_type].head(1)
        test_samples.extend(type_samples['사고설명'].tolist())
    
    test_queries = test_samples[:3]  # 최대 3개 샘플
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- 테스트 쿼리 {i} ---")
        tester.test_single_query(query, train_df, top_k=3)

if __name__ == "__main__":
    main() 