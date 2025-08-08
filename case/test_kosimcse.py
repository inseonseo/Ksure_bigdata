import torch
from transformers import AutoModel, AutoTokenizer

def cal_score(a, b):
    if len(a.shape) == 1: a = a.unsqueeze(0)
    if len(b.shape) == 1: b = b.unsqueeze(0)

    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    return torch.mm(a_norm, b_norm.transpose(0, 1)) * 100

print("🔧 KoSimCSE 모델 로딩 중...")
model = AutoModel.from_pretrained('BM-K/KoSimCSE-roberta')
tokenizer = AutoTokenizer.from_pretrained('BM-K/KoSimCSE-roberta')
print("✅ 모델 로드 완료!")

sentences = ['치타가 들판을 가로 질러 먹이를 쫓는다.',
             '치타 한 마리가 먹이 뒤에서 달리고 있다.',
             '원숭이 한 마리가 드럼을 연주한다.']

print(f"\n📝 테스트 문장들:")
for i, sentence in enumerate(sentences):
    print(f"  {i+1}. {sentence}")

inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
embeddings, _ = model(**inputs, return_dict=False)

score01 = cal_score(embeddings[0][0], embeddings[1][0])
score02 = cal_score(embeddings[0][0], embeddings[2][0])

print(f"\n🎯 유사도 점수:")
print(f"  문장 1 vs 문장 2: {score01.item():.2f}")
print(f"  문장 1 vs 문장 3: {score02.item():.2f}")
print(f"\n✅ 테스트 완료!") 