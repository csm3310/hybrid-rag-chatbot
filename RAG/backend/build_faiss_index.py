# build_faiss_index.py
import pickle
import numpy as np
import faiss

# chunk_df.pkl 저장된 파일 로드
with open("../data/chunk_df.pkl", "rb") as f:
    df = pickle.load(f)

print("✅ df 로드 완료:", df.shape)

# embedding 컬럼을 (N, D) numpy array로 변환
embeddings = np.vstack(df["embedding"].apply(lambda x: np.array(x)))
print("✅ embeddings.shape:", embeddings.shape)  

# L2 정규화 (cosine similarity용)
emb_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# FAISS 인덱스 생성 + 벡터 추가
dim = emb_norm.shape[1]   # 1024 (지금 네 데이터 기준)
index = faiss.IndexFlatIP(dim)
index.add(emb_norm)

print("✅ FAISS 인덱스에 추가된 벡터 수:", index.ntotal)

# 인덱스 + 메타데이터 저장
faiss.write_index(index, "../data/chunk.index")

with open("../data/metadata.pkl", "wb") as f:
    pickle.dump(df, f)

print("🎉 저장 완료: chunk.index, metadata.pkl")

