import pickle
import faiss
import numpy as np

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.docstore import InMemoryDocstore
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from datetime import datetime
import re
from langchain_community.chat_models import ChatOllama

# 메타데이터 & FAISS 로드

# 원문 전체 데이터 로드
with open("../data/sch_metadata.pkl", "rb") as f:
    full_df = pickle.load(f)

# chunk 데이터 (embedding용)
with open("../data/chunk_df.pkl", "rb") as f:
    chunk_df = pickle.load(f)

# CPU index 로드
faiss_index = faiss.read_index("../data/chunk.index")

# GPU ID
gpu_id = 0

# CPU → GPU 변환
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, faiss_index)

print("GPU index loaded:", gpu_index.ntotal)
print(f"✅ 원문 메타데이터 로드: {full_df.shape}")
print(f"✅ 청크 메타데이터 로드: {chunk_df.shape}")
print(f"✅ FAISS 로드 (ntotal={faiss_index.ntotal})")

from sentence_transformers import CrossEncoder
reranker = CrossEncoder("BAAI/bge-reranker-base", device="cuda")

# ---------------------------
# LangChain 문서 docstore 구성 
# ---------------------------
documents = {}
index_to_docstore_id = {}

for i, row in chunk_df.iterrows():
    doc_id = str(i)
    documents[doc_id] = Document(
        page_content=row["content_chunk"],
        metadata={
            "chunk_index": i,                         
            "original_id": int(row["original_id"]),
            "title": row["title"],
            "url": row["url"],
        },
    )
    index_to_docstore_id[i] = doc_id


docstore = InMemoryDocstore(documents)

# BM25 초기화
from rank_bm25 import BM25Okapi

corpus = [doc.page_content for doc in documents.values()]
from konlpy.tag import Okt
okt = Okt()

def tokenize_kr(text: str):
    # 명사(Noun) 중심 토큰화
    tokens = okt.nouns(text)
    return tokens

# corpus -> 형태소 기반 토큰화
corpus = [doc.page_content for doc in documents.values()]
tokenized_corpus = [tokenize_kr(text) for text in corpus]

from rank_bm25 import BM25Okapi
bm25 = BM25Okapi(tokenized_corpus)

# chunk → 원문 full text 로드 함수
def load_full_documents_from_chunks(chunks):
    """chunk 검색 결과에서 original_id를 추출 후 full text 리스트 반환"""
    original_ids = list({doc.metadata["original_id"] for doc in chunks})
    full_docs = []

    for oid in original_ids:
        # full_df에서 해당 원문 텍스트 가져오기
        rows = full_df.loc[full_df["id"] == oid, "content"]
        if len(rows) == 0:
            continue
        full_text = rows.values[0]
        full_docs.append(full_text)

    return full_docs

def expand_context(full_text, chunk_text, window=350):
    idx = full_text.find(chunk_text)
    if idx == -1:
        return chunk_text
    start = max(0, idx - window)
    end = idx + len(chunk_text) + window
    return full_text[start:end]

# Query embedding: e5
embedding_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={"device": "cuda"}
)

# 4) GPU 기반 검색 함수 
def gpu_search(query: str, top_k=3):
    query_vec = embedding_model.embed_query(query)
    query_vec = np.array([query_vec]).astype("float32")

    # top_k 후보 직접 검색
    D, I = gpu_index.search(query_vec, top_k)

    docs = []
    for i in I[0]:
        doc = documents[index_to_docstore_id[i]]
        docs.append(doc)

    return docs

print("🔧 GPU Search 준비 완료")

def hybrid_search(query, top_k=5):
    # === BM25 search ===
    tokenized_query = tokenize_kr(query)  
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top = np.argsort(bm25_scores)[::-1][:top_k]

    # === Dense search ===
    query_vec = embedding_model.embed_query(query)
    query_vec = np.array([query_vec]).astype("float32")
    D, I = gpu_index.search(query_vec, top_k)
    dense_top = I[0]

    # === score merge ===
    combined = {}

    for idx in bm25_top:
        combined[idx] = combined.get(idx, 0) + float(bm25_scores[idx]) * 0.5  # BM25 가중치

    for rank, idx in enumerate(dense_top):
        combined[idx] = combined.get(idx, 0) + float(D[0][rank]) * 1.0  # Dense 가중치

    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, score in ranked[:top_k]]

    docs = [documents[index_to_docstore_id[i]] for i in top_indices]
    return docs

# LLM (Ollama)
llm = Ollama(
    model="llama3.1:8b",
    temperature=0.1,
    num_ctx=4096,
    num_predict=400,
)

# RAG 프롬프트
qa_prompt_text = """
You are an AI assistant for the 'Soonchunhyang University Information RAG Chatbot'.

Response Rules:
1. Answer strictly based on the provided context.
2. If the answer cannot be found in the context, reply: "I don't know based on the provided context."
3. If the question is clearly unrelated to Soonchunhyang University (e.g., general knowledge, politics, entertainment, international news, product reviews, etc.), respond exactly:
   "저는 순천향대학교 챗봇입니다. 해당 질문에 대해서는 답변할 수 없습니다."

   However, if the topic may still be useful for Soonchunhyang University students
   (e.g., student competitions, hackathons, scholarships, public data challenges, startup programs, certifications, job opportunities, AI/Big Data events),
   then provide information only if it exists in the given context.

   If the relevant information is not found in the provided context, answer:
   "I don't know based on the provided context."

   If the request is ambiguous, ask a clarifying question rather than refusing.
   Example: "Which type of competition are you looking for? (AI, startup, public data, healthcare, etc.)"

4. For follow-up questions, do not repeat the previous answer; provide new information such as differences, updates, steps, time, or location.
5. Do not copy context verbatim; summarize concisely and naturally in Korean.
6. Format the response in 3-5 sentences with clean paragraph structure.
7. Do not hallucinate, infer beyond facts, or generate misleading assumptions.
8. Prioritize active or ongoing information. If content is outdated or past the deadline, do not include it in the answer. If necessary, mention briefly that the information has expired.
9. If multiple documents exist, respond focusing only on the most recent and currently valid content.


------------------------
[Context]
{context}

------------------------
[Previous Conversation Summary]
{chat_history}

------------------------
[User Question]
{question}

------------------------
[Answer]
"""

qa_prompt = PromptTemplate(
    template=qa_prompt_text,
    input_variables=["context", "question", "chat_history"],
)

# 7) 직접 구현한 메모리 (티키타카 강화)
chat_history = []


def get_history_text():
    """직전 턴의 질문/답변 일부를 써서 티키타카 강화"""
    if len(chat_history) == 0:
        return ""
    last_turn = chat_history[-1]
    last_answer = str(last_turn["assistant"])
    if len(last_answer) > 200:
        last_answer = last_answer[:200] + "..."
    return f"이전 질문: {last_turn['user']}\n이전 답변(요약): {last_answer}"


# 문서 포맷터
def _join_texts(texts, limit=3500):
    """
    full_docs 리스트를 하나의 컨텍스트로 합치되,
    LLM context 길이를 넘지 않도록 제한.
    """
    sep = "\n\n--- 문서 ---\n\n"
    out = ""
    for t in texts:
        if len(out) + len(sep) + len(t) > limit:
            remain = limit - len(out) - len(sep)
            if remain > 0:
                out += sep + t[:remain]
            break
        out += (sep if out else "") + t
    return out


print("✅ LangChain 최신 RAG 체인 준비 완료")


# Query Rewrite Chain
rewrite_prompt = PromptTemplate.from_template("""
Rewrite the user query into a clearer and more specific information search query.
Expand the meaning if necessary, but DO NOT add explanations, reasoning, examples, or parentheses.

Rules:
- Output only the rewritten query
- Include the domain keyword '순천향대학교'
- Make the query semantically richer and more complete
- Do not create facts or answer the question

User query: {query}
Rewritten:
""")

rewrite_chain = rewrite_prompt | ChatOllama(model="llama3.1:8b", temperature=0.1)


def preprocess_query(query: str):
    # 짧고 모호한 경우만 rewrite
    if len(query) <= 6 or query.count(" ") <= 1:
        rewritten = rewrite_chain.invoke({"query": query})
        return rewritten.content if hasattr(rewritten, "content") else rewritten
    return query

# embedding 캐시 생성
emb_cache = dict(zip(chunk_df.index, chunk_df["embedding"]))

def rerank_followup_docs(query, docs):
    """Semantic reranking for follow-up questions."""
    query_emb = embedding_model.embed_query(query)
    query_emb = np.array(query_emb).astype("float32")

    doc_scores = []
    for d in docs:
        chunk_index = int(d.metadata["chunk_index"])  
        emb = np.array(emb_cache[chunk_index]).astype("float32")

        score = np.dot(emb, query_emb)
        doc_scores.append((score, d))

    reranked = [doc for _, doc in sorted(doc_scores, key=lambda x: x[0], reverse=True)]
    return reranked

def cross_encoder_rerank(query, docs, top_k=5):
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)

    reranked = [doc for _, doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]
    return reranked[:top_k]
# 외부에서 호출할 함수 
def chat(query: str):
    global chat_history

    # Query Rewrite 적용
    rewritten_query = preprocess_query(query)
    print("🪄 Rewritten Query:", rewritten_query)

    source_docs = None

    # Follow-up: 이전 문서 내부 검색 시도
    if len(chat_history) > 0:
        last_docs = chat_history[-1].get("docs", [])
        filtered = []

        query_emb = embedding_model.embed_query(rewritten_query)
        query_emb = np.array(query_emb).astype("float32")

        for d in last_docs:
            chunk_index = int(d.metadata["chunk_index"])
            emb = np.array(emb_cache[chunk_index]).astype("float32")

            score = np.dot(emb, query_emb)
            if score > 0.40:  # semantic threshold
                filtered.append((score, d))

        if len(filtered) > 0:
            print("🎯 Follow-up detected → Semantic Filter + Dense Reranking + CrossEncoder Reranking")
            filtered_docs = [d for score, d in sorted(filtered, key=lambda x: x[0], reverse=True)]
            
            # CrossEncoder Reranking 적용
            source_docs = cross_encoder_rerank(rewritten_query, filtered_docs, top_k=5)


    # 필요한 경우 새로 GPU 검색 
    if source_docs is None:
        print("🔍 New search performed")
        source_docs = hybrid_search(rewritten_query, top_k=10)
        print("🔁 CrossEncoder Re-ranking...")
        source_docs = cross_encoder_rerank(rewritten_query, source_docs, top_k=5)

    # 컨텍스트 구성: chunk → original_id → 원문 full text 불러오기
    def get_neighbor_chunks(doc, k=1):
        idx = int(doc.metadata["chunk_index"])
        neighbors = chunk_df[
            (chunk_df["original_id"] == doc.metadata["original_id"]) &
            (chunk_df["chunk_index"].between(idx-k, idx+k))
        ]["content_chunk"].tolist()
        return " ".join(neighbors)

    # partial expansion 적용
    expanded_docs = []
    for doc in source_docs:
        expanded_docs.append(get_neighbor_chunks(doc, k=1))


    context_text = _join_texts(expanded_docs, limit=2000)

    # 히스토리 
    history_text = get_history_text()

    # 프롬프트 생성
    prompt = qa_prompt.format(
        context=context_text,
        question=query,
        chat_history=history_text,
    )

    # LLM 호출
    answer = llm.invoke(prompt)

    # URL 추출 
    top_doc = source_docs[0]
    url = top_doc.metadata.get("url", None)

    # 히스토리 저장 
    chat_history.append({
        "user": query,
        "assistant": answer,
        "docs": source_docs   
    })

    # 결과 반환
    return {
        "answer": answer,
        "url": url,
        "source_documents": source_docs
    }
