# 🎓 Soonchunhyang University RAG Chatbot

순천향대학교 공지 및 학사 정보를 자연어 질문으로 탐색할 수 있는  
**RAG(Retrieval-Augmented Generation) 기반 챗봇 시스템**입니다.

사용자는 키워드 검색 없이 질문만으로 원하는 정보를 빠르게 얻을 수 있습니다.

---

## 1. Project Overview

기존의 학교 정보 탐색 방식은 다음과 같은 문제를 가지고 있습니다.

- 공지사항을 직접 찾아야 하는 **수동 검색 구조**
- 커뮤니티 질문은 **응답 속도와 정확성 보장 불가**

👉 이를 해결하기 위해

> **"자연어 질문 → 관련 문서 검색 → 답변 생성"**

구조의 RAG 챗봇을 설계했습니다.

---

## 🎥 Demo

[![Demo Video](./assets/demo.png)](./video/figure1.mp4)

---

## 2. Key Features

-  순천향대학교 정보에 특화된 챗봇
-  Hybrid Search (BM25 + Dense Retrieval)
-  CrossEncoder 기반 Re-ranking
-  Follow-up 질문 대응 (대화 메모리)
-  Query Rewrite로 검색 성능 향상
-  답변 근거 URL 제공 (출처 기반 응답)
-  평균 응답 시간 2~3초

---

## 3. Folder Structure

```
.
├── RAG/
│   ├── backend/
│   ├── frontend/
│   └── data/
├── video/
│   └── figure1
│   └── demo
├── README.md
├── requirements.txt
```

---

## 4. Data

본 프로젝트는 순천향대학교 홈페이지 데이터를 기반으로 구축되었습니다.

- 총 데이터 수: 약 2228건 (2025 기준)

### 데이터 구성

- `sch_metadata.pkl` → 원문 메타데이터
- `chunk_df.pkl` → 청크 단위 문서 + embedding
- `chunk.index` → FAISS 벡터 검색 인덱스

### ⚠️ Note

- 전처리 및 데이터 생성 pipeline 전체 코드는 포함되어 있지 않습니다.
- 대신 실행 가능한 **검색용 데이터 및 인덱스는 포함**되어 있습니다.

---

## 5. Core RAG Pipeline

```
User Query
   ↓
Query Rewrite
   ↓
Hybrid Search (BM25 + Dense)
   ↓
CrossEncoder Re-ranking
   ↓
Follow-up Filtering
   ↓
Context Assembly
   ↓
LLM Answer Generation
   ↓
Response + Source URL
```

---

## 6. Evaluation

| Metric | Value |
|------|------|
| Speed | 2~3초 |
| Recall | 0.850 |
| MRR | 0.825 |

---

## 7. Tech Stack

- Flask
- LangChain
- FAISS
- BM25
- Ollama (llama3.1:8b)
- KoNLPy

---

## 8. Installation

```bash
pip install -r requirements.txt
```

```bash
ollama serve
ollama pull llama3.1:8b
```

---

## ▶️ 9. How to Run

### 1. Ollama 실행

```bash
ollama serve
```

### 2. Backend 실행

```bash
cd RAG/backend
python llm_server.py
```

서버 주소:
```
http://localhost:5001
```

### 3. Frontend 실행

```bash
cd RAG/frontend
python -m http.server 5500
```

---

## 10. API Example

### POST /ask

```json
{
  "question": "장학금 뭐 있어?"
}
```

---

## ⚠️ 11. Limitations

- preprocessing pipeline 미포함
- 최신 데이터 자동 업데이트 미지원

---

## 12. Future Work

- Query Intent Classification
- Domain 별 Index 분리
- 자동 데이터 업데이트

---

## 🙌 13. Summary

본 프로젝트는 순천향대학교 정보 탐색의 비효율을 해결하기 위해  
RAG 기반 챗봇을 구현한 시스템입니다.
추후 의료용 챗봇으로 업그레이드 하고자 합니다.
