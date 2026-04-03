from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_chain import chat  

app = Flask(__name__)
CORS(app)  # 프론트엔드 HTML에서 직접 호출 가능하게

# TEST ENDPOINT 추가

@app.route("/test", methods=["GET"])
def test():
    return jsonify({
        "status": "ok",
        "message": "server alive"
    }), 200


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True, silent=True)

    print("📌 받은 RAW JSON:", data)  

    if not data:
        return jsonify({"error": "JSON을 받지 못했습니다."}), 400


    question = data.get("question") or data.get("query") or ""
    question = question.strip()

    if not question:
        return jsonify({
            "error": "질문(question)이 비어 있습니다."
        }), 400

    # chat() 실행 (RAG + Memory 전체)
    result = chat(question)

    answer = result["answer"]
    sources = result["source_documents"]

    # LangChain Document 객체는 그대로 못 보내므로 dict로 변환
    formatted_sources = []
    for doc in sources:
        formatted_sources.append({
            "title": doc.metadata.get("title"),
            "url": doc.metadata.get("url"),
            "category": doc.metadata.get("category"),
            "id": doc.metadata.get("id")
        })

    return jsonify({
        "answer": answer,
        "url": result["url"],         
        "sources": formatted_sources
    })



if __name__ == "__main__":
    # 기본 포트 5001 → 프론트엔드에서 http://localhost:5001/ask 로 요청
    app.run(host="0.0.0.0", port=5001, debug=False)
