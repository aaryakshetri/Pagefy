import os
import uuid
import time
from flask import Flask, request, render_template, jsonify

from rag_utils import chunk, embed, top_k
from pdf_utils import extract_text
from llm import generate

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB cap
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# In-memory index store. Keyed by doc_id so multiple PDFs can coexist.
STORE = {}

# Cap context sent to the LLM for whole-doc tasks (~50k chars ≈ 12k tokens).
MAX_CONTEXT_CHARS = 30_000


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are allowed"}), 400

    doc_id = str(uuid.uuid4())
    path = os.path.join(app.config["UPLOAD_FOLDER"], f"{doc_id}.pdf")
    file.save(path)

    try:
        text = extract_text(path)
        if not text:
            return jsonify({"error": "Could not extract any text from this PDF "
                                     "(it may be a scanned image)"}), 400

        chunks = chunk(text)
        embs = embed(chunks)

        STORE[doc_id] = {
            "chunks": chunks,
            "embs": embs,
            "filename": file.filename,
            "full_text": text,
        }

        return jsonify({
            "doc_id": doc_id,
            "filename": file.filename,
            "num_chunks": len(chunks),
            "char_count": len(text),
        })
    except Exception as e:
        return jsonify({"error": f"Failed to process PDF: {e}"}), 500


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json() or {}
    doc_id = data.get("doc_id")
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "Question is required"}), 400
    if doc_id not in STORE:
        return jsonify({"error": "Document not found"}), 404

    doc = STORE[doc_id]
    results = top_k(question, doc["chunks"], doc["embs"], k=4)

    context = "\n\n---\n\n".join(
        f"[Excerpt {i+1}]\n{text}" for i, (text, _) in enumerate(results)
    )

    prompt = (
        "Use only the excerpts below to answer the question. "
        "If the answer isn't in the excerpts, say so clearly.\n\n"
        f"EXCERPTS:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "ANSWER:"
    )

    answer = generate(prompt, max_tokens=800)

    return jsonify({
        "answer": answer,
        "sources": [
            {"text": t[:220].replace("\n", " "), "score": round(s, 3)}
            for t, s in results
        ],
    })


@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json() or {}
    doc_id = data.get("doc_id")

    if doc_id not in STORE:
        return jsonify({"error": "Document not found"}), 404

    doc = STORE[doc_id]
    text = doc["full_text"]

    # Short doc — summarize directly
    if len(text) <= 30_000:
        prompt = (
            "Write a clear, well-structured summary of the document below. "
            "Cover the main arguments, key findings, and conclusions. "
            "Aim for 4-6 paragraphs.\n\n"
            f"DOCUMENT:\n{text}\n\nSUMMARY:"
        )
        return jsonify({"summary": generate(prompt, max_tokens=1500)})

    # Long doc — map-reduce: summarize sections, then combine
    # Split into ~25k char sections so each fits within limits
    sections = [text[i:i + 25_000] for i in range(0, len(text), 25_000)]
    partial_summaries = []

    for i, section in enumerate(sections):
        prompt = (
            f"Summarize this section (part {i+1} of {len(sections)}) of a longer document. "
            "Focus on key facts, arguments, and conclusions. Keep it to 2-3 paragraphs.\n\n"
            f"SECTION:\n{section}\n\nSUMMARY:"
        )
        partial_summaries.append(generate(prompt, max_tokens=600))
        time.sleep(6)  # stay under per-minute limits

    # Combine the partial summaries into one final summary
    combined = "\n\n".join(
        f"[Part {i+1}]\n{s}" for i, s in enumerate(partial_summaries)
    )
    final_prompt = (
        "Combine these section summaries into one cohesive 4-6 paragraph summary "
        "of the full document.\n\n"
        f"SECTION SUMMARIES:\n{combined}\n\nFINAL SUMMARY:"
    )
    return jsonify({"summary": generate(final_prompt, max_tokens=1500)})


@app.route("/extract", methods=["POST"])
def extract():
    data = request.get_json() or {}
    doc_id = data.get("doc_id")
    list_type = data.get("type", "key_points")

    if doc_id not in STORE:
        return jsonify({"error": "Document not found"}), 404

    prompts = {
        "key_points": "Extract the 5-10 most important key points from this document as a bulleted list.",
        "action_items": "Extract all action items, tasks, decisions, or next steps mentioned in this document as a bulleted list. If none exist, say so.",
        "questions": "Generate 5-8 insightful questions a careful reader would want to ask about this document.",
        "entities": "Extract all important names, organizations, places, and dates mentioned in this document, grouped by category.",
    }

    instruction = prompts.get(list_type, prompts["key_points"])
    text = STORE[doc_id]["full_text"][:MAX_CONTEXT_CHARS]

    prompt = f"{instruction}\n\nDOCUMENT:\n{text}\n\nOUTPUT:"

    result = generate(prompt, max_tokens=1500)
    return jsonify({"result": result, "type": list_type})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
