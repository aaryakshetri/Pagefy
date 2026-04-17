# Author - Aarya Kshetri

# Pagefy — PDF Assistant

A small Flask app that lets you upload a PDF and then summarize it, ask questions, or extract structured lists — powered by a tiny RAG pipeline using free/local tools.

## How it works

1. **Upload** → PDF text is extracted with `pypdf`, split into overlapping chunks, and embedded with `all-MiniLM-L6-v2` (runs locally, free).
2. **Ask** → Your question is embedded and matched against the chunks via cosine similarity. The top 4 chunks are sent to the LLM as context.
3. **Summarize / Extract** → Sends the document to the LLM with a task-specific prompt. Long documents are handled with map-reduce (summarize sections first, then combine).

## Tech stack

| Component | Tool | Cost |
|-----------|------|------|
| PDF extraction | `pypdf` | Free |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) | Free, local |
| Retrieval | Cosine similarity (NumPy) | Free, local |
| Generation | Groq API (Llama 3.3 70B) | Free tier |
| Web framework | Flask | Free |

## Setup

### 1. Install dependencies

```bash
cd pagefy
python -m venv venv
source venv/bin/activate              # macOS/Linux
venv\Scripts\activate                 # Windows

pip install -r requirements.txt
```

### 2. Get a free Groq API key

- Sign up at [console.groq.com](https://console.groq.com) (no credit card required)
- Go to **API Keys** → create a new key
- Copy it

### 3. Create a `.env` file

Create a file named `.env` in the project root with:

```
GROQ_API_KEY=gsk_your_key_here
```

⚠️ **Windows users:** Do not use Notepad to create this file — it saves with a UTF-16 BOM that breaks `python-dotenv`. Instead, use VS Code, or run this in PowerShell:

```powershell
Set-Content -Path .env -Value "GROQ_API_KEY=gsk_your_key_here" -Encoding ASCII
```

Or in Command Prompt:

```cmd
echo GROQ_API_KEY=gsk_your_key_here> .env
```

### 4. Run

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000).

> **First run is slow** — `sentence-transformers` downloads the embedding model (~90MB) once, then it's fast.

## File layout

```
pagefy/
├── app.py              # Flask routes
├── rag_utils.py        # chunk / embed / top_k
├── pdf_utils.py        # PDF text extraction
├── llm.py              # Groq API wrapper
├── requirements.txt
├── .env                # your API key (do not commit!)
└── templates/
    └── index.html      # UI
```

## Features

- **Upload PDFs** up to 25 MB
- **Question answering** — retrieves relevant chunks and generates grounded answers with source excerpts
- **Summarization** — full-document summary (auto-switches to map-reduce for long docs)
- **Extract lists** — key points, action items, suggested questions, or named entities

## Known limitations

### Rate limits (Groq free tier)

The free tier has **tokens-per-minute (TPM) caps** that can cause `413` errors on large documents:

| Model | Free-tier TPM |
|-------|---------------|
| `llama-3.3-70b-versatile` | 12,000 |
| `llama-3.1-8b-instant` | 6,000 |

**Practical impact:**
- Q&A works on documents of any size (only 4 chunks sent per query)
- Summarize/extract works reliably on PDFs **under ~30 pages**
- Longer PDFs use map-reduce with an 8-second sleep between section calls, which handles larger docs but takes longer
- Very long books (500+ pages) may still hit limits — consider upgrading to Groq's Dev tier or swapping to a local Ollama setup

### No OCR

`pypdf` only extracts text from PDFs that already contain a text layer. **Scanned PDFs or image-based PDFs won't work.** To add OCR, install `pytesseract` + `pdf2image` and modify `pdf_utils.py`.

### In-memory storage

The document index (`STORE` dict in `app.py`) lives in memory. **Restarting the server wipes all uploaded documents.** For persistence, pickle `STORE` to disk or store embeddings per-doc with `np.save`.

### Single-user by design

No authentication, no session isolation. Multiple users hitting the same server can see each other's `doc_id` values if they guess them. Fine for local/personal use, not for production.

### Character-based chunking

`chunk()` splits on raw character counts, which can cut mid-sentence or mid-word. Retrieval still works reasonably well thanks to 100-char overlap, but semantic chunking (by paragraph or sentence) would improve quality.

### Context cap for whole-doc tasks

Summarize and extract are capped at the first `MAX_CONTEXT_CHARS` (30,000) of the document for direct calls. Beyond that, map-reduce kicks in. Neither approach is perfect for documents where late content is critical — if your doc buries conclusions at the end, the map-reduce merge step may lose some nuance.

## Troubleshooting

**`TemplateNotFound: index.html`**
Make sure `index.html` is inside a `templates/` folder next to `app.py`, and that you're running `python app.py` from the project root.

**`Could not resolve authentication method`**
Your `GROQ_API_KEY` isn't loading. Check the `.env` file encoding (must be UTF-8 or ASCII, no BOM).

**`UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff`**
Your `.env` was saved as UTF-16 (usually by Windows Notepad). Recreate it using VS Code or the PowerShell/cmd commands in the setup section.

**`413 Request too large`**
You've hit Groq's TPM limit. Either use a smaller PDF, increase the sleep interval in the map-reduce loop, or upgrade to the Dev tier.

## Possible next steps

- **Page-level citations** — attach page numbers to chunks so answers cite "page 7"
- **Streaming responses** — use Groq's SSE streaming so answers appear as they generate
- **Persistence** — pickle `STORE` to disk, reload on startup
- **Multi-doc search** — query across several PDFs at once
- **Semantic chunking** — split on paragraph/sentence boundaries instead of raw chars
- **Switch to Ollama** — fully local, no rate limits, just needs a ~4GB model download

## Changelog

- **v0.3** — Added map-reduce summarization for documents longer than 20k chars
- **v0.2** — Switched from Anthropic API to Groq free tier; added `python-dotenv` support
- **v0.1** — Initial Flask app with upload, ask, summarize, and extract endpoints
