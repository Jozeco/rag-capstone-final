# Wikipedia RAG Capstone

A small, local-corpus RAG example with a bounded retrieval agent. The system searches
Wikipedia HTML files with fused BM25 keyword and Chroma vector retrieval, can inspect an
article or follow links, and asks a chat model to answer from passages with citations.

## Design

The app keeps the important steps visible:

1. **Retrieve:** BM25 and vector search each produce candidates from the local
   `Wikipedia/` collection. Min-max normalized scores are fused with equal weights.
2. **Expand only when needed:** The controller can inspect an article or follow its
   links when the initial passages do not answer the question.
3. **Bound the work:** Retrieval stops at five steps, and repeated actions are stopped.
4. **Ground the answer:** The answer model receives the question and retrieved passages,
   and is instructed to cite article filenames and state when evidence is insufficient.
5. **Remember the conversation:** The six most recent successful exchanges help resolve
   follow-up questions. Grounded prior answers can answer a new turn without another search.
6. **Keep source data local:** HTML parsing, indexing, and link traversal happen locally.
   The question, retrieval history, article names, and selected passages are sent to the
   configured model provider.

This intentionally leaves out evaluation dashboards and alternate user-facing pipelines
so the core RAG flow stays easy to follow.

## Run

Use Python 3.11 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Add the provided OpenRouter key to a `.env` file in the project root:

```dotenv
OPENROUTER_API_KEY=your-key
# Optional; defaults to openai/gpt-4o-mini
OPENROUTER_MODEL=openai/gpt-4o-mini
```

Ensure the local Wikipedia HTML collection is in `Wikipedia/`, then start the app:

```bash
python main.py
```

Ask questions in the terminal and enter `quit` to exit. The first startup builds and
caches the BM25 index, vector index, and link graph; later runs reuse valid caches. Building
the vector index calls the configured embedding API and can take several minutes.
