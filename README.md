# 📄 PDF Question Answering with RAG (Hugging Face + PyTorch)

A simple **Retrieval-Augmented Generation (RAG)** project that lets you ask questions about a PDF and get answers generated from its content.

Everything is built with only **core Hugging Face (`transformers`) and PyTorch**. No LangChain, no paid APIs, no API keys. All models run locally on your machine.

---

## ✨ Features

- Reads text from any text-based PDF
- Splits the document into overlapping chunks and filters out garbled text
- Converts chunks to embeddings with a Hugging Face model
- Stores them in a simple vector database (a saved PyTorch tensor)
- Retrieves the most relevant chunks using cosine similarity
- Refuses to answer when nothing relevant is found (no made-up answers)
- Generates the answer with a Hugging Face LLM, streamed live to the terminal
- Runs offline after the first model download

---

## 🧠 How It Works

```
PDF → Extract text → Chunking → Embeddings → Vector DB (.pt file)
                                                   ↓
Question → Embedding → Cosine similarity → Relevance check → Top-K chunks
                                                   ↓
                          Chunks + Question → LLM → Final answer
```

| Step | What happens | Tool used |
|---|---|---|
| 1. Read PDF | Extract raw text | `pypdf` |
| 2. Chunking | Fixed-size chunks (800 chars) with 150-char overlap; garbled chunks are skipped | Plain Python |
| 3. Embeddings | Text → vectors using tokenizer + model + mean pooling | `sentence-transformers/all-MiniLM-L6-v2` via `AutoTokenizer` / `AutoModel` |
| 4. Vector DB | Embeddings stored as a tensor and saved to disk | `torch.save` |
| 5. Retrieval | Cosine similarity, top 3 chunks | PyTorch |
| 6. Relevance check | If the best similarity score is below the threshold, answers "not found" | Plain Python |
| 7. Generation | Answer written from the retrieved context only | `Qwen/Qwen2.5-0.5B-Instruct` via `AutoModelForCausalLM.generate()` |

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **PyTorch**: tensors, similarity search, storage
- **Hugging Face Transformers**: tokenizers and models
- **pypdf**: PDF text extraction

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac / Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run

```bash
python rag_pdf_video.py your_file.pdf
```

The first run downloads the models and builds the vector database. Later runs with the same PDF load the saved database instantly.

### 5. Ask questions

```
Ask a question (or 'exit'): What does Krishna say about desire and anger?

Answer:
...

Top source (similarity 0.52):
  - ...
```

Type `exit` to quit.

---

## ⚙️ Configuration

Edit these settings at the top of `rag_pdf_video.py`:

| Setting | Default | Meaning |
|---|---|---|
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Embedding model |
| `LLM_MODEL` | `Qwen2.5-0.5B-Instruct` | Answer model. Use `Qwen2.5-1.5B-Instruct` for better answers if you have a GPU or a fast CPU |
| `CHUNK_SIZE` | 800 | Characters per chunk |
| `CHUNK_OVERLAP` | 150 | Characters shared between chunks |
| `TOP_K` | 3 | Chunks given to the LLM as context |
| `SHOW_SOURCES` | 1 | Source chunks displayed under the answer |
| `MIN_SCORE` | 0.30 | Minimum similarity needed to answer. Lower it if good questions get "not found", raise it if off-topic questions get answered |
| `MAX_NEW_TOKENS` | 400 | Maximum answer length (lower = faster) |

---

## 💡 Tips for Better Answers

- Ask using the **document's own vocabulary**. Semantic search works best when your question matches the topics in the text.
- Each question is independent. The system has **no chat memory**.
- Check the printed similarity score. A low score means the retrieval was weak.

---

## ⚠️ Limitations

- Works only with **text-based PDFs**. Scanned PDFs (images) need OCR first.
- PDFs with unusual fonts can extract as garbled text. The code skips such chunks, which may remove some content.
- Small local models can make mistakes or ignore instructions. Answers are only as good as the retrieved chunks.
- On CPU, answers take a while to generate. A GPU (or Google Colab) makes it much faster.

---

## 🔮 Future Improvements

- Use a dedicated vector database (FAISS / ChromaDB)
- Try a stronger embedding model (e.g. `BAAI/bge-small-en-v1.5`)
- Add a web interface (Gradio / Streamlit)
- Support multiple PDFs and chat history
- Add OCR for scanned documents
- Use a larger LLM on GPU

---

## 👤 Author

**<Your Name>**
MCA, <Your College>
Minor Project

---

## 📜 License

This project is for educational purposes.
