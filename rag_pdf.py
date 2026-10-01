"""
RAG over a PDF using only core Hugging Face (transformers) + PyTorch.

Pipeline:
  PDF -> text -> chunks
      -> embeddings   (AutoTokenizer + AutoModel + mean pooling)
      -> vector DB    (torch tensor, saved with torch.save)
      -> question     -> cosine similarity -> top-k chunks
      -> answer       (AutoModelForCausalLM + model.generate)

Usage:
    python rag_pdf_video.py Gita.pdf
"""
import os
import sys

import torch
import torch.nn.functional as F
from pypdf import PdfReader
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, TextStreamer

# ---------------- Settings ----------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # loaded with plain AutoModel
LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"                 # small and fast on CPU
CHUNK_SIZE = 800        # characters
CHUNK_OVERLAP = 150
TOP_K = 3               # chunks given to the LLM as context
SHOW_SOURCES = 1        # how many source chunks to display
MIN_SCORE = 0.30        # below this similarity, the document is treated as not relevant
MAX_NEW_TOKENS = 400    # answer length (lower = faster)
MAX_LENGTH = 256        # token truncation for embeddings
DB_FILE = "vector_db_v2.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# ---------------- Step 1: PDF -> chunks ----------------
def load_pdf(path):
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


GOOD_CHARS = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \n.,;:'\"()-?!"
)


def is_clean(piece, threshold=0.93):
    """Skip chunks of garbled text (e.g. Sanskrit in a legacy font)."""
    good = sum(1 for ch in piece if ch in GOOD_CHARS)
    return good / len(piece) >= threshold


def chunk_text(text):
    chunks, start = [], 0
    while start < len(text):
        piece = text[start:start + CHUNK_SIZE].strip()
        if piece and is_clean(piece):
            chunks.append(piece)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ---------------- Step 2: embeddings (tokenizer + model + torch) ----------------
embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
embed_model = AutoModel.from_pretrained(EMBED_MODEL).to(device).eval()


@torch.no_grad()
def embed(texts, batch_size=32):
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = embed_tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(device)
        out = embed_model(**inputs).last_hidden_state          # (batch, tokens, hidden)

        # mean pooling that ignores padding tokens
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        all_vecs.append(F.normalize(pooled, p=2, dim=1).cpu())
    return torch.cat(all_vecs)                                  # (num_texts, hidden)


# ---------------- Step 3: vector DB (torch tensor on disk) ----------------
def build_vector_db(pdf_path):
    if os.path.exists(DB_FILE):
        db = torch.load(DB_FILE)
        if db["source"] == pdf_path:
            print(f"Loaded existing vector DB ({len(db['chunks'])} chunks).")
            return db

    print("Reading PDF...")
    chunks = chunk_text(load_pdf(pdf_path))
    print(f"Creating vector DB from {len(chunks)} chunks...")
    db = {"source": pdf_path, "chunks": chunks, "embeddings": embed(chunks)}
    torch.save(db, DB_FILE)
    return db


def retrieve(db, question, k=TOP_K):
    q_vec = embed([question])                                   # (1, hidden)
    scores = (db["embeddings"] @ q_vec.T).squeeze(1)            # cosine similarity
    top = torch.topk(scores, k=min(k, len(db["chunks"])))
    return [
        (db["chunks"][i], s)
        for i, s in zip(top.indices.tolist(), top.values.tolist())
    ]


# ---------------- Step 4: generation (AutoModelForCausalLM.generate) ----------------
print(f"Loading LLM ({LLM_MODEL})... first run downloads it.")
llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
llm = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
).to(device).eval()


@torch.no_grad()
def generate_answer(question, context_chunks):
    context = "\n\n---\n\n".join(context_chunks)
    messages = [
        {
            "role": "system",
            "content": (
                "You answer questions about a document using ONLY the context provided. "
                "Rules: "
                "1) Use only information stated in the context. Do NOT add outside "
                "knowledge, general advice, or guesses. "
                "2) Write a clear answer in 2-3 paragraphs, explaining the teachings "
                "found in the context in simple language, and mention verse numbers "
                "if they appear in the context. "
                "3) If the context does not answer the question, reply exactly: "
                "I could not find this in the document."
            ),
        },
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    inputs = llm_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)

    streamer = TextStreamer(llm_tokenizer, skip_prompt=True, skip_special_tokens=True)
    output_ids = llm.generate(
        **inputs,
        streamer=streamer,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.2,
        top_p=0.9,
        repetition_penalty=1.1,
    )
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]   # drop the prompt
    return llm_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------- Main ----------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python rag_pdf_video.py your_file.pdf")

    db = build_vector_db(sys.argv[1])
    print("Vector DB ready!\n")

    while True:
        q = input("Ask a question (or 'exit'): ").strip()
        if q.lower() in ("exit", "quit", ""):
            break
        results = retrieve(db, q)                       # list of (chunk, score)
        best_score = results[0][1]

        if best_score < MIN_SCORE:
            print(f"\nAnswer:\nI could not find this in the document. "
                  f"(best match score {best_score:.2f} is below {MIN_SCORE})\n")
            continue

        chunks = [c for c, _ in results]
        print("\nAnswer:")
        generate_answer(q, chunks)   # answer is streamed to the screen while generating

        print(f"\nTop source (similarity {best_score:.2f}):")
        for c in chunks[:SHOW_SOURCES]:
            print("  -", c[:200].replace("\n", " "), "...")
        print()