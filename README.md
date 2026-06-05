# LLM From Scratch: Fundamentals and Advanced

Wanted to understand how LLMs actually work from the inside before touching any framework. This repo documents that process, starting from implementing attention in numpy all the way to fine-tuning a 7B model with QLoRA.

Built low-level first, framework second. That order was intentional.

---

## Structure

```
llm-fundamentals/
├── 01_attention_from_scratch/
├── 02_transformers_and_tokenization/
├── 03_rag/
│   ├── faiss_rag/
│   └── chromadb_rag/
├── 04_finetuning/
│   ├── distilbert_sentiment/
│   ├── lora_flant5_squad/
│   └── qlora_mistral_alpaca/
└── README.md
```

---

## What is covered

### 01 - Attention From Scratch
Implemented self-attention using only numpy. No PyTorch, no frameworks.

- Created random token embeddings
- Built Q, K, V weight matrices manually
- Computed scaled dot product scores
- Wrote softmax from scratch
- Produced the final attention output

This was the starting point. Understanding the math before using any library that does it automatically.

### 02 - Transformers and Tokenization
Loaded GPT2 tokenizer and model manually using HuggingFace Transformers.

- Explored how BPE tokenization handles normal words, compound words, and Hindi script
- Extracted hidden states layer by layer
- Computed cosine similarity between semantically related and unrelated sentences
- Compared GPT2 embeddings against Sentence Transformers to see the difference in semantic quality
- Explored pipelines: sentiment analysis, NER, zero-shot classification, fill-mask

### 03 - RAG with FAISS
Built a RAG pipeline from scratch without any framework.

- Ingested a PDF and chunked it using a sliding window (1500 chars, 1300 stride)
- Encoded chunks with `all-MiniLM-L6-v2`
- Stored embeddings in FAISS `IndexFlatIP` with L2 normalization
- Queried with cosine similarity, passed top-k chunks as context to Llama 3.1 8B via Groq
- Saved index and chunks to disk for persistence
- Built an interactive chat loop over a 221k character government schemes document

### 03 - RAG with ChromaDB and LangChain
Same RAG concept, different stack.

- ChromaDB as the persistent vector store
- LangChain `RecursiveCharacterTextSplitter` for cleaner chunking with overlap
- HuggingFace embeddings via LangChain wrappers
- `RetrievalQA` chain with a custom `PromptTemplate` forcing answers only from context
- `ConversationalRetrievalChain` with `ConversationBufferMemory` for multi-turn chat
- Tested on the Constitution of India PDF

### 04 - DistilBERT Fine-tuning
Fine-tuned DistilBERT on IMDB sentiment classification two ways.

- Manual PyTorch training loop: custom Dataset class, DataLoader, AdamW optimizer, 3 epochs
- HuggingFace Trainer: TrainingArguments, compute_metrics callback, early stopping
- Same task done both ways on purpose to understand what the Trainer abstracts away

### 04 - LoRA on Flan-T5
Fine-tuned Flan-T5-base on SQuAD v2 for question answering using LoRA via PEFT.

- Config: r=8, alpha=16, targeting q and v attention modules, dropout 0.1
- Handled unanswerable questions by mapping empty answers to "unanswerable"
- Trained on 30k samples with gradient accumulation
- Compared base model output vs fine-tuned output on the same question

### 04 - QLoRA on Mistral 7B
Fine-tuned Mistral-7B-Instruct on the Alpaca instruction dataset using QLoRA.

- 4-bit NF4 quantization via BitsAndBytes with double quantization
- LoRA on q_proj and v_proj, r=8, alpha=16
- SFTTrainer from TRL
- Saved LoRA adapters separately
- Ran on Kaggle 2x T4 GPU (recommended over Colab for long training runs)

---

## GPU Notes

If you are running these notebooks without a good local GPU, use Kaggle over Google Colab. Kaggle lets you connect the kernel directly in VS Code via extension, and it does not cut out mid-training the way Colab does. On 2x T4 GPU, QLoRA training on 3 epochs takes around 7-8 hours. Do not go AFK for more than 40 minutes or the session will disconnect.

Save your weights to the Kaggle notebook output and import them locally for inference.

---

## Stack

Python, NumPy, PyTorch, HuggingFace Transformers, PEFT, TRL, Sentence Transformers, FAISS, ChromaDB, LangChain, Groq, BitsAndBytes

---

## Disclaimer

API keys have been removed from all notebooks. Use environment variables for any API keys before running locally.

---

## Goal

Every framework abstracts something. This repo exists to understand what.
