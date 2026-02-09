# FsExplorer Agent 🤖📂

**An AI-powered document search agent that explores your files like a human would — scanning, reasoning, and following cross-references.**

> **Unlike traditional RAG systems** that rely on pre-computed embeddings and chunking, this agent dynamically navigates documents to find answers, preserving context and logic.

## Why Agentic Search?

Traditional RAG (Retrieval-Augmented Generation) has limitations:
- **Chunks lose context**: Splitting documents destroys relationships between sections.
- **Cross-references are invisible**: "See Exhibit B" means nothing to vector embeddings.
- **Similarity ≠ Relevance**: Semantic matching misses logical connections.

**FsExplorer uses a human-like three-phase strategy:**
1. **Parallel Scan** 🔍: Previews all documents in a folder at once to identify relevance.
2. **Deep Dive** 📖: Fully parses and reads only the relevant documents.
3. **Backtrack** ↩️: Follows cross-references ("See file X") to previously skipped documents.

---

## Features

- **6 Powerful Tools**: `scan_folder`, `preview_file`, `parse_file`, `read`, `grep`, `glob`
- **Multi-Format Support**: PDF, DOCX, PPTX, XLSX, HTML, Markdown (powered by [Docling](https://github.com/DS4SD/docling))
- **Intelligent Navigation**: Can traverse subdirectories and follow leads.
- **Rich CLI Interface**: Beautiful terminal output with live updates.

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/fs-explorer-agent.git
   cd fs-explorer-agent
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: .\env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download local models** (OCR & Layout Analysis)
   This prevents the agent from hanging on the first run.
   ```bash
   python download_models.py
   ```

## Configuration

Create a `.env` file in the root directory:

```ini
DEEPSEEK_API_KEY=sk-your-api-key
# Optional:
# DEEPSEEK_BASE_URL=https://api.deepseek.com
```

## Usage

Run the agent with a natural language query:

```bash
python -m cli --task "What is the total purchase price?"
```

The agent will:
1. Scan the current directory.
2. Identify relevant files (e.g., "sales_contract.pdf").
3. Read them to find the answer.
4. If a file references another (e.g., "See Addendum A"), it will find and read that too.
5. Output the final answer with **citations**.

## Architecture

- **LangGraph**: Orchestrates the cyclic workflow (Plan → Act → Observe).
- **LangChain**: Handles LLM interactions and tool binding.
- **Docling**: State-of-the-art document parsing and layout analysis.
- **RapidOCR**: Fast and accurate local OCR.

## License

MIT
