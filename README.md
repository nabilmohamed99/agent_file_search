# FsExplorer Agent 🤖📂

**An AI-powered autonomous agent that explores your local file system to answer complex questions.**

> **Unlike traditional RAG systems** that rely on pre-computed embeddings and chunking, this agent dynamically navigates your files like a human would — scanning directories, reading content, and following cross-references to build a complete answer.

---

## 🚀 The Approach (How it Works)

This project implements a **Plan-Act-Observe** loop using **LangGraph**. The agent is not just a chatbot; it's a state machine that moves through specific stages to solve a task.

### 1. The Workflow
The agent follows a cyclical integration pattern:
1. **Analyze Task**: The LLM understands the user's question and current context.
2. **Select Tool**: It decides which action to take next (e.g., "I need to see what files are in this folder").
3. **Execute Tool**: The system runs the Python function (e.g., `scan_folder`, `read_file`).
4. **Observe Output**: The tool returns raw data (file lists, text content, OCR results).
5. **Reason & Loop**: The LLM analyzes the output.
   - *Is this relevant?* -> Read deeper.
   - *Is it a dead end?* -> Backtrack or try another folder.
   - *Is the answer found?* -> Stop and report.

### 2. Key Capabilities
- **Parallel Scanning** 🔍: Can preview multiple files in a directory simultaneously to filter for relevance before reading.
- **Deep Dive Reading** 📖: Uses **Docling** and **RapidOCR** to parse complex documents (PDFs, images, tables) into clean Markdown.
- **Self-Correction** 🛠️: If the LLM generates invalid JSON or a tool fails, the agent captures the error and retries automatically.
- **Memory**: Maintains a conversation history and a "scratchpad" of findings to synthesize the final answer.

---

## 🛠️ Architecture

- **Orchestrator**: [LangGraph](https://langchain-ai.github.io/langgraph/) manages the cyclic workflow.
- **LLM**: Powered by **DeepSeek-V3** (via API) for reasoning and JSON generation.
- **Document Parsing**: [Docling](https://github.com/DS4SD/docling) for high-fidelity PDF/Document conversion.
- **OCR Engine**: [RapidOCR](https://github.com/RapidAI/RapidOCR) for extracting text from images locally (CPU/GPU).
- **Tooling**: Custom Python tools for filesystem interaction (`os`, `glob`, `grep`).

---

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/agent_file_search.git
   cd agent_file_search
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv env
   .\env\Scripts\activate  # Windows
   # source env/bin/activate  # Mac/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download local models** (Critical Step)
   This pre-downloads the OCR and Layout models so the agent doesn't hang on first run.
   ```bash
   python download_models.py
   ```

5. **Configure Environment**
   Create a `.env` file in the root directory:
   ```ini
   DEEPSEEK_API_KEY=sk-your-api-key
   # Optional: DEEPSEEK_BASE_URL=https://api.deepseek.com
   ```

---

## 💻 Usage

Run the CLI with your question:

```bash
python -m cli --task "What is the total purchase price defined in the logic agreement?"
```

**What happens next:**
1. The agent starts in the current directory (`.`).
2. It scans for files containing "agreement", "price", or "logic".
3. It reads the most relevant PDF/Doc.
4. It extracts the answer and cites the source.

---

## 🔧 Troubleshooting

### "ValueError: DEEPSEEK_API_KEY not found"
- Ensure you created the `.env` file.
- We use `python-dotenv` to load it automatically. Try running `pip install python-dotenv`.

### "RapidOCR / Docling models downloading..."
- If the agent seems stuck at startup, it's downloading 500MB+ of models.
- Run `python download_models.py` to see the progress explicitly.

### Symlink Warnings (Windows)
- HuggingFace cache uses symlinks. On Windows, you may see warnings. You can ignore them or enable Developer Mode in Windows Settings.

---

## License
MIT
