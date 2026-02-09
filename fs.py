"""
Filesystem utilities for the FsExplorer agent.
"""
import os
import re
import glob as glob_module
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from docling.document_converter import DocumentConverter

# Configuration
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({
    ".pdf", ".docx", ".doc", ".pptx", ".xlsx", ".html", ".md"
})

DEFAULT_PREVIEW_CHARS = 3000
DEFAULT_SCAN_PREVIEW_CHARS = 1500
MAX_PREVIEW_LINES = 30
DEFAULT_MAX_WORKERS = 4

# Cache
_DOCUMENT_CACHE: dict[str, str] = {}

def clear_document_cache() -> None:
    _DOCUMENT_CACHE.clear()

def _get_cached_or_parse(file_path: str) -> str:
    abs_path = os.path.abspath(file_path)
    cache_key = f"{abs_path}:{os.path.getmtime(abs_path)}"

    if cache_key not in _DOCUMENT_CACHE:
        converter = DocumentConverter()
        result = converter.convert(file_path)
        _DOCUMENT_CACHE[cache_key] = result.document.export_to_markdown()

    return _DOCUMENT_CACHE[cache_key]

def describe_dir_content(directory: str) -> str:
    if not os.path.exists(directory) or not os.path.isdir(directory):
        return f"No such directory: {directory}"

    children = os.listdir(directory)
    if not children:
        return f"Directory {directory} is empty"

    files = []
    directories = []

    for child in children:
        fullpath = os.path.join(directory, child)
        if os.path.isfile(fullpath):
            files.append(fullpath)
        else:
            directories.append(fullpath)

    description = f"Content of {directory}\n"
    description += "FILES:\n- " + "\n- ".join(files)

    if not directories:
        description += "\nThis folder does not have any sub-folders"
    else:
        description += "\nSUBFOLDERS:\n- " + "\n- ".join(directories)

    return description

def read_file(file_path: str) -> str:
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        return f"No such file: {file_path}"

    with open(file_path, "r") as f:
        return f.read()

def grep_file_content(file_path: str, pattern: str) -> str:
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        return f"No such file: {file_path}"

    with open(file_path, "r") as f:
        content = f.read()

    regex = re.compile(pattern=pattern, flags=re.MULTILINE)
    matches = regex.findall(content)

    if matches:
        return f"MATCHES for {pattern} in {file_path}:\n\n- " + "\n- ".join(matches)
    return "No matches found"

def glob_paths(directory: str, pattern: str) -> str:
    if not os.path.exists(directory) or not os.path.isdir(directory):
        return f"No such directory: {directory}"

    search_path = Path(directory) / pattern
    matches = glob_module.glob(str(search_path))

    if matches:
        return f"MATCHES for {pattern} in {directory}:\n\n- " + "\n- ".join(matches)
    return "No matches found"

def preview_file(file_path: str, max_chars: int = DEFAULT_PREVIEW_CHARS) -> str:
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        return f"No such file: {file_path}"

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return (
            f"Unsupported file extension: {ext}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    try:
        full_content = _get_cached_or_parse(file_path)
        preview = full_content[:max_chars]

        total_len = len(full_content)
        if total_len > max_chars:
            preview += (
                f"\n\n[... PREVIEW TRUNCATED. Full document has {total_len:,} "
                f"characters. Use parse_file() to read the complete document ...]"
            )

        return f"=== PREVIEW of {file_path} ===\n\n{preview}"
    except Exception as e:
        return f"Error previewing {file_path}: {e}"

def parse_file(file_path: str) -> str:
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        return f"No such file: {file_path}"

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return (
            f"Unsupported file extension: {ext}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    try:
        return _get_cached_or_parse(file_path)
    except Exception as e:
        return f"Error parsing {file_path}: {e}"

def _preview_single_file(file_path: str, preview_chars: int) -> dict:
    filename = os.path.basename(file_path)
    try:
        content = _get_cached_or_parse(file_path)
        preview = content[:preview_chars]
        return {
            "file": file_path,
            "filename": filename,
            "preview": preview,
            "total_chars": len(content),
            "status": "success"
        }
    except Exception as e:
        return {
            "file": file_path,
            "filename": filename,
            "preview": "",
            "total_chars": 0,
            "status": f"error: {e}"
        }

def scan_folder(
    directory: str,
    max_workers: int = DEFAULT_MAX_WORKERS,
    preview_chars: int = DEFAULT_SCAN_PREVIEW_CHARS,
) -> str:
    if not os.path.exists(directory) or not os.path.isdir(directory):
        return f"No such directory: {directory}"

    doc_files = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path):
            ext = os.path.splitext(item)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                doc_files.append(item_path)

    if not doc_files:
        return (
            f"No supported documents found in {directory}. "
            f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(_preview_single_file, f, preview_chars): f
            for f in doc_files
        }
        for future in as_completed(future_to_file):
            results.append(future.result())

    results.sort(key=lambda x: x["filename"])

    output = []
    output.append("═══════════════════════════════════════════════════════════════")
    output.append(f"  PARALLEL DOCUMENT SCAN: {directory}")
    output.append(f"  Found {len(results)} documents")
    output.append("═══════════════════════════════════════════════════════════════")
    output.append("")

    for i, result in enumerate(results, 1):
        output.append("┌─────────────────────────────────────────────────────────────")
        output.append(f"│ [{i}/{len(results)}] {result['filename']}")
        output.append(f"│ Path: {result['file']}")
        output.append(f"│ Status: {result['status']} | Total size: {result['total_chars']:,} chars")
        output.append("├─────────────────────────────────────────────────────────────")

        if result['status'] == 'success' and result['preview']:
            preview_lines = result['preview'].split('\n')
            for line in preview_lines[:MAX_PREVIEW_LINES]:
                output.append(f"│ {line}")
            if len(preview_lines) > MAX_PREVIEW_LINES:
                output.append("│ ... (preview truncated)")
        else:
            output.append("│ [No preview available]")

        output.append("└─────────────────────────────────────────────────────────────")
        output.append("")

    output.append("═══════════════════════════════════════════════════════════════")
    output.append("  NEXT STEPS:")
    output.append("  1. Assess which documents are RELEVANT to the user's query")
    output.append("  2. Use parse_file() for DEEP DIVE into relevant documents")
    output.append("  3. Watch for cross-references to other docs (may need backtracking)")
    output.append("═══════════════════════════════════════════════════════════════")

    return "\n".join(output)