"""
Pre-download all required models (RapidOCR + Docling).
Run this once before using the agent:
    python download_models.py
"""
from rich.console import Console
from rich.panel import Panel

console = Console()

def download_rapidocr_models():
    """Download RapidOCR detection, classification, and recognition models."""
    console.print("[bold cyan]📥 Downloading RapidOCR models (torch)...[/]")
    from rapidocr import RapidOCR
    from rapidocr.utils.typings import EngineType
    engine = RapidOCR(params={
        "Det.engine_type": EngineType.TORCH,
        "Cls.engine_type": EngineType.TORCH,
        "Rec.engine_type": EngineType.TORCH,
    })
    console.print("[bold green]✅ RapidOCR models ready![/]")

def download_docling_models():
    """Download Docling layout analysis models from HuggingFace."""
    console.print("[bold cyan]📥 Downloading Docling models (layout + TableFormer)...[/]")
    from docling.document_converter import DocumentConverter
    converter = DocumentConverter()
    console.print("[bold green]✅ Docling models ready![/]")

if __name__ == "__main__":
    console.print(Panel("[bold]🔧 Pre-downloading all required models[/]", border_style="blue"))

    download_rapidocr_models()
    download_docling_models()

    console.print(Panel("[bold green]🎉 All models downloaded! You can now run the agent.[/]", border_style="green"))
