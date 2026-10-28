
from pathlib import Path

# Folder containing cybersecurity knowledge documents
KNOWLEDGE_DIR = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "knowledge_base"
)


def load_documents() -> list[dict]:
    """Load text and Markdown documents from the knowledge-base folder."""
    documents = []

    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

    for file_path in KNOWLEDGE_DIR.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in {".txt", ".md"}:
            try:
                text = file_path.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeError):
                continue

            if text:
                documents.append({
                    "source": str(file_path.relative_to(KNOWLEDGE_DIR)),
                    "text": text,
                })

    return documents
