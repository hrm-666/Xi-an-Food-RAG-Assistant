import re
from typing import Iterable, List

from langchain_core.documents import Document


_DROP_PAGE_PATTERNS = (
    "原创性声明",
    "使用授权说明",
    "目录",
    "参考文献",
    "致谢",
    "附录",
)

_NOISE_LINE_PATTERNS = [
    re.compile(r"^\s*\d+\s*$"),
    re.compile(r"^\s*图\s*[A-Za-z0-9\-.]+\s*.*$"),
    re.compile(r"^\s*表\s*[A-Za-z0-9\-.]+\s*.*$"),
    re.compile(r"^\s*[Ff]-?\d+\s*.*$"),
    re.compile(r"^\s*[\W_]*$"),
]


def _should_drop_page(text: str) -> bool:
    compact = text.replace(" ", "")
    if len(compact) < 30:
        return True

    return any(pattern in compact for pattern in _DROP_PAGE_PATTERNS)


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    cleaned_lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            cleaned_lines.append("")
            continue

        if any(pattern.match(line) for pattern in _NOISE_LINE_PATTERNS):
            continue

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # Merge hard line breaks that usually come from PDF extraction while
    # preserving paragraph boundaries.
    text = re.sub(r"(?<!\n)\n(?!\n)", "", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    return text.strip()


def clean_documents(documents: Iterable[Document]) -> List[Document]:
    cleaned_documents: List[Document] = []

    for doc in documents:
        text = _clean_text(doc.page_content)
        if not text or _should_drop_page(text):
            continue

        metadata = dict(doc.metadata)
        metadata["cleaned"] = True
        cleaned_documents.append(
            Document(page_content=text, metadata=metadata)
        )

    print(f"Cleaned documents: kept {len(cleaned_documents)} pages after filtering.")
    return cleaned_documents
