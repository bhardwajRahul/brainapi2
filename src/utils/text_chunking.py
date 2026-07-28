def chunk_text(text: str, max_chars: int = 6000) -> list[str]:
    cleaned = (text or "").strip()
    if not cleaned or len(cleaned) <= max_chars:
        return [cleaned] if cleaned else [""]
    paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [cleaned]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(paragraph) <= max_chars:
            current = paragraph
            continue
        for i in range(0, len(paragraph), max_chars):
            piece = paragraph[i : i + max_chars].strip()
            if piece:
                chunks.append(piece)
        current = ""
    if current:
        chunks.append(current)
    return chunks or [cleaned]
