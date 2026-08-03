def chunk_text(text: str, max_chars: int = 6000) -> list[str]:
    cleaned = (text or "").strip()
    if not cleaned or len(cleaned) <= max_chars:
        return [cleaned] if cleaned else [""]

    # Prefer blank-line paragraphs, then single newlines (dialogue turns),
    # else hard character slices.
    if "\n\n" in cleaned:
        parts = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
        joiner = "\n\n"
    elif "\n" in cleaned:
        parts = [p.strip() for p in cleaned.split("\n") if p.strip()]
        joiner = "\n"
    else:
        parts = [cleaned]
        joiner = "\n"

    if not parts:
        parts = [cleaned]

    chunks: list[str] = []
    current = ""
    for part in parts:
        candidate = f"{current}{joiner}{part}".strip() if current else part
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        if len(part) <= max_chars:
            current = part
            continue
        for i in range(0, len(part), max_chars):
            piece = part[i : i + max_chars].strip()
            if piece:
                chunks.append(piece)
        current = ""
    if current:
        chunks.append(current)
    return chunks or [cleaned]
