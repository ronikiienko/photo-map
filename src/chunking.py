def chunk_uniform(text, tokenizer, size, overlap):
    """Cut text into windows of `size` tokens; neighbouring windows share `overlap` tokens.
    Returns [(start_char, chunk_text)], where start_char is the chunk's position in `text`."""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc.input_ids, enc.offset_mapping  # offsets[i] = (start, end) chars of token i
    if not ids:
        return []
    step = size - overlap
    # stop before a window that would contain only overlap
    starts = range(0, max(len(ids) - overlap, 1), step)
    return [(offsets[i][0], tokenizer.decode(ids[i : i + size])) for i in starts]
