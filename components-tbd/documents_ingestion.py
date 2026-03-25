def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100):
	if not text:
		return []
	if chunk_size <= 0:
		raise ValueError("chunk_size must be > 0")
	if chunk_overlap < 0:
		raise ValueError("chunk_overlap must be >= 0")

	step = max(1, chunk_size - chunk_overlap)
	chunks = []
	start = 0
	text_len = len(text)
	min_boundary = 0.6

	def _boundary_index(start_idx: int, end_idx: int) -> int:
		search_from = start_idx + int((end_idx - start_idx) * min_boundary)

		paragraph = text.rfind("\n\n", search_from, end_idx)
		if paragraph != -1:
			return paragraph

		line_break = text.rfind("\n", search_from, end_idx)
		if line_break != -1:
			return line_break

		for marker in (". ", "! ", "? "):
			sentence = text.rfind(marker, search_from, end_idx)
			if sentence != -1:
				return sentence + 1

		period = text.rfind(".", search_from, end_idx)
		if period != -1:
			return period + 1

		whitespace = text.rfind(" ", search_from, end_idx)
		if whitespace != -1:
			return whitespace

		return end_idx

	while start < text_len:
		end = min(start + chunk_size, text_len)
		if end < text_len:
			natural_end = _boundary_index(start, end)
			if natural_end > start:
				end = natural_end

		chunk = text[start:end].strip()
		if chunk:
			chunks.append(chunk)

		if end >= text_len:
			break
		start += step

	return chunks
