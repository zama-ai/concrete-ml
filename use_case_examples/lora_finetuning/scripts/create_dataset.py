import json
import re
from pathlib import Path

from transformers import AutoTokenizer


def init_tokenizer():
    return AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")


def chunk_text_by_tokens(text, tokenizer, max_tokens=128):
    """Split text into chunks that don't exceed max_tokens with overlap."""
    overlap_tokens = max_tokens // 2
    tokens = tokenizer.encode(text)
    chunks = []

    # Start indices for each chunk
    start_idx = 0

    while start_idx < len(tokens):
        # Calculate end index for current chunk
        end_idx = min(start_idx + max_tokens, len(tokens))

        # Get current chunk
        current_chunk = tokens[start_idx:end_idx]
        chunk_text = tokenizer.decode(current_chunk, skip_special_tokens=True)

        if chunk_text.strip():
            chunks.append(chunk_text)

        # Move start_idx forward by (max_tokens - overlap_tokens)
        start_idx += max_tokens - overlap_tokens

        # If the remaining text is shorter than the overlap, we're done
        if len(tokens) - start_idx < overlap_tokens:
            break

    return chunks


def split_code_into_snippets(code):
    # Split code into functions, classes, and other logical blocks
    pattern = re.compile(r"^\s*(def |class )", re.MULTILINE)
    indices = [match.start() for match in pattern.finditer(code)]
    indices.append(len(code))
    snippets = [code[indices[i] : indices[i + 1]] for i in range(len(indices) - 1)]
    return snippets


def process_code_file(code_file_path, tokenizer, max_tokens=128):
    with open(code_file_path, "r", encoding="utf-8") as file:
        code = file.read()
    snippets = split_code_into_snippets(code)
    # Further split snippets if they exceed token limit
    tokenized_snippets = []
    for snippet in snippets:
        tokenized_snippets.extend(chunk_text_by_tokens(snippet, tokenizer, max_tokens))
    return tokenized_snippets


def process_documentation_file(doc_file_path, tokenizer, max_tokens=128):
    with open(doc_file_path, "r", encoding="utf-8") as file:
        documentation = file.read()
    snippets = documentation.split("\n\n")
    # Further split snippets if they exceed token limit
    tokenized_snippets = []
    for snippet in snippets:
        tokenized_snippets.extend(chunk_text_by_tokens(snippet, tokenizer, max_tokens))
    return tokenized_snippets


def save_to_jsonl(snippets, output_file_path):
    with open(output_file_path, "w", encoding="utf-8") as outfile:
        for snippet in snippets:
            snippet = snippet.strip()
            if snippet:
                json_line = json.dumps({"text": snippet})
                outfile.write(json_line + "\n")


def main():
    # Get the absolute path to the script's location
    script_dir = Path(__file__).resolve().parent

    # Calculate paths relative to the script location
    output_dir = script_dir.parent / "data_finetune"

    # Paths to your code and documentation files
    code_file_path = output_dir / "raw_cml_1.7.0_examples.txt"
    output_file_path = output_dir / "dataset.jsonl"

    # Initialize tokenizer
    tokenizer = init_tokenizer()
    max_tokens = 128

    # Process code files with token control
    code_snippets = process_code_file(code_file_path, tokenizer, max_tokens)

    # Combine snippets
    all_snippets = code_snippets

    # Save to dataset.jsonl
    save_to_jsonl(all_snippets, output_file_path)
    print(f"Dataset saved to {output_file_path}")


if __name__ == "__main__":
    main()
