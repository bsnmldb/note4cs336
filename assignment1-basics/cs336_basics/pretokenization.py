import os
from typing import BinaryIO, List, Dict
import regex as re
import multiprocessing
import pickle


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def merge_pretokenization_results(results: List[Dict[bytes, int]]) -> Dict[bytes, int]:
    if len(results) == 0:
        return {}
    merge_result = results[0]
    for i in range(1, len(results)):
        for token, count in results[i].items():
            if token in merge_result:
                merge_result[token] += count
            else:
                merge_result[token] = count
    return merge_result

def pre_tokenization(
    chunk: str,
    special_tokens: List[str] | None
) -> Dict[bytes, int]:
    if special_tokens is not None:
        delimiter = "|".join(map(re.escape, special_tokens))
        docs = re.split(delimiter, chunk)
        return merge_pretokenization_results([pre_tokenization(doc, None) for doc in docs])
    
    pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    result = {}

    for m in re.finditer(pattern, chunk):
        pretoken = m.group(0).encode("utf-8")

        if pretoken not in result:
            result[pretoken] = 1
        else:
            result[pretoken] += 1

    return result

def pretokenize_worker(file_path: str, start: int, end: int, special_tokens: List[str]) -> Dict[bytes, int]:
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    return pre_tokenization(chunk, special_tokens)

def pretrain_bpe(input_file: str, output_file: str | None, special_tokens: List[str], num_works: int = 32, num_workers: int = 4) -> Dict[bytes, int]:
    with open(input_file, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_works, b"<|endoftext|>")

    with multiprocessing.Pool(num_workers) as pool:
        results = pool.starmap(pretokenize_worker, [(input_file, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])])

    pretokenization_result = merge_pretokenization_results(results)

    if output_file is not None:
        with open(output_file, "wb") as f:
            pickle.dump(pretokenization_result, f)
            
    
    # debug
    # with open("run_res/pretokenization_debug.txt", "w", encoding="utf-8") as f:
    #     for token, count in sorted(pretokenization_result.items(), key=lambda item: -item[1])[:1000]:
    #         f.write(f"{token.decode('utf-8', errors='ignore')}\t{count}\n")
    
    return pretokenization_result

def main():
    pretrain_bpe(
        input_file="data/owt_train.txt",
        output_file="run_res/pretokenization_owt_train.pkl",
        special_tokens=["<|endoftext|>"],
        num_works=32,
        num_workers=4
    )

if __name__ == "__main__":
    main()