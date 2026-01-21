from typing import Dict, Tuple, List, Iterable, Iterator
import pickle
import regex as re

class BPETokenizer:
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: List[str] | None = None):
        """
        Initializes the BPE Tokenizer with the given vocabulary and merges.
        """
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.reverse_vocab = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}
        if special_tokens is None:
            self.special_tokens = []
        else:
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)

    @classmethod
    def from_files(cls, vocab_merges_filepath: str, special_tokens: List[str] | None = None) -> 'BPETokenizer':
        """
        Loads the BPE Tokenizer from the given vocabulary and merges files.
        """
        with open(vocab_merges_filepath, "rb") as f:
            data = pickle.load(f)
        vocab, merges = data
        assert vocab is not None, "Vocabulary not found in the provided file."
        assert merges is not None, "Merges not found in the provided file."
        return cls(vocab, merges, special_tokens)
    
    def _pre_tokenize(self, text: str) -> List[bytes]:
        pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        tokens = []
        for m in re.finditer(pattern, text):
            pretoken = m.group(0).encode("utf-8")
            tokens.append(pretoken)
        return tokens
    
    def _apply_merge(self, tokens: bytes) -> List[bytes]:
        # tokens = [bytes([b]) for b in tokens]
        # for merge in self.merges:
        #     merged_token = merge[0] + merge[1]
        #     i = 0
        #     new_tokens = []
        #     while i < len(tokens):
        #         if i < len(tokens) - 1 and tokens[i] == merge[0] and tokens[i + 1] == merge[1]:
        #             new_tokens.append(merged_token)
        #             i += 2
        #         else:
        #             new_tokens.append(tokens[i])
        #             i += 1
        #     tokens = new_tokens
        #     if len(tokens) == 1:
        #         break
        # return tokens
        tokens = [bytes([b]) for b in tokens]
        while True:
            candidate_pairs: Dict[Tuple[bytes, bytes], List[int]] = {}
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                candidate_pairs[pair] = candidate_pairs.get(pair, []) + [i]
            if not candidate_pairs:
                break
            best_pair = min(
                candidate_pairs.keys(),
                key=lambda pair: self.merge_ranks.get(pair, float('inf'))
            )
            if best_pair not in self.merge_ranks:
                break
            positions = candidate_pairs[best_pair]
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i in positions:
                    new_tokens.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
            if len(tokens) == 1:
                break
        return tokens
            
            
    def _convert_tokens_to_ids(self, tokens: List[bytes]) -> List[int]:
        token_ids = []
        for token in tokens:
            if token in self.reverse_vocab:
                token_id = self.reverse_vocab[token]
            else:
                raise ValueError(f"Token {token} not found in vocabulary.")
            token_ids.append(token_id)
        return token_ids
    
    def encode(self, text: str) -> List[int]:
        """
        Encodes the given text into a list of token IDs using BPE.
        """
        delimiter = "|".join(map(re.escape, self.special_tokens))
        pattern = f"({delimiter})"
        texts = re.split(pattern, text) if self.special_tokens else [text]
        encoded_token_ids = []
        for text in texts:
            if text in self.special_tokens:
                token_bytes = text.encode("utf-8")
                assert token_bytes in self.reverse_vocab, f"Special token {text} not in vocabulary."
                token_id = self.reverse_vocab.get(token_bytes)
                encoded_token_ids.append(token_id)
            else:
                pre_tokens = self._pre_tokenize(text)
                bpe_tokens = []
                for pre_token in pre_tokens:
                    bpe_token_part = self._apply_merge(pre_token)
                    bpe_tokens.extend(bpe_token_part)
                token_ids = self._convert_tokens_to_ids(bpe_tokens)
                encoded_token_ids.extend(token_ids)
        return encoded_token_ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Encodes an iterable of strings into a flat iterator of token IDs using BPE.
        """
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id
            
    def decode(self, ids: List[int]) -> str:
        """
        Decodes a list of token IDs back into the original text using BPE.
        """
        results = []
        for token_id in ids:
            if token_id >= self.vocab_size:
                results.append(b'\xff\xfd')
            else:
                results.append(self.vocab[token_id])
        return b''.join(results).decode('utf-8', errors='replace')


import numpy as np
import json
import os
from tqdm import tqdm

def read_iterable_txt_tqdm(filepath: str, encoding: str = "utf-8") -> Iterator[str]:
    total_bytes = os.path.getsize(filepath)
    # unit_scale=True 会把 bytes 自动显示成 KB/MB/GB
    with open(filepath, "r", encoding=encoding, newline="") as f, tqdm(
        total=total_bytes, unit="B", unit_scale=True, unit_divisor=1024, desc="Reading+Tokenizing"
    ) as pbar:
        for line in f:
            # 以“该行在磁盘上的字节数”更新进度
            # newline="" 能更稳定地反映换行符处理；这里把 '\n' 补回去计入字节更接近真实文件大小
            b = (line + "\n").encode(encoding)
            pbar.update(len(b))
            yield line

def write_npy_shards(dirpath: str, token_iter: Iterator[int], chunk_tokens: int = 2_000_000_000):
    os.makedirs(dirpath, exist_ok=True)

    buf = np.empty(chunk_tokens, dtype=np.uint16)
    filled = 0
    part = 0
    total_tokens = 0

    for t in token_iter:
        buf[filled] = t
        filled += 1
        total_tokens += 1

        if filled == chunk_tokens:
            np.save(os.path.join(dirpath, f"part_{part:05d}.npy"), buf)
            part += 1
            filled = 0

    if filled:
        np.save(os.path.join(dirpath, f"part_{part:05d}.npy"), buf[:filled])
        part += 1

    with open(os.path.join(dirpath, "index.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"dtype": "uint16", "chunk_tokens": chunk_tokens, "parts": part, "total_tokens": total_tokens},
            f,
            ensure_ascii=False,
            indent=2,
        )

if __name__ == "__main__":
    # def read_passages(filepath: str, n: int = 10) -> Iterator[str]:
    #     special_token = "<|endoftext|>"
    #     with open(filepath, "r", encoding="utf-8") as f:
    #         results = []
    #         tmp_res = ""
    #         for line in f:
    #             if special_token in line:
    #                 parts = line.split(special_token)
    #                 for part in parts[:-1]:
    #                     tmp_res += part
    #                     results.append(tmp_res)
    #                     tmp_res = ""
    #                 tmp_res += parts[-1]
    #             else:
    #                 tmp_res += line
    #             if len(results) >= n:
    #                 return results
    #         if tmp_res:
    #             results.append(tmp_res)
    #     return results
                

    # tokenizer = BPETokenizer.from_files("run_res/bpe_TinyStoriesV2-GPT4-train.txt.pkl", special_tokens=["<|endoftext|>"])
    # byte_lens = 0
    # token_lens = 0
    # for passage in read_passages("data/TinyStoriesV2-GPT4-train.txt", n=10):
    #     byte_lens += len(passage.encode("utf-8"))
    #     token_ids = tokenizer.encode(passage)
    #     token_lens += len(token_ids)
    #     reconstructed_passage = tokenizer.decode(token_ids)
    #     assert passage == reconstructed_passage, "Reconstructed passage does not match the original."
    #     print(f"Original Passage: {passage[:50]}... | Tokens: {token_ids} | Reconstructed: {reconstructed_passage[:50]}...")
    
    # print(f"Average compression ratio (bytes/tokens): {byte_lens / token_lens:.2f}")
    
    # byte_lens = 0
    # token_lens = 0
    # for passage in read_passages("data/TinyStoriesV2-GPT4-train.txt", n=1000):
    #     byte_lens += len(passage.encode("utf-8"))
    #     token_ids = list(tokenizer.encode_iterable([passage]))
    #     token_lens += len(token_ids)
    # print(f"Average compression ratio for 1000 passages (bytes/tokens): {byte_lens / token_lens:.2f}")
    
    # byte_lens = 0
    # token_lens = 0
    # for passage in read_passages("data/owt_valid.txt", n=1000):
    #     byte_lens += len(passage.encode("utf-8"))
    #     token_ids = list(tokenizer.encode_iterable([passage]))
    #     token_lens += len(token_ids)
    # print(f"Average compression ratio for OWT valid set (bytes/tokens): {byte_lens / token_lens:.2f}")
    

    tokenizer = BPETokenizer.from_files("run_res/bpe_TinyStoriesV2-GPT4-train.txt.pkl", special_tokens=["<|endoftext|>"])
    passages = read_iterable_txt_tqdm("data/TinyStoriesV2-GPT4-train.txt")
    write_npy_shards("run_res/tokenized_TinyStoriesV2-GPT4-train", tokenizer.encode_iterable(passages))
    
    passages = read_iterable_txt_tqdm("data/TinyStoriesV2-GPT4-valid.txt")
    write_npy_shards("run_res/tokenized_TinyStoriesV2-GPT4-valid", tokenizer.encode_iterable(passages))
    
    tokenizer = BPETokenizer.from_files("run_res/bpe_owt_train.txt.pkl", special_tokens=["<|endoftext|>"])
    passages = read_iterable_txt_tqdm("data/owt_train.txt")
    write_npy_shards("run_res/tokenized_owt_train", tokenizer.encode_iterable(passages))
    
    passages = read_iterable_txt_tqdm("data/owt_valid.txt")
    write_npy_shards("run_res/tokenized_owt_valid", tokenizer.encode_iterable(passages))