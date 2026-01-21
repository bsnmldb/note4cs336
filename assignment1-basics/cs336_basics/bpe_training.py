from .pretokenization import pretrain_bpe
from typing import Dict, List, Tuple
import pickle
from tqdm import tqdm
import time

class PretokenizationCache:
    def __init__(self, pretokenization_result: Dict[bytes, int]) -> None:
        self.cache: Dict[Tuple[int, ...], int] = {}
        self.pair_counts: Dict[Tuple[int, int], int] = {}
        
        for token, count in pretokenization_result.items():
            token_tuple = tuple(token)
            self.cache[token_tuple] = count
            for i in range(len(token_tuple) - 1):
                pair = (token_tuple[i], token_tuple[i + 1])
                self.pair_counts[pair] = self.pair_counts.get(pair, 0) + count
    
    def get_most_frequent_pairs(self) -> List[Tuple[int, int]]:
        if not self.pair_counts:
            return []
        max_count = max(self.pair_counts.values())
        most_frequent_pairs = [pair for pair, count in self.pair_counts.items() if count == max_count]
        return most_frequent_pairs
    
    def update_cache(self, pair_to_merge: Tuple[int, int], token: int) -> None:
        new_cache: Dict[Tuple[int, ...], int] = {}
        
        for existing_token, count in self.cache.items():
            # Skip tokens that don't contain the pair
            if pair_to_merge[0] not in existing_token:
                new_cache[existing_token] = count
                continue
            
            # Check if this token actually contains the pair to merge
            has_pair = False
            for i in range(len(existing_token) - 1):
                if (existing_token[i], existing_token[i + 1]) == pair_to_merge:
                    has_pair = True
                    break
            
            if not has_pair:
                new_cache[existing_token] = count
                continue
            
            # This token contains the pair, need to update pair counts
            # Remove old pairs from pair_counts
            for i in range(len(existing_token) - 1):
                pair = (existing_token[i], existing_token[i + 1])
                self.pair_counts[pair] -= count
                if self.pair_counts[pair] == 0:
                    del self.pair_counts[pair]
            
            # Merge the pair in this token
            new_token_parts: List[int] = []
            i = 0
            while i < len(existing_token):
                if i < len(existing_token) - 1 and (existing_token[i], existing_token[i + 1]) == pair_to_merge:
                    new_token_parts.append(token)
                    i += 2
                else:
                    new_token_parts.append(existing_token[i])
                    i += 1
            
            new_token_tuple = tuple(new_token_parts)
            
            # Add new pairs to pair_counts
            for i in range(len(new_token_tuple) - 1):
                pair = (new_token_tuple[i], new_token_tuple[i + 1])
                self.pair_counts[pair] = self.pair_counts.get(pair, 0) + count
            
            if new_token_tuple not in new_cache:
                new_cache[new_token_tuple] = count
            else:
                new_cache[new_token_tuple] += count
        
        self.cache = new_cache


def train_bpe(input_path: str, vocab_size: int, special_tokens: List[str], store_output: bool = False) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Train a Byte Pair Encoding (BPE) model on the given input text file.

    Args:
        input_path (str): Path to the input text file.
        vocab_size (int): Desired vocabulary size for the BPE model.
        special_tokens (list, optional): List of special tokens to include in the vocabulary.

    Returns:
        vocab (dict[int, bytes]): A dictionary mapping token IDs to byte sequences.
        merges (list[tuple[bytes, bytes]]): A list of byte pair merges used
    """
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []
    start_time = time.time()
    pretokenization_result = pretrain_bpe(input_path, f"run_res/{input_path.split('/')[-1]}.pkl" if 0 else None, special_tokens)
    end_time = time.time()
    print(f"Pretokenization completed in {end_time - start_time:.2f} seconds.")
    
    cache = PretokenizationCache(pretokenization_result)

    start_time = time.time()
    for _ in tqdm(range(vocab_size - len(special_tokens) - 256), desc="Training BPE"):
        most_frequent_pairs = cache.get_most_frequent_pairs()
        if not most_frequent_pairs:
            raise RuntimeError("No more pairs to merge.")
        pair_to_merge = max(most_frequent_pairs, key=lambda pair: (vocab[pair[0]], vocab[pair[1]]))
        token = len(vocab)
        vocab[token] = vocab[pair_to_merge[0]] + vocab[pair_to_merge[1]] 
        merges.append((vocab[pair_to_merge[0]], vocab[pair_to_merge[1]]))
        cache.update_cache(pair_to_merge, token)
    end_time = time.time()
    print(f"BPE training completed in {end_time - start_time:.2f} seconds.")
    
    for special_token in special_tokens:
        assert special_token.encode("utf-8") not in vocab.values(), f"Special token {special_token} already in vocabulary."
        vocab[len(vocab)] = special_token.encode("utf-8")
        
    if store_output:
        output_path = f"run_res/bpe_{input_path.split('/')[-1]}.pkl"
        with open(output_path, "wb") as f:
            pickle.dump((vocab, merges), f)

    return vocab, merges

def main():
    # train_bpe(
    #     input_path="data/TinyStoriesV2-GPT4-train.txt",
    #     vocab_size=10000,
    #     special_tokens=["<|endoftext|>"],
    #     store_output=True
    # )
    
    train_bpe(
        input_path="data/owt_train.txt",
        vocab_size=32000,
        special_tokens=["<|endoftext|>"],
        store_output=True
    )

if __name__ == "__main__":
    main()