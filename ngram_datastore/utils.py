from typing import List, Set, Tuple
from transformers import AutoTokenizer, PreTrainedTokenizer
from datasets import load_dataset
from tqdm import tqdm

def get_ngrams_from_ShareGPT(tokenizer: PreTrainedTokenizer, dataset_name: str, n: int) -> Set[Tuple[str]]:
    all_ngrams = set()
    dataset = load_dataset(dataset_name, split='train')
    
    for i, conversations in enumerate(tqdm(dataset)):
        if i > 10:
            break
        for sample in conversations['conversations']:
            tokens = tokenizer.encode(sample['value'])
            sample_ngrams = get_ngrams_from_list(tokens, n)
            all_ngrams.update(sample_ngrams)

    return all_ngrams

def get_ngrams_from_list(l: List[str], n: int) -> Set[Tuple[str]]:
    return set(tuple(l[i:i+n]) for i in range(len(l) - n + 1))