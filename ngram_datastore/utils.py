from typing import List, Set, Union
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from datasets import load_dataset

def get_ngrams_from_dataset(tokenizer: Union[PreTrainedTokenizer | PreTrainedTokenizerFast], dataset_name: str, n: int) -> Set[List[str]]:
    all_ngrams = set()
    dataset = load_dataset(dataset_name, split='train')
    
    for conversations in dataset:
        for sample in conversations['conversations']:
            tokens = tokenizer.encode(sample['value'])
            sample_ngrams = get_ngrams_from_list(tokens, n)
            all_ngrams.update(sample_ngrams)

    return all_ngrams

def get_ngrams_from_list(l: List[str], n: int) -> Set[List[str]]:
    return set(list(l[i:i+n]) for i in range(len(l) - n + 1))