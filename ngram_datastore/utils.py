from typing import List, Set, Tuple
from transformers import AutoTokenizer, PreTrainedTokenizer
from datasets import load_dataset
from tqdm import tqdm
from itertools import islice

def get_ngrams_from_ShareGPT(tokenizer: PreTrainedTokenizer, dataset_name: str, ngram_n: int, num_conversations: int) -> Set[Tuple[str]]:
    all_ngrams = set()
    dataset = load_dataset(dataset_name, split='train')
    
    for conversations in tqdm(islice(dataset, num_conversations)):
        for sample in conversations['conversations']:
            tokens = tokenizer.encode(sample['value'])
            sample_ngrams = get_ngrams_from_list(tokens, ngram_n)
            all_ngrams.update(sample_ngrams)

    return all_ngrams

def get_ngrams_from_list(l: List[str], ngram_n: int) -> Set[Tuple[str]]:
    return set(tuple(l[i:i+ngram_n]) for i in range(len(l) - ngram_n + 1))