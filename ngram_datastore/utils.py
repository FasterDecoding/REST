from typing import List, Set, Tuple
from transformers import PreTrainedTokenizer
from datasets import load_dataset
from tqdm import tqdm
from itertools import islice
from collections import defaultdict

def get_ngrams_from_sharegpt(tokenizer: PreTrainedTokenizer, dataset_name: str, ngram_n: int, num_conversations: int, num_top_ngrams: int) -> Set[Tuple[str]]:
    all_ngram_counts = defaultdict(int)
    dataset = load_dataset(dataset_name, split='train')
    
    for conversations in tqdm(islice(dataset, num_conversations)):
        for sample in conversations['conversations']:
            tokens = tokenizer.encode(sample['value'])
            sample_ngrams = get_ngrams_from_list(tokens, ngram_n)
            
            for sample_ngram in sample_ngrams:
                all_ngram_counts[sample_ngram] += 1

    sorted_ngram_counts = sorted(all_ngram_counts.items(), key=(lambda item : (-item[1], item[0])))
    top_ngrams = [ngram for ngram, _ in sorted_ngram_counts][:num_top_ngrams]
    return top_ngrams

def get_ngrams_from_list(l: List[str], ngram_n: int) -> Set[Tuple[str]]:
    return set(tuple(l[i:i+ngram_n]) for i in range(len(l) - ngram_n + 1))