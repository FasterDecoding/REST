import sys
from datetime import datetime
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import List, Set, Tuple
# from ngram_datastore.utils import get_ngrams_from_list

def get_ngrams_from_list(l: List[str], n: int) -> Set[Tuple[str]]:
    return set(tuple(l[i:i+n]) for i in range(len(l) - n + 1))

import argparse
parser = argparse.ArgumentParser()


parser.add_argument(
    "--model-path",
    type=str,
    default="lmsys/vicuna-7b-v1.5",
    help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
)
parser.add_argument(
    "--large-datastore",
    type=bool,
    default=False,
    help="Whether to use a large datastore",
)

args = parser.parse_args()
if args.large_datastore:
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    all_three_grams = set()
    all_four_grams = set()
    dataset = load_dataset('stingning/ultrachat', split='train')
    total_length = len(dataset)
    date_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    with open(f'ultrachat_three_four_data_{date_string}.csv', 'w') as f:
        f.write(f'i,three,four\n')
        for i, conversations in enumerate(tqdm(dataset, total=total_length)):
            for sample in conversations['data']:
                token_list = tokenizer.encode(sample['value'])
                three_grams = get_ngrams_from_list(token_list, 3)
                all_three_grams.update(three_grams)
                four_grams = get_ngrams_from_list(token_list, 4)
                all_four_grams.update(four_grams)
                # writer.add_entry(token_list)

            if i % 1000 == 0:
                f.write(f'{i},{len(all_three_grams)},{len(all_four_grams)}\n')
                f.flush()
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    all_three_grams = set()
    all_four_grams = set()
    dataset = load_dataset('Aeala/ShareGPT_Vicuna_unfiltered', split='train')
    total_length = len(dataset)
    date_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    with open(f'share_gpt_three_four_data_{date_string}.csv', 'w') as f:
        f.write(f'i,three,four\n')
        for i, conversations in enumerate(tqdm(dataset, total=total_length)):
            for sample in conversations['conversations']:
                token_list = tokenizer.encode(sample['value'])
                three_grams = get_ngrams_from_list(token_list, 3)
                all_three_grams.update(three_grams)
                four_grams = get_ngrams_from_list(token_list, 4)
                all_four_grams.update(four_grams)
                # writer.add_entry(token_list)

            if i % 1000 == 0:
                f.write(f'{i},{len(all_three_grams)},{len(all_four_grams)}\n')
                f.flush()