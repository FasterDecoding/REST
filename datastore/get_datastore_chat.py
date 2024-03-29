from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer
import draftretriever
from tqdm import tqdm
import json

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
print(args)

tokenizer = AutoTokenizer.from_pretrained(args.model_path)


def get_n_grams(list, n):
    return set(tuple(list[i:i+n]) for i in range(len(list) - n + 1))

datastore_path = './datastore_chat_large.idx' if args.large_datastore else './datastore_chat_small.idx'
writer = draftretriever.Writer(
    index_file_path=datastore_path,
    max_chunk_len=512*1024*1024,
    vocab_size=tokenizer.vocab_size,
)
if args.large_datastore:
    dataset = load_dataset('stingning/ultrachat', split='train')
    total_length = len(dataset)
    print("number of samples: ", total_length)
    for conversations in tqdm(dataset, total=total_length):
        for sample in conversations['data']:
            token_list = tokenizer.encode(sample)
            writer.add_entry(token_list)
else:
    all_three_grams = set()
    all_four_grams = set()
    dataset = load_dataset('Aeala/ShareGPT_Vicuna_unfiltered', split='train')
    total_length = len(dataset)
    date_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    with open(f'three_four_data_{date_string}.csv', 'w') as f:
        f.write(f'i,three,four\n')
        for i, conversations in enumerate(tqdm(dataset, total=total_length)):
            for sample in conversations['conversations']:
                token_list = tokenizer.encode(sample['value'])
                three_grams = get_n_grams(token_list, 3)
                all_three_grams = all_three_grams.union(three_grams)
                four_grams = get_n_grams(token_list, 4)
                all_four_grams = all_four_grams.union(four_grams)
                # writer.add_entry(token_list)

            if i % 100 == 0:
                f.write(f'{i},{len(all_three_grams)},{len(all_four_grams)}\n')

# writer.finalize()

