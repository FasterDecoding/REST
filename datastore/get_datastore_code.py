from datasets import load_dataset
from transformers import AutoTokenizer
import draftretriever
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser()

parser.add_argument(
    "--model-path",
    type=str,
    default="codellama/CodeLlama-7b-instruct-hf",
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
segment = 30 if args.large_datastore else 1 # Maximum number of segment: 144
data_files = []
for i in range(segment):
    if i>=100:
        data_files.append(f"data-00{i}-of-00144.parquet")
    elif i >=10:
        data_files.append(f"data-000{i}-of-00144.parquet")
    else:
        data_files.append(f"data-0000{i}-of-00144.parquet")
print("data_files:", data_files)

dataset = load_dataset('bigcode/the-stack-dedup', \
    data_dir='data/python', split='train', data_files=data_files)


datastore_path = './datastore_stack_large.idx' if args.large_datastore else './datastore_stack_small.idx'
writer = draftretriever.Writer(
    index_file_path=datastore_path,
    max_chunk_len=512 * 1024 * 1024,
    vocab_size=tokenizer.vocab_size + len(tokenizer.get_added_vocab()),
)

total_length = len(dataset)
print("number of samples: ", total_length)

for sample in tqdm(dataset, total=len(dataset)):
    token_list = tokenizer.encode(sample['content'])
    writer.add_entry(token_list)

writer.finalize()
