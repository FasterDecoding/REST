from draftretriever import Reader
from ngram_datastore.utils import *
from transformers import AutoTokenizer
from tqdm import tqdm

import os
import time
import pickle

class NGramDatastore:
    def __init__(self, dataset_name: str, num_conversations: int, model_path: str, reader: Reader, ngram_n: int, num_top_ngrams: int) -> None:
        self.dataset_name = dataset_name
        self.num_conversations = num_conversations
        self.reader = reader
        self.model_path = model_path
        self.ngram_n = ngram_n
        self.num_top_ngrams = num_top_ngrams
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.datastore_dpath = f"./datastore/ngram/{dataset_name}/"
        self.datastore_path = os.path.join(self.datastore_dpath, f"n{self.ngram_n}-convs{num_conversations}-top{num_top_ngrams}.pkl")
        os.makedirs(self.datastore_dpath, exist_ok=True)

    def build(self) -> None:
        self.datastore = dict()

        if self.dataset_name == "Aeala/ShareGPT_Vicuna_unfiltered":
            ngrams = get_ngrams_from_sharegpt(self.tokenizer, self.dataset_name, self.ngram_n, self.num_conversations, self.num_top_ngrams)
        elif self.dataset_name == "bigcode/the-stack":
            pass # TODO: make function to read ngrams from the stack dataset
        else:
            print("We only support Aeala/ShareGPT_Vicuna_unfiltered or bigcode/the-stack datasets for now")
            quit()

        for ngram in tqdm(ngrams):
            tree = self.reader.search(list(ngram))
            self.datastore[ngram] = tree

        with open(self.datastore_path, 'wb') as f:
            pickle.dump(self.datastore, f)
    
    def load_or_build(self) -> None:
        if os.path.exists(self.datastore_path):
            start_time = time.time()
            with open(self.datastore_path, 'rb') as f:
                self.datastore = pickle.load(f)
            duration = time.time() - start_time
            print(f"Took {duration}s to load {self.datastore_path}")
        else:
            start_time = time.time()
            self.build()
            duration = time.time() - start_time
            print(f"Took {duration}s to build {self.datastore_path}")