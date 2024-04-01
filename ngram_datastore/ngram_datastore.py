from draftretriever import Reader
from ngram_datastore.utils import get_ngrams_from_dataset

class NGramDatastore:
    DEFAULT_NGRAM_N = 3

    def __init__(self, dataset_name: str, reader: Reader) -> None:
        self.dataset_name = dataset_name
        self.reader = reader

    def build(self) -> None:
        self.ngram_datastore = dict()
        ngrams = get_ngrams_from_dataset(self.dataset_name, NGramDatastore.DEFAULT_NGRAM_N)

        for ngram in ngrams:
            tree = self.reader.search(ngram)
            self.ngram_datastore[ngram] = tree