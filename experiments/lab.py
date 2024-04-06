from ngram_datastore.ngram_datastore import NGramDatastore
import draftretriever

model_path = 'lmsys/vicuna-7b-v1.5'
dataset_name = 'Aeala/ShareGPT_Vicuna_unfiltered'
datastore_path = './datastore/datastore_chat_small.idx'
reader = draftretriever.Reader(
    index_file_path=datastore_path,
)
ngram_datastore = NGramDatastore(dataset_name, 10, model_path, reader)
ngram_datastore.load_or_build()
# print(ngram_datastore.datastore)
