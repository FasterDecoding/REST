from ngram_datastore.ngram_datastore import NGramDatastore
import draftretriever

model_path = 'lmsys/vicuna-7b-v1.5'
dataset_name = 'Aeala/ShareGPT_Vicuna_unfiltered'
datastore_path = './datastore/datastore_chat_small.idx'
reader = draftretriever.Reader(
    index_file_path=datastore_path,
)
ngram_datastore = NGramDatastore(dataset_name, model_path, reader)
ngram_datastore.load_or_build()
# ngram_datastore2 = NGramDatastore(dataset_name, model_path, reader)
# ngram_datastore2.build()
# print(type(ngram_datastore.datastore), type(ngram_datastore2.datastore))
# print(len(ngram_datastore.datastore), len(ngram_datastore2.datastore))
# print(ngram_datastore.datastore[(450, 817, 363)], "|||\n", ngram_datastore2.datastore[(450, 817, 363)])
# print(ngram_datastore.datastore[(450, 817, 363)] == ngram_datastore2.datastore[(450, 817, 363)])
# print(ngram_datastore.datastore == ngram_datastore2.datastore)