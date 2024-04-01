from ngram_datastore.ngram_datastore import NGramDatastore
from draftretriever import Writer, Reader

dataset_name = 'Aeala/ShareGPT_Vicuna_unfiltered'
datastore_path = './datastore_chat_small.idx'
writer = draftretriever.Writer(
    index_file_path=datastore_path,
    max_chunk_len=512*1024*1024,
    vocab_size=tokenizer.vocab_size,
)
ngram_datastore = NGramDatastore(dataset_name, reader)