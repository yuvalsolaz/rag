'''
 To load the default "wiki_dpr" dataset with 21M passages from wikipedia (index name is 'compressed' or 'exact')
'''

from transformers import RagRetriever

retriever_name_or_path = "facebook/dpr-ctx_encoder-single-nq-base"
dataset="wiki_dpr"
index_name="compressed"

def load_documents(check_pnt=chk_pnt, dataset=dataset, index_name=index_name):
    retriever = RagRetriever.from_pretrained(retriever_name_or_path=retriever_name_or_path,
                                             dataset=dataset,
                                             index_name=index_name)

    # To load your own indexed dataset built with the datasets library. More info on how to build the indexed dataset in examples/rag/use_own_knowledge_dataset.py
    from transformers import RagRetriever


# To load your own indexed dataset built with the datasets library that was saved on disk.
# More info in examples/rag/use_own_knowledge_dataset.py

from transformers import RagRetriever

dataset_path = "path/to/my/dataset"  # dataset saved via *dataset.save_to_disk(...)*
index_path = "path/to/my/index.faiss"  # faiss index saved via *dataset.get_index("embeddings").save(...)*
retriever = RagRetriever.from_pretrained(
    "facebook/dpr-ctx_encoder-single-nq-base",
    index_name="custom",
    passages_path=dataset_path,
    index_path=index_path,
)

# To load the legacy index built originally for Rag's paper
from transformers import RagRetriever

retriever = RagRetriever.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base", index_name="legacy")