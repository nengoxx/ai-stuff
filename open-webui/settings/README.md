# Settings

## Defaults

Embedding Model: sentence-transformers/all-MiniLM-L6-v2
Reranking Model: BAAI/bge-reranker-v2-m3

## Testing

### Embedding Models

BAAI/bge-large-en-v1.5 : Slow
BAAI/bge-small-en-v1.5 : Good?
jinaai/jina-embeddings-v3 : ?

intfloat/multilingual-e5-large-instruct : 500M
intfloat/e5-large-v2 : 300M
sentence-transformers/all-mpnet-base-v2 : 100M

### Reranking Models

dunzhang/stella_en_1.5B_v5 : Slow but good
dunzhang/stella_en_400M_v5 : Best, throws "Cross encoder error"?
