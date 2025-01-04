# Settings

## Add support for SmallThinker-3B & other reasoning models

Open webui doesn't send values by default in the payload even if you set them in the User Settings, so the max_tokens parameter is going to be misssing. You must configure it per model in the Admin Settings > Models page.

8k is whats SmallThinker was trained on, but this will just use the number set in the UI if present.

Additionally, remember to set the Stopping string or else the model will rant until the 8k mark.

For this particular model, its '### Final Solution'.

## Testing

### Defaults

Embedding Model: sentence-transformers/all-MiniLM-L6-v2
Reranking Model: BAAI/bge-reranker-v2-m3

### Embedding Models

BAAI/bge-large-en-v1.5 : Slow
BAAI/bge-small-en-v1.5 : Good?
jinaai/jina-embeddings-v3 : ?

intfloat/multilingual-e5-large-instruct : 500M
intfloat/e5-large-v2 : 300M
sentence-transformers/all-mpnet-base-v2 : 100M

google's text-embeddings-004 : Probably the best, not local tho.

### Reranking Models

dunzhang/stella_en_1.5B_v5 : Slow but good
dunzhang/stella_en_400M_v5 : Best, throws "Cross encoder error"?
