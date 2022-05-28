# BM25
BM25 is a text retrieval function that can find similar documents or rank search in a set of documents based on the query terms appearing in each document irrespective of their proximity to each other. It is an improved and more generalised version of TF-IDF algorithm in NLP.

## Installation
It can be installed using pip:
```zsh
pip install bagmodels
```

## Getting started

```py
import re
from bagmodels import BM25

# Load corpus
corpus = list({
    "Yo, this NLP model is amazing!",
    "I love this NLP model",
    "Blah Blah Blah whatever!"
})

# Clean manually if needed or pass custom tokenizer to BM25
corpus = [re.sub(r",|!", " ", doc).strip() for doc in corpus]

# Initialize model
model = BM25(corpus=corpus)

# Similarity
model.similarity(corpus[0], corpus[1])

```