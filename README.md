# BagModels
[![PyPI version](https://badge.fury.io/py/bagmodels.svg)](https://badge.fury.io/py/bagmodels)
[![Python tests](https://github.com/HimanshuMittal01/bagmodels/actions/workflows/python-app.yml/badge.svg)](https://github.com/HimanshuMittal01/bagmodels/actions/workflows/python-app.yml)

BagModels is a repository of various bag of words (BoW) algorithms in machine learning. Currently it includes OkapiBM25. More coming soon.

BM25 is a text retrieval function that can find similar documents or rank search in a set of documents based on the query terms appearing in each document irrespective of their proximity to each other. It is an improved and more generalised version of TF-IDF algorithm in NLP.

## Installation
It can be installed using pip:
```zsh
pip install bagmodels
```

## Getting started

### Basic usage
```py
import re
from bagmodels import BM25

# Load corpus
corpus = list({
    "Yo, I love NLP model",
    "I like algorithms",
    "I love ML!"
})

# Clean manually if needed or pass custom tokenizer to BM25
corpus = [re.sub(r",|!", " ", doc).strip() for doc in corpus]

# Initialize model
model = BM25(corpus=corpus)

# Similarity
model.similarity("I love NLP model", "I like NLP model") # 0.775
model.similarity("I love blah", "I love algorithms") # 0.446
```

### Save and reuse models
```py
# libaries imported and corpus already loaded before it
model = BM25(corpus=corpus)

# write to save path
model.save("output/bm25_v1.jbl")

# load again
model = BM25.load("output/bm25_v1.jbl")

# add documents if required
model.resume(corpus=additonal_corpus)

# predict / search / find / retrieve like
model.similarity(doc_a, doc_b)
```

## Coming soon
Please feel free to open an issue to request a feature or discuss any changes. Pull requests are most welcome.

I am trying to actively add the following:
- [X] OkapiBM25
- [ ] BM25 variations
- [ ] MultiThreading
