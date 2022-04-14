"""
 @author    : Himanshu Mittal
 @contact   : https://www.linkedin.com/in/himanshumittal13/
 @created on: 04-03-2022 15:06:33
"""

from typing import List, Union, Callable

# Type-aliases
T_Documents = Union[List[str], str]
T_Tokenizer_Args = [T_Documents]
T_Tokenizer_Returns = Union[List[List[str]], List[str]]
T_Tokenizer = Callable[[*T_Tokenizer_Args], T_Tokenizer_Returns]

# Tokenizer functions
def tokenize_default(corpus: T_Documents) -> T_Tokenizer_Returns:
    import re

    # Initiate
    tokenized_corpus = [None]*len(corpus)

    # Cleaner
    for doc_id, s in enumerate(corpus):
        tokenized_corpus[doc_id] = re.sub(r'[\W]+', ' ', s)

    # whitespace tokenization
    for doc_id, s in enumerate(tokenized_corpus):
        tokenized_corpus[doc_id] = s.strip().split(' ')
        tokenized_corpus[doc_id] = list(map(lambda x:x.lower(), corpus[doc_id]))
    
    return tokenized_corpus
