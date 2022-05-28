"""
 @author    : Himanshu Mittal
 @contact   : https://www.linkedin.com/in/himanshumittal13/
 @created on: 04-03-2022 15:06:33
"""

import re
from typing import List, Union, Callable

# Type-aliases
T_Documents = Union[List[str], str]
T_Tokenizer_Args = [T_Documents]
T_Tokenizer_Returns = Union[List[List[str]], List[str]]
T_Tokenizer = Callable[[*T_Tokenizer_Args], T_Tokenizer_Returns]

# Tokenizer functions
def tokenize_default(documents: T_Documents) -> T_Tokenizer_Returns:
    return_single_document = False
    if isinstance(documents, str):
        documents = [documents]
        return_single_document = True

    tokenized_documents = [None]*len(documents)
    for i, document in enumerate(documents):
        tokenized_documents[i] = (
            re.sub("\s+", " ", document.strip().lower()).split(' ')
        )

    return (
        tokenized_documents
        if not return_single_document
        else tokenized_documents[0]
    )
