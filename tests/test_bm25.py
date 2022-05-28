'''
 @author    : Himanshu Mittal
 @contact   : https://www.linkedin.com/in/himanshumittal13/
 @created on: 14-04-2022 15:30:49
'''


import unittest
from bagmodels import BM25, whitespace_tokenizer

class TestBM25(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Load corpus
        cls.tokenized_corpus = [
            ["i","am","himanshu","mittal","thriving","as","a","full","stack","data","scientist"],
            ["experience","scientist","going","mad"],
            ["life","is","a","collection","of","experiences"],
            ["experience","of","collection","of","fully","lived","moments"],
            [""],
            [],
            ["live","every","moment","to","its","fullest"],
            ["live","life","to","its","fullest"],
            ["choose","your","experience","of","life"]
        ]

        # Load BM25 model
        cls.bm25 = BM25(corpus=cls.tokenized_corpus, tokenizer=whitespace_tokenizer, k1=1.1, k2=1.0, b=0.8)
    
    def test_corpus_size(self):
        """Check corpus size, must ignores empty documents"""
        self.assertEqual(8, self.bm25.corpus_size)
    
    def test_average_document_length(self):
        """Check average document length"""
        self.assertAlmostEqual(5.625,self.bm25.avgdl,delta=0.01)
    
    def test_term_frequencies_per_document(self):
        """Check frequency of each token per document"""
        self.assertEqual(2, self.bm25.tfs[3]["of"])
        self.assertEqual(1, self.bm25.tfs[0]["scientist"])
    
    def test_document_frequencies(self):
        """Check in how many documents token appears"""
        self.assertEqual(2, self.bm25.dfs["scientist"])
        self.assertEqual(3, self.bm25.dfs["experience"])
        self.assertEqual(1, self.bm25.dfs["experiences"])
        self.assertEqual(3, self.bm25.dfs["of"])
    
    def test_inverse_document_frequencies(self):
        """Check calculation of inverse document frequency of tokens"""
        self.assertAlmostEqual(1.2809, self.bm25.idfs["scientist"], delta=0.01)
        self.assertAlmostEqual(0.9444, self.bm25.idfs["experience"], delta=0.01)
        self.assertAlmostEqual(1.7917, self.bm25.idfs["moments"], delta=0.01)
        self.assertAlmostEqual(0.9444, self.bm25.idfs["of"], delta=0.01)
    
    def test_str_similarity(self):
        """Check similarity on two untokenized strings/sentences"""
        self.assertAlmostEqual(0.4302, self.bm25.similarity("scientist experience", "moments of scientist"), delta=0.001)
        self.assertAlmostEqual(0.8924, self.bm25.similarity("fully thriving collection", "fully thriving packet"), delta=0.001)
