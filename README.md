# SemEval2017
**SemEval Task 3 - SubTask D (Arabic cQA) : Question Reranking Problem**

> Given a question and the first ten pairs of question/answer in its question thread, rerank 
the 10 pairs according to their relevance with respect to the original question.

----
* Preprocessing text chunks using IBM normalization, 
* Tokenization using Stanford CoreNLP.
* TextRank in order to shrink the chunks using co-occurences 
count as weight for the tree edges, taking the first 10 vertices
and filtering using the POS taggings of words.
* Computing cbow vectors using fasttext and webteb corpus
scraped from webteb.com after preprocessing.
* Computing weighted terms matrix factorization.
* Using lexical features such as cosine, damerau, dice...
* SVMrank for ranking using combinations of the 
aforementioned features.

**References**
