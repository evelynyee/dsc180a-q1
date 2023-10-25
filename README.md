# DSC 180A Quarter 1 Project
## Goal: **Replication of TF-IDF and Word2Vec baselines from ConWea paper**
Evelyn Yee
### Description
For replication, we will aim to reproduce some baseline methods used in ConWea: (1) **IR-TF-IDF** and (2) **Word2Vec**. We are skipping the neural network models since DSMLP doesn’t have enough GPU memory to support BERT model training.

Quoting from the ConWea paper’s Section 6.2, we have

_IR-TF-IDF treats the seed word set for each class as a query. The relevance of a document to a label is computed by aggregated TF-IDF values of its respective seed words. The label with the highest relevance is assigned to each document.
Word2Vec first learns word vector representations (Mikolov et al., 2013) for all terms in the corpus and derive label representations by aggregating the vectors of its respective seed words. Finally, each document is labeled with the most similar label based on cosine similarity._

We will explain these two methods in the discussion session. If you have any questions, please get prepared to raise them during the discussion. 

### Advice
Here are some tips:

- In IR-TF-IDF, it is basically counting. The Python library TfidfVectorizer might be useuful. The IDF calculation is also straightforward after tokenization, so please feel free to implement it yourself (e.g., using for-loop and dictionary) in Python.
- In Word2Vec, you are asked to train word2vec embedding based on the input documents in each dataset. You may find gesim package useful.
For the seed words, please refer to the ConWea repo.


## Expectations
At the minimum, you are expected to show:

- A table of your replicated results, including the two methods mentioned above and the micro-F1 and macro-F1 scores. Specifically, we will focus on the two coarse-grained datasets in ConWea: (1) NYT-coarse and (2) 20News-coarse. Please feel free to include the fine-grained datasets.
- The implementation details of your replication. It will be best to include some concrete steps, formulas, and package information.
- The well-documented code repo of your replication.
- Any takeaways, challenges encountered, and thoughts are encouraged to be included in the report.

**Note:** If you found the performance is worse than the paper’s reported values, please pay attention to pre-processing and other tuning effort — think about tokenization, stopwords, stemming, lemmatization, different TF-IDF variants, hyper-parameters in word2vec, etc. Please keep in mind that In the paper, we tried our best to tune the baselines to make them strong. Therefore, in this replication practice, we don’t have to exactly re-produce the performance. As long as your implementation is on the right track, it is okay to be slightly worse than the reported numbers.
