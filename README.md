# DSC 180A Quarter 1 Project
## Goal: **Replication of TF-IDF and Word2Vec baselines from the [ConWea paper](https://aclanthology.org/2020.acl-main.30.pdf])**
Evelyn Yee

### Description
For replication, we will aim to reproduce some baseline methods used in ConWea: (1) **IR-TF-IDF** and (2) **Word2Vec**. We are skipping the neural network models since DSMLP doesn’t have enough GPU memory to support BERT model training.

Quoting from the ConWea paper’s Section 6.2, we have

_IR-TF-IDF treats the seed word set for each class as a query. The relevance of a document to a label is computed by aggregated TF-IDF values of its respective seed words. The label with the highest relevance is assigned to each document.
Word2Vec first learns word vector representations (Mikolov et al., 2013) for all terms in the corpus and derive label representations by aggregating the vectors of its respective seed words. Finally, each document is labeled with the most similar label based on cosine similarity._

### Dependencies
- python (3.11.6)
- scikit-learn (1.3.2)
- gensim (4.3.2)
- numpy (1.26.1)
- tqdm (4.66.1)
- pandas (2.1.1)

### Setup
Install the specified dependencies. Clone [the ConWea repo](https://github.com/dheeraj7596/ConWea). If it is not in a sister directory to this one, specify the path in the `CONWEA_DATA_PATH` variable in `src/constants.py`. If you want to re-organize the directory, make sure all the contant filepaths at the top of `src/constants.py` match your desired filepaths for reading/writing. (Otherwise, you can run everything as-is.)

### Replication
- **TF-IDF Scores:** from within the `src` folder, run `python tf-idf.py --set <DATASET> --gran <GRANULARITY>`, where <DATASET> is either "TWENTY_NEWS" or "NYT" (required), and <GRANULARITY> is either "coarse" or "fine" (coarse by default).
  - Optional argument --hyper can be used to specify the hyperparameters for the [sklearn TfIDfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html). By default, the system will run with the same hyperparameters I used for the results in my report (default hyperparameters from the library).
- **Word2Vec Scores:** from within the `src` folder, run `python word2vec.py --set <DATASET> --gran <GRANULARITY>`, where <DATASET> is either "TWENTY_NEWS" or "NYT" (required), and <GRANULARITY> is either "coarse" or "fine" (coarse by default).
  - Optional argument --token can be used to specify the tokenizing function to be used ([gensim.utils.tokenize](https://tedboy.github.io/nlps/generated/generated/gensim.utils.tokenize.html) by default). If you want to use a different tokenizing function, you must import it [here](https://github.com/evelynyee/dsc180a-q1/blob/2f89a2a48cad1435171c002e130316f6cf6af5e8/word2vec.py#L6) (or define it yourself below), and you may also need to set up the tokenizer object [here](https://github.com/evelynyee/dsc180a-q1/blob/2f89a2a48cad1435171c002e130316f6cf6af5e8/word2vec.py#L119).
  - Optional argument --hyper can be used to specify the hyperparameters for the [gensim word2vec]([https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html](https://radimrehurek.com/gensim/models/word2vec.html)). By default, the system will run with the same hyperparameters I used for the results in my report (default hyperparameters from the library, except context window size of 20 and training over 40 epochs).
  - Trained Word2Vec embedding vectors for each dataset will be stored in `data/models` by default.
- Results (F1 scores) for both methods are recorded in `src/f1_scores.txt` by default (this can also be changed in `constants.py`).
