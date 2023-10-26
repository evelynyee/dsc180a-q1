"""
Add tf-idf labels to dataset.
TODO: SET UP MANUAL TF-IDF
"""
from sklearn.feature_extraction.text import TfidfVectorizer

from constants import *
tqdm.pandas()

def get_score(index, label, agg=np.mean):
    """
    Get the aggregated tf-idf score for a document (specified by its index) and a label pair.
    """
    scores = []
    for word in seeds[label]:
        if word not in tfidf.vocabulary_:
            scores.append(0)
            continue
        word_index = tfidf.vocabulary_[word]
        score = vecs[index,word_index]
        scores.append(score)
    return agg(scores)
def get_label(index):
    """
    Get the ideal label for a document (specified by its index), using tf-idf to the seed words.
    """
    scores = []
    for label in seeds:
        scores.append((get_score(index, label), label))
    return max(scores)[1]

# Set up command-line arguments parser
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument("--set", type=str, required=True,
                    help="Which dataset to use.")
parser.add_argument("--gran", type=str, default='coarse',
                    help="Which dataset granularity to use.")
parser.add_argument("--hyper", nargs='*',
                    help="Hyperparameters for the Word2Vec model. \
Arguments should all be specified like `argname=value`.\
See https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html\
for a list of possible hyperparameters.")

args = parser.parse_args()
set = eval(args.set)
gran = args.gran
hyper = {}
if args.hyper:
    for kwarg in args.hyper: # These are optional
        argname,value = kwarg.split("=") # parse the argument names and values from the command-line strings
        hyper[argname] = value
if hyper:
    style = format_hyperparams(hyper)
else:
    style = 'auto'
style = 'tfidf '+style

print('Loading dataset.')
df_path = get_data_path(set, granularity=gran)
df = get_data(set, granularity=gran)
seeds = get_data(set, granularity=gran, type='seedwords')

print('Getting TF-IDF encodings.')
tfidf = TfidfVectorizer(**hyper)
vecs = tfidf.fit_transform(tqdm(df['sentence']))

print("Identifying best labels.")
df[style] = df.index.to_series().progress_apply(get_label)
with open(df_path, 'wb') as f:
    pickle.dump(df, f)

macro_f1, micro_f1 = f1_scores(df,style)
results = f'{set}-{gran}: TF-IDF finished running at {time.ctime()}. Macro F1: {macro_f1}; Micro F1: {micro_f1}'
with open(RESULTS_FILE, 'a') as f:
    f.write(results)
    f.write('\n')
print(f'F1 scores saved at {RESULTS_FILE}')