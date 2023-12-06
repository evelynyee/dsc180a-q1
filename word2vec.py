"""
Add word2vec labels to dataset.
"""
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.callbacks import CallbackAny2Vec
from gensim.utils import tokenize
from nltk import word_tokenize
import re
from scipy.spatial.distance import cosine

from constants import *
tqdm.pandas()

def clean_string(s):
    """
    Clean document strings.
    """
    # convert to lowercase
    s = s.lower()

    # Remove non-alphanumeric characters
    s = re.sub(r'\W+', ' ', s)

    # Replace all whitespace with a single space
    s = re.sub(r'\s+', ' ', s)
    return s

class TQDMCallback(CallbackAny2Vec):
    """
    Callback to visualize Word2Vec training progress using tqdm.
    """
    def __init__(self, epochs):
        self.progress_bar = tqdm(total=epochs, desc="Epochs", position=0)
    def on_epoch_end(self, model):
        self.progress_bar.update(1)
    def on_train_end(self, model):
        self.progress_bar.close()

def get_embeddings (set,gran='coarse', tokenizer=tokenize, hyperparams={}):
    """
    Get the specified word2vec embeddings.
    If the embeddings have not been previously recorded, train a word2vec model to get embeddings.
    """
    kwargs = {}
    if tokenizer.__name__ != tokenize.__name__:
        kwargs['token'] = tokenizer.__name__
    kwargs.update(hyperparams)
    style = format_hyperparams(**kwargs)
    save_path = os.path.join(MODELS_PATH,format_hyperparams(prefix=set+'-'+gran, suffix='.txt', **kwargs))
    if not os.path.isfile(save_path): # no saved embeddings, calculate new.
        print("No embeddings found for specified dataset, hyperparameters.")
        df = get_data(set, granularity=gran)
        print("Tokenizing dataset.")
        sents = df['sentence'].progress_apply(tokenizer)
        print("Converting tokenized data to tuples.")
        sents = sents.progress_apply(tuple)
        print("Generating Word2Vec embeddings.")
        if 'epochs' in hyperparams:
            callback = TQDMCallback(epochs=hyperparams['epochs'])
        else:
            callback = TQDMCallback(epochs=5) # default is 5 epochs
        w2v = Word2Vec(sents, **hyperparams, callbacks=[callback])
        w2v.wv.save_word2vec_format(save_path)
        print(f'Saving Word2Vec embeddings to {save_path}')
    print(f'Retrieving Word2Vec embeddings from {save_path}')
    return KeyedVectors.load_word2vec_format(save_path), style

def aggregate_embeddings (doc:[str,list],tokenizer,embeds, agg=lambda x:np.mean(x,axis=0)):
    """
    Aggregate embedding vectors for words in a string (document) or a list (seed words).
    """
    if isinstance(doc, str):
        words = tokenizer(doc)
    else:
        words = doc
    word_embeds = []
    for word in words:
        if word in embeds:
            word_embeds.append(embeds[word])
    return agg(word_embeds)

def get_label (doc_embed, label_embeds, similarity):
    """
    Get the ideal label for a document (specified by its embedding), using similarity to the seedword embeddings.
    """
    scores = []
    for label, embed in label_embeds.items():
        scores.append((similarity(doc_embed, embed), label))
    return max(scores)[1]

def cosine_sim (u,v):
    """Cosine similarity between two vectors"""
    return 1-cosine(u,v)

# Set up command-line arguments parser
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument("--set", type=str, required=True,
                    help="Which dataset to use.")
parser.add_argument("--gran", type=str, default='coarse',
                    help="Which dataset granularity to use.")
parser.add_argument("--clean", type=bool, default=False,
                    help="Whether to apply the string-cleaning function to documents before prediction.")
parser.add_argument("--token", type=str, default='tokenize',
                    help="Tokenizer to use. Default is 'tokenize' from gensim.")
parser.add_argument("--sim", type=str, default='cosine_sim',
                    help="Which similarity function to use. Default is cosine similarity.")
parser.add_argument("--hyper", nargs='*',default=['epochs=40','window=20', 'workers=4'],
                    help="Hyperparameters for the Word2Vec model. \
Arguments should all be specified like `argname=value`.\
See https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec\
for a list of possible hyperparameters.")

# parse arguments
args = parser.parse_args()
set = eval(args.set)
gran = args.gran
tokenizer = eval(args.token)
sim = eval(args.sim)
clean = args.clean
hyper = {}
if args.hyper:
    for kwarg in args.hyper: # These are optional
        argname,value = kwarg.split("=") # parse the argument names and values from the command-line strings
        try:
            # Try to convert to integer
            value = int(value)
        except ValueError:
            try:
                # Try to convert to float
                value = float(value)
            except ValueError:
                # If both conversions fail, keep as string
                pass
        hyper[argname] = value

print('Loading dataset.')
df_path = get_data_path(set, granularity=gran)
df = get_data(set, granularity=gran)
seeds = get_data(set, granularity=gran, type='seedwords')

if clean and 'sentence_clean' in df: # cleaned version of sentences already stored
    docs = df['sentence_clean']
elif clean: # need to clean sentences
    print('Cleaning strings')
    df['sentence_clean'] = df['sentence'].progress_apply(clean_string)
    docs = df['sentence_clean']
else: # use original input sentences
    docs = df['sentence']

print('Getting word embeddings')
embeds,style = get_embeddings(set,gran=gran,tokenizer=tokenizer,hyperparams=hyper)
if not style:
    style = 'auto'
style = 'w2v '+style
print(embeds)

print('Aggregating class embeddings')
class_embeds = {label:aggregate_embeddings(seedwords,tokenizer,embeds) for label,seedwords in tqdm(seeds.items())}

print("Aggregating document embeddings")
doc_embeddings = df['sentence'].progress_apply(lambda doc: aggregate_embeddings(doc,tokenizer,embeds))

print("Identifying best labels")
df[style] = doc_embeddings.progress_apply(lambda doc: get_label(doc,class_embeds, sim))
with open(df_path, 'wb') as f:
    pickle.dump(df, f)

macro_f1, micro_f1 = f1_scores(df,style)
results = f'{set}-{gran}({"cleaned" if clean else "original"}) (hyperparameters: {hyper if hyper else "default"}): Word2Vec finished running at {time.ctime()}. Macro F1: {macro_f1}; Micro F1: {micro_f1}'
print(results)
with open(RESULTS_FILE, 'a') as f:
    f.write(results)
    f.write('\n')
print(f'F1 scores saved at {RESULTS_FILE}')

