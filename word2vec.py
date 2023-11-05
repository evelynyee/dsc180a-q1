"""
Add word2vec labels to dataset.
TODO: SET UP MANUAL WORD2VEC
"""
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.callbacks import CallbackAny2Vec
from gensim.utils import tokenize

from constants import *
tqdm.pandas()

class TQDMCallback(CallbackAny2Vec):
    """Callback to visualize Word2Vec training progress using tqdm."""
    def __init__(self, epochs):
        self.progress_bar = tqdm(total=epochs, desc="Epochs", position=0)
    def on_epoch_end(self, model):
        self.progress_bar.update(1)
    def on_train_end(self, model):
        self.progress_bar.close()

def get_embeddings (set,gran='coarse', tokenizer=tokenize, hyperparams={}):
    kwargs = {}
    if tokenizer.__name__ != tokenize.__name__:
        kwargs['token'] = tokenizer.__name__()
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

def get_label ():
    assert False, "get_label function not implemented yet."
    # TODO

# Set up command-line arguments parser
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument("--set", type=str, required=True,
                    help="Which dataset to use.")
parser.add_argument("--gran", type=str, default='coarse',
                    help="Which dataset granularity to use.")
parser.add_argument("--token", type=str, default='tokenize',
                    help="Tokenizer to use. Default is 'tokenize' from gensim.")
parser.add_argument("--hyper", nargs='*',
                    help="Hyperparameters for the Word2Vec model. \
Arguments should all be specified like `argname=value`.\
See https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec\
for a list of possible hyperparameters.")

# parse arguments
args = parser.parse_args()
set = eval(args.set)
gran = args.gran
tokenizer = eval(args.token)
hyper = {}
if args.hyper:
    for kwarg in args.hyper: # These are optional
        argname,value = kwarg.split("=") # parse the argument names and values from the command-line strings
        hyper[argname] = value


print('Loading dataset.')
df_path = get_data_path(set, granularity=gran)
df = get_data(set, granularity=gran)
seeds = get_data(set, granularity=gran, type='seedwords')

embeds,style = get_embeddings(set,gran=gran,tokenizer=tokenizer,hyperparams=hyper)
if not style:
    style = 'auto'
style = 'w2v '+style

print('Getting class embeddings')
# TODO

print("Identifying best labels")
df[style] = df.index.to_series().progress_apply(get_label)
with open(df_path, 'wb') as f:
    pickle.dump(df, f)

macro_f1, micro_f1 = f1_scores(df,style)
results = f'{set}-{gran}: TF-IDF finished running at {time.ctime()}. Macro F1: {macro_f1}; Micro F1: {micro_f1}'
with open(RESULTS_FILE, 'a') as f:
    f.write(results)
    f.write('\n')
print(f'F1 scores saved at {RESULTS_FILE}')

