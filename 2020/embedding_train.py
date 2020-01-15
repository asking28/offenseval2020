from gensim.test.utils import datapath
from gensim import utils

class MyCorpus(object):
    """An interator that yields sentences (lists of str)."""

    def __iter__(self):
        #corpus_path = datapath('lee_background.cor')
        for line in df_dummy['text']:
            # assume there's one document per line, tokens separated by whitespace
            yield utils.simple_preprocess(line)
import gensim.models

sentences = MyCorpus()
model = gensim.models.Word2Vec(sentences=sentences)
import tempfile

with tempfile.NamedTemporaryFile(prefix='/content/drive/My Drive/offenseval/2020/gensim-model-', delete=False) as tmp:
    temporary_filepath = tmp.name
    model.save(temporary_filepath)
    #
    # The model is now safely stored in the filepath.
    # You can copy it to other machines, share it with others, etc.
    #
    # To load a saved model
print(model.most_similar(positive=['hell'], topn=5))
model.wv.save_word2vec_format('/content/drive/My Drive/offenseval/2020/twitter_trained_emb.bin', binary=True)
model = gensim.models.KeyedVectors.load_word2vec_format('/content/drive/My Drive/offenseval/2020/twitter_trained_emb.bin', binary=True)