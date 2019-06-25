from rnngen.processing.preprocessing import ProcessData
from rnngen.recurrentnetwork.trainer import Generator
from rnngen.word2vec.word2vec import word2vec_trainer
from rnngen import datasets


def pre_process(save_pickle_dir, use_text_dir):
    # Create processed data
    ProcessData(save_pickle_dir, use_text_dir)


def generate(text_dir, use_word2vec=True,
             embeddings_dir=datasets.SIMPSONS_EMBEDDINGS_300):
    Generator(text_dir, use_word2vec,
              embeddings_dir)


def word2vec(text_dir, save_dir, previously_trained_emb=''):
    word2vec_trainer(text_dir, save_dir, previously_trained_emb)
