from rnngen.processing.preprocessing import ProcessData
from rnngen.recurrentnetwork.trainer import Generator


def pre_process(save_pickle_dir, use_text_dir):
    # Create processed data
    ProcessData(save_pickle_dir, use_text_dir)


def generate(text_dir, word2vec_setting='new',
             emb_dir='embeddings.npy', use_word2vec=True):
    Generator(text_dir, word2vec_setting,
              emb_dir, use_word2vec)
