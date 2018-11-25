import numpy as np
from gensim.models import KeyedVectors as Kv


def word_embeddings(fname, b, d, seq_l):
    """
    Returns a feature array with for each sample the embeddings of the words in the sentence
    :param fname: string, file name of the pretrained embedding
    :param b: bool, is the file binary or not
    :param d: list of list of string, sentences to process
    :param seq_l: int, length of the sequence we want to return
    :return:
    """
    # Load the pretrained model
    model = Kv.load_word2vec_format(fname, binary=b)
    feat_l = len(model['hello'])

    x = []
    for sentence in d:
        # Create an embedding for each sentence
        sentence_embedding = []

        for word in sentence:
            try:
                sentence_embedding.append(model[word])
            except KeyError:
                pass                # Do nothing for OOV items

        # Check that it is not empty
        if sentence_embedding:

            # Remove elements if necessary
            if len(sentence_embedding) >= seq_l:
                x.append(sentence_embedding[0:seq_l])

            # Perform zero-padding if necessary
            else:
                for i in range(seq_l - len(sentence_embedding)):
                    sentence_embedding.append(np.zeros(feat_l))
                x.append(sentence_embedding)

    return np.asarray(x)


if __name__ == '__main__':

    embedding_fname = 'train_data/word_embeddings/GoogleNews-vectors-negative300.bin'
    binary = True
    data = [["I", "eat", "a", "cow"],
            ["the", "bull", "is", "dead"]]
    seq_len = 5

    X = word_embeddings(embedding_fname, binary, data, seq_len)
    print(X[0, 4, 0:5])
    print(X[0, 2, 0:5])

