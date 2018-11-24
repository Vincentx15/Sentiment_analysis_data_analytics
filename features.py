import numpy as np
from gensim.models import KeyedVectors as KV

def word_embeddings(embedding_fname, binary, data, time_steps):

    model = KV.load_word2vec_format(embedding_fname, binary=binary)
    len_features = len(model['hello'])

    x = []
    for sentence in data:
        sentence_embedding = []
        for word in sentence:
            try:
                sentence_embedding.append(model[word])
            except KeyError:
                pass

        if sentence_embedding:
            if len(sentence_embedding) >= time_steps:
                x.append(sentence_embedding[0:time_steps])
            else:
                for i in range(time_steps-len(sentence_embedding)):
                    sentence_embedding.append(np.zeros(len_features))
                x.append(sentence_embedding)

    return np.asarray(x)


if __name__ == '__main__':

    data = [["I", "eat", "a", "cow"],
            ["the", "bull", "is", "dead"]]
    embedding_fname = 'data/word_embeddings/GoogleNews-vectors-negative300.bin'
    binary = True
    time_steps = 5

    X = word_embeddings(embedding_fname, binary, data, time_steps)
    print(X[0,4,0:5])
    print(X[0, 2, 0:5])

