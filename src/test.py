from gensim.models import word2vec

model = word2vec.Word2Vec([[]], size=128, window=5, min_count=5, workers=4)
