import my_word2vec

model = my_word2vec.Word2Vec(size=128, window=5, min_count=1, workers=4, keep_raw_vocab=True)
model.build_vocab([["c","c","c","a"],["a"],["a"]], n_build=0)
print("raw_vocab:\t",model.raw_vocab)
print("wv.vocab:\t",model.wv.vocab)
model.train([["c","c","c","a"],["a"],["a"]],total_examples=model.corpus_count, epochs=model.iter)

model.build_vocab([["b","b"]], update=True, n_build=1)
print("raw_vocab:\t",model.raw_vocab)
print("wv.vocab:\t",model.wv.vocab)
