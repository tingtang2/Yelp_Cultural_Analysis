import lda

model = lda.LDA(n_topics=4)
model.fit(lda.X, lda.cc, lda.ps, lda.ls)

print(model.topic_word_)
