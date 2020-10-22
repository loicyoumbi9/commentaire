#corpusdir = 'data/cv_corpus'
#corpus = PlaintextCorpusReader(corpusdir,'.*',encoding='windows-1252')
#print("Preprocessing words....")
#sents = [[token.lemma_ for token in nlp(" ".join(self.clean(sent)).lower()) if token.lemma_ not in stopset] for sent in corpa.sents()]
#print("training word vectors....")
from gensim.models import Word2Vec
from gensim.test.utils import get_tmpfile

with open('b.txt','r') as f :
    lines = f.readlines()
    sents = [[token.lower() for token in line.split()] for line in lines] 
    print(sents)
    model = Word2Vec(sents,window=3, size=20,min_count=1, workers=4)
    fname = get_tmpfile("vectors.kv")
    model.wv.save(fname)
    l = model.most_similar(positive=['permission'], topn=5)
    print(l)
    print(model.wv['permission'])
    print("cv_to_matrix model saved")


