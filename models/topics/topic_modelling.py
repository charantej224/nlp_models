import pandas as pd
import os
import gensim
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import numpy as np
import nltk
from gensim import corpora, models

np.random.seed(2018)
nltk.download('wordnet')
stemmer = SnowballStemmer("english")

root_dir = "/home/charan/Documents/workspaces/python_workspaces/Data/Saria"
dataset_path = "LIWC2015 Results (mypersonality_2_pREC1.csv).csv"
data_with_topic = "FB_Topic.csv"
dataset_path = os.path.join(root_dir, dataset_path)
data_with_topic = os.path.join(root_dir, data_with_topic)

data = pd.read_csv(dataset_path, error_bad_lines=False)
data['index'] = data.index
data_text = data[['index', 'CLEAN_STATUS']]
documents = data_text
print(len(documents))
print(documents[:5])


def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


processed_docs = documents['CLEAN_STATUS'].map(preprocess)
dictionary = gensim.corpora.Dictionary(processed_docs)
count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break

dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]

lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=10, id2word=dictionary, passes=2, workers=4)
list_topics = []
for each in bow_corpus:
    topics_each = set()
    sorted_topics = sorted(lda_model_tfidf[each], key=lambda tup: -1 * tup[1])
    for index, score in sorted_topics[:5]:
        print("\nScore: {}\t \nTopic: {}".format(score, lda_model_tfidf.print_topic(index, 10)))
        topics_each.add(lda_model_tfidf.show_topic(index, topn=1)[0][0])
    list_topics.append(" ".join(list(topics_each)[::-1]))

data["TOPICS"] = pd.DataFrame(list_topics, columns=['TOPICS'])
data.drop(columns=['Unnamed: 0.1'], inplace=True)
data.to_csv(data_with_topic, header=True, index=False)
print("modelling finished")
