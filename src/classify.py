import sklearn
import sys
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import gensim
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


 
# Parameter: classify.py, Name des Modells, Trainingsdatei, Testdatei, Speicherordner (Ergebnisse und Modell)


param = "classify.py Modell Features Trainingsdatei Testdatei Speicherordner\n"
embeds = "Mögliche Features: BOW, Unigramme, Doc2Vec\n"
modelle = "Mögliche Modelle: GaussianNB, MultinomialNB, LogisticRegression, RF\n"
USAGE = "Korrekte Nutzung:\n"+param+embeds+modelle

if len(sys.argv) != 6:
    raise Exception(USAGE)
else:
    modellname = sys.argv[1]
    embedding = sys.argv[2]
    train_path = sys.argv[3]
    test_path = sys.argv[4]
    save_at = sys.argv[5]


# Daten laden
with open(train_path, mode="r", encoding="utf-8") as in_train:
    train_data = in_train.readlines()
    train_data = [tweet.strip().split("\t") for tweet in train_data]
    train_labels = [tweet[2] for tweet in train_data]

with open(test_path, mode="r", encoding="utf-8") as in_test:
    test_data = in_test.readlines()
    test_data = [tweet.strip().split("\t") for tweet in test_data]
    test_labels = [tweet[2] for tweet in test_data]



# Preprocessing & Feature Extraction

#vectorizer = TfidfVectorizer(lowercase=False)
# Features:
# 1. Zeichen-N-Gramme, char_wb: Unigramme innerhalb von Wortgrenzen (Leerzeichen)
#vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(1, 1))
#X = vectorizer.fit_transform(text_data)


# Annotation konvertieren
labels_vvh = ["KEINE", "VVH"]
labels_gruppe = ["KeineGruppe", "Gruppe"]
labels_handlung = ["KeineHandlung", "Handlung"]

def labeling(label):
    if label == "VVH": return 1
    elif label == "KEINE": return 0
    elif label == "Gruppe": return 1
    elif label == "KeineGruppe": return 0
    elif label == "Handlung": return 1
    elif label == "KeineHandlung": return 0

tags_train = list(map(labeling, train_labels))
tags_test = list(map(labeling, test_labels))



# Bag of Words
if embedding == "BOW":
    X_train, X_test, y_train, y_test = [], [], [], []
    True

# doc2vec, s. https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html

# Preprocessing inkl. Tokenisierung
def read_corpus_gensim(datei, tokens_only=False):
    with open(datei, encoding="utf-8") as f:
        for i, line in enumerate(f):
            id, tweet, anno = line.strip().split("\t")
            # TODO: ändern, so dass nicht alles lowercase ist
            tokens = gensim.utils.simple_preprocess(tweet)
            if tokens_only:
                yield tokens
            else:
                # For training data, add tags; Tag z.B. Zeilennummer ([i]), oder: Korpus-ID
                yield gensim.models.doc2vec.TaggedDocument(tokens, [id])

# Vektoren berechnen; s. https://towardsdatascience.com/multi-class-text-classification-with-doc2vec-logistic-regression-9da9947b43f4;
def feature_vectors(model, docs, tags):
    targets, tags_final = zip(*[(tags[line], model.infer_vector(doc.words, epochs=20)) for line, doc in enumerate(docs)])
    return targets, tags_final


if embedding == "Doc2Vec":
    train_corpus = list(read_corpus_gensim(train_path))
    test_corpus = list(read_corpus_gensim(test_path))

    # Distributed Memory
    model_dm = gensim.models.doc2vec.Doc2Vec(vector_size=300, min_count=2, epochs=30)
    model_dm.build_vocab(train_corpus)
    model_dm.train(train_corpus, total_examples=model_dm.corpus_count, epochs=model_dm.epochs)

    # Distributed Bag of Words
    model_dbow = gensim.models.doc2vec.Doc2Vec(dm=0, vector_size=300, min_count=2, epochs=30)
    model_dbow.build_vocab(train_corpus)
    model_dbow.train(train_corpus, total_examples=model_dbow.corpus_count, epochs=model_dbow.epochs)

    # zur Kombination von DBOW und DM
    concat_model = ConcatenatedDoc2Vec([model_dbow, model_dm])

    y_train, X_train = feature_vectors(concat_model, train_corpus, tags_train)
    y_test, X_test = feature_vectors(concat_model, test_corpus, tags_test)



model = GaussianNB()

# Training
if modellname == "GaussianNB":
    model = GaussianNB()

model.fit(X_train,y_train)


# Modell speichern



# Evaluation
predicted = model.predict(X_test)
print(metrics.classification_report(y_test, predicted, target_names=labels_gruppe))
print(metrics.confusion_matrix(y_test, predicted))


# Evaluationsergebnisse speichern
