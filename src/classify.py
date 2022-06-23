import enum
# import sklearn
import sys
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import gensim
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

 
# Parameter: classify.py, Name des Modells, Trainingsdatei, Testdatei


param = "classify.py Modell Features Trainingsdatei Testdatei\n"
embeds = "Mögliche Features: BOW, Unigramme, Doc2Vec\n"
modelle = "Mögliche Modelle: GaussianNB, MultinomialNB, LogisticRegression, RF\n"
USAGE = "Korrekte Nutzung:\n"+param+embeds+modelle

if len(sys.argv) != 5:
    raise Exception(USAGE)
else:
    modellname = sys.argv[1]
    features = sys.argv[2]
    train_path = sys.argv[3]
    test_path = sys.argv[4]


# Daten laden
def read_corpus(path):
    with open(path, mode="r", encoding="utf-8") as inf:
        data = inf.readlines()
        data = [tweet.strip().split("\t") for tweet in data]
        labels = [tweet[2] for tweet in data]
    return (data, labels)

train_data, train_labels = read_corpus(train_path)
test_data, test_labels = read_corpus(test_path)


# Preprocessing & Feature Extraction


# Annotation konvertieren, Labels festlegen

if set(train_labels) == {"NEG", "NOT"}:
    labels = ["NEG", "NOT"]
if set(train_labels) == {"KEINE", "VVH"}:
    labels = ["KEINE", "VVH"]
if set(train_labels) == {"KeineGruppe", "Gruppe"}:
    labels = ["KeineGruppe", "Gruppe"]
if set(train_labels) == {"KeineHandlung", "Handlung"}:
    labels = ["KeineHandlung", "Handlung"]

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
if features == "BOW":
    # für Naive Bayes: binary=True
    vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(1, 2), lowercase=True)
    # Optional: stattdessen TfidfVectorizer
    X = vectorizer.fit_transform(train_data)
    X_train = X.toarray()
    X_test = vectorizer.transform(test_data)
    X_test = X_test.toarray()

    X_train, X_test, y_train, y_test = X_train, X_test, train_labels, test_labels

# doc2vec, s. https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html

# Preprocessing mit Gensim
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
    targets, tags_final = zip(*[(model.infer_vector(doc.words, epochs=20), tags[line]) for line, doc in enumerate(docs)])
    return targets, tags_final


if features == "Doc2Vec":
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

    X_train, y_train = feature_vectors(concat_model, train_corpus, tags_train)
    X_test, y_test = feature_vectors(concat_model, test_corpus, tags_test)





# Training

# Standardmodell
model = GaussianNB()

if modellname == "GaussianNB":
    model = GaussianNB()
elif modellname == "MultinomialNB":
    model = MultinomialNB()
elif modellname == "LogisticRegression":
    model = LogisticRegression()
elif modellname == "RandomForestClassifier":
    model = RandomForestClassifier()

model.fit(X_train,y_train)


# Modell speichern
#with open(save_at+"/Modell.pkl", mode="w", encoding="utf-8") as outf:
#    outf.write(pickle.dumps(model))


# Evaluation
predicted = model.predict(X_test)
print(metrics.classification_report(y_test, predicted, target_names=labels))
print(metrics.confusion_matrix(y_test, predicted))


# Evaluationsergebnisse speichern
#with open(save_at+"/Ergebnisse.txt", mode="w", encoding="utf-8") as oute:
#    specs = "Modell: " + str(modelle) + "; gespeichert unter " + str(save_at) + "/Modell.pkl" + "; Features: " + str(features)
#    oute.write(specs+"\n")
#    oute.write(str(metrics.classification_report(y_test, predicted, target_names=labels)))
#    oute.write("\nConfusion Matrix\n")
#    oute.write(str(metrics.confusion_matrix(y_test, predicted))) 