from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

ALPHA = 0.147
K_FOLD = 22

categories = ['alt.atheism', 'comp.graphics', 'sci.electronics', 'sci.space', 'talk.politics.misc']
#categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x']
news_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories=categories) #, 'quotes'
news_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), categories=categories) #, 'quotes'
pprint(list(news_train.target_names))

#print("Train data target labels:",news_train.target)
#print("Test data target labels:", news_test.target)
#print("Train data target names:",news_train.target_names)
#print("Test data target names:",news_test.target_names)
#print("Total train data:", len(news_train.data))
#print("Total test data:", len(news_test.data))

# So, first converting text data into vectors of numerical values using tf-idf to form feature vector
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
news_train_vectors = vectorizer.fit_transform(news_train.data)
news_test_vectors = vectorizer.transform(news_test.data)

print(news_train_vectors.shape)
print(news_train.target.shape)

Xtr = news_train_vectors # store training feature matrix in "Xtr"
#print("Xtr:", Xtr)
#print("Size: ", Xtr.shape)
Ytr = news_train.target # store training response vector in "ytr"
#print("Ytr:", Ytr)
#print(str(news_train.data[2712][0]))

Xtt = news_test_vectors
#print("Xtt:", Xtt)
Ytt = news_test.target
#print("Ytt:", Ytt)


# Implementing classification model- using MultinomialNB
clf_MNB = MultinomialNB(alpha=ALPHA) # Instantiate the estimator
clf_MNB.fit(Xtr, Ytr) # Fit the model with data (aka "model training")
y_pred = clf_MNB.predict(Xtt) # Predict the response for a new observation
#print("Predicted Class Labels:",y_pred)
y_pred_score_mnb = clf_MNB.predict_proba(Xtt) # Predict the response score for a new observation
#print("Predicted Score:\n",y_pred_score_mnb)

# cross-validation with MNB model
print("Multinomial Naive Bayes accuracy:",100*cross_val_score(clf_MNB, Xtr, Ytr, cv=K_FOLD, scoring='accuracy').mean())


def show_top10(classifier, vectorizer, categories):
    feature_names = np.asarray(vectorizer.get_feature_names())
    for i, category in enumerate(categories):
        top10 = np.argsort(classifier.coef_[i])[-10:]
        print("%s: %s" % (category, " ".join(feature_names[top10])))

show_top10(clf_MNB, vectorizer, news_train.target_names)


'''f = []
ALPHAS = np.arange(0.001, 1.0, 0.001)
for a in ALPHAS:
    f.append(100 * cross_val_score(MultinomialNB(alpha=a), Xtr, Ytr, cv=K_FOLD, scoring='accuracy').mean())

#print(f)

plt.plot(ALPHAS, f)
plt.xlabel('Value of ALPHA parameter')
plt.ylabel('Score (%)')
plt.show()

print("Best alpha: " + str(ALPHAS[f.index(max(f))]))
print("Score: " + str(max(f)))'''

'''k = np.arange(2, 27, 2)
sc = [85.735, 86.804, 87.541, 87.467, 87.653, 87.725, 87.762, 87.946, 87.798, 88.021, 88.168, 87.947, 88.020]

plt.plot(k, sc)
plt.xlabel('Value of K')
plt.ylabel('Score (%)')
plt.show()'''

#LEARNING CURVE Mean Squared Error
'''train_sizes = [1, 100, 200, 500, 1000, 1500, 2000, 2170]
train_sizes, train_scores, validation_scores = learning_curve(
    estimator=MultinomialNB(alpha=ALPHA),
    X=Xtr, y=Ytr,
    train_sizes=train_sizes,cv=K_FOLD,
    scoring='neg_mean_squared_error')

train_scores_mean = -train_scores.mean(axis = 1)
validation_scores_mean = -validation_scores.mean(axis = 1)

plt.style.use('seaborn')
plt.plot(train_sizes, train_scores_mean, label = 'Training error')
plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')
plt.ylabel('MSE', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Learning curves for a Multinomial NB', fontsize = 18, y = 1.03)
plt.legend()
#plt.ylim(0,40)
plt.show()'''

#LEARNING CURVE
train_sizes = [1, 100, 200, 500, 1000, 1500, 2000, 2170]
train_sizes, train_scores, validation_scores = learning_curve(
    MultinomialNB(alpha=ALPHA),
    Xtr, Ytr,
    train_sizes=train_sizes,cv=K_FOLD,
    scoring='accuracy')

validation_scores_mean = validation_scores.mean(axis = 1)

plt.style.use('seaborn')
plt.plot(train_sizes, validation_scores_mean)
plt.ylabel('Score', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Learning curve for a Multinomial NB', fontsize = 18, y = 1.03)
plt.legend()
plt.show()
