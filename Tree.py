from sklearn.tree import DecisionTreeClassifier
from pprint import pprint
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import pandas as pd

K_FOLD = 20
RANDOM_STATE = 730
MAX_DEPTH = 74
MIN_SAMPLES_SPLIT = 6

categories = ['alt.atheism', 'comp.graphics', 'sci.electronics', 'sci.space', 'talk.politics.misc']
#categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x']
news_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories=categories) #, 'quotes'
news_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), categories=categories) #, 'quotes'
pprint(list(news_train.target_names))

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

Xtt = news_test_vectors
#print("Xtt:", Xtt)
Ytt = news_test.target
#print("Ytt:", Ytt)

# Implementing classification model- using Decision Tree Classifier
clf = DecisionTreeClassifier(ccp_alpha= 0, max_depth=74, random_state=RANDOM_STATE, min_samples_split = MIN_SAMPLES_SPLIT)
clf.fit(Xtr, Ytr)
y_pred = clf.predict(Xtt)
y_pred_score_mnb = clf.predict_proba(Xtt)


# cross-validation with tree model
score = cross_val_score(clf, Xtr, Ytr, cv=K_FOLD, scoring='accuracy')
print("Decision Tree Classifier accuracy: " + str(100 * score.mean()) + "%")



'''K = np.arange(2,27,2)
MAX_DEPTHS = np.arange(1, 80)
sc = []
for k in K:
    f = []
    for d in MAX_DEPTHS:
        f.append(100 * cross_val_score(DecisionTreeClassifier(max_depth=d), Xtr, Ytr, cv=k, scoring='accuracy').mean())
    sc.append(max(f))


plt.plot(K, sc)
plt.xlabel('Value of K')
plt.ylabel('Score (%)')
plt.title("Decision tree classifier")
plt.show()

print("Best K: " + str(K[sc.index(max(sc))]))
print("Score: " + str(max(sc)))
'''

'''f = []
MAX_DEPTHS = np.arange(1, 80)
for d in MAX_DEPTHS:
    f.append(100 * cross_val_score(DecisionTreeClassifier(max_depth=d), Xtr, Ytr, cv=20, scoring='accuracy').mean())

#print(f)

plt.plot(MAX_DEPTHS, f)
plt.xlabel('Value of MAX_DEPTH parameter')
plt.ylabel('Score (%)')
plt.title("k = 20")
plt.show()

print("Best depth: " + str(MAX_DEPTHS[f.index(max(f))]))
print("Score: " + str(max(f)))'''

'''f = []
RANDOM_STATES = np.arange(0, 1000, 5)
for r in RANDOM_STATES:
    f.append(100 * cross_val_score(DecisionTreeClassifier(max_depth=74, random_state=r), Xtr, Ytr, cv=20, scoring='accuracy').mean())

#print(f)

plt.plot(RANDOM_STATES, f)
plt.xlabel('Value of RANDOM_STATE parameter')
plt.ylabel('Score (%)')
plt.title("k = 20 and depth = 74")
plt.show()

print("Best random_state: " + str(RANDOM_STATES[f.index(max(f))]))
print("Score: " + str(max(f)))'''

'''ALPHAS = np.arange(0, 1, 0.02)
f = []
for a in ALPHAS:
    f.append(100 * cross_val_score(DecisionTreeClassifier(ccp_alpha= a, max_depth=74, random_state=RANDOM_STATE, min_samples_split = MIN_SAMPLES_SPLIT), Xtr, Ytr, cv=20, scoring='accuracy').mean())
    print(a)
#print(f)

plt.plot(ALPHAS, f)
plt.xlabel('Value of CCP_ALPHA parameter')
plt.ylabel('Score (%)')
plt.title("k = 20, depth = 74 and random_state = 730")
plt.show()

print("Best ccp_alpha: " + str(ALPHAS[f.index(max(f))]))
print("Score: " + str(max(f)))'''

#LEARNING CURVE Mean Squared Error
'''train_sizes = [1, 100, 200, 500, 1000, 1500, 2000, 2170]
train_sizes, train_scores, validation_scores = learning_curve(
    estimator=DecisionTreeClassifier(max_depth=74, random_state=RANDOM_STATE, min_samples_split = MIN_SAMPLES_SPLIT),
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
plt.title('Learning curves for a Decision Tree Classifier', fontsize = 18, y = 1.03)
plt.legend()
plt.show()'''


#LEARNING CURVE
train_sizes = [1, 100, 200, 500, 1000, 1500, 2000, 2170]
train_sizes, train_scores, validation_scores = learning_curve(
    DecisionTreeClassifier(max_depth=74, random_state=RANDOM_STATE, min_samples_split = MIN_SAMPLES_SPLIT),
    Xtr, Ytr,
    train_sizes=train_sizes,cv=K_FOLD,
    scoring='accuracy')


validation_scores_mean = validation_scores.mean(axis = 1)

plt.style.use('seaborn')
plt.plot(train_sizes, validation_scores_mean)
plt.ylabel('Score', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Learning curve for a Decision Tree Classifier', fontsize = 18, y = 1.03)
plt.legend()
plt.show()
