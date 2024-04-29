import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

iris = load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris.data,
                                                    iris.target,
                                                    test_size=3/10,
                                                    random_state=4)
k_range = range(1, 26)
scores = {}
scores_list = []
ideal_k = 1
max_score = 0
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    scores[k] = accuracy_score(y_pred, y_test)
    if (scores[k] > max_score):
        max_score = scores[k]
        ideal_k = k
    scores_list.append(scores[k])

plt.plot(k_range, scores_list)
plt.xlabel('Values of K for KNN')
plt.ylabel('Testing Accuracy')
plt.show()

print(f'The ideal k value is: {ideal_k}\n')
knn = KNeighborsClassifier(n_neighbors=ideal_k)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
accuracy = accuracy_score(y_pred, y_test)
print(f'The accuracy is: {accuracy}\n')
cm = confusion_matrix(y_pred, y_test)
print('Confusion Matrix:')
print(cm)
