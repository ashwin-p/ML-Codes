from sklearn.datasets import fetch_california_housing
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

housing = fetch_california_housing(as_frame=True)
x = housing.data[['Latitude', 'Longitude', 'MedInc']]
y = housing.target
sns.scatterplot(data=x, x='Latitude', y='Longitude', hue='MedInc')
plt.show()

x_norm = normalize(x)

K = range(2, 8)
fits = []
scores = []
ideak_k = 2
max_score = 0

for k in K:
    model = KMeans(n_clusters=k, random_state=0, n_init='auto')
    model.fit(x_norm)
    score = silhouette_score(x_norm, model.labels_, metric='euclidean')
    scores.append(score)
    if (score > max_score):
        max_score = score
        ideal_k = k

sns.lineplot(x=K, y=scores)
plt.show()
model = KMeans(n_clusters=ideal_k, random_state=0, n_init='auto')
model.fit(x_norm)
sns.scatterplot(data=x, x='Latitude', y='Longitude', hue=model.labels_)
plt.show()
