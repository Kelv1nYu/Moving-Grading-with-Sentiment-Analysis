import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances

import nltk
import string

import matplotlib.pyplot as plt

# load in dataset
df = pd.read_csv('RT_cleaned.csv', header = 0)
print(df.head())

# drop unwanted columns
df = df.drop(columns=['Unnamed: 0','id', 'publisher', 'date', 'critic', 'top_critic'])
print(df.head())

# rating: 2.0/5 -> 2.0
# delete '[Full review in Spanish]' in the review
df['rating'] = df['rating'].astype(str).str[:-2]
df['review'] = df['review'].map(lambda x: x.rstrip('[Full review in Spanish]'))
print(df.head())

# show how rating distributed
import matplotlib.pyplot as plt
ax = df['rating'].value_counts().plot(kind='bar',
                                    figsize=(14,8),
                                    title="Frequency of Rating")
ax.set_xlabel("Rating")
ax.set_ylabel("Frequency")
plt.show()

# convert to lower case, remove punctuation
from nltk.corpus import stopwords
import re

def clean_column(data):
  data =  data.lower()
  data = re.sub('re:', '', data)
  data = re.sub('-', '', data)
  data = re.sub('_', '', data)
  # removes punctuation
  data = re.sub(r'[^\w\s]','',data)
  data = re.sub(r'\n',' ',data)
  data = re.sub(r'[0-9]+','',data)
  return data

df['review_new'] = df['review'].apply(clean_column)
print(df.head())

import nltk
nltk.download('stopwords')

# remove stopwords, remove some meaningless words
stopwords_list = stopwords.words('english')
stopwords_list += ['movie', 'film', 'good', 'great', 'review', 'just', 'fun', 'like', 'enjoy', 'best',
                   'wa', 'hi', 'ha', 'movi', 'movies', 'films', 'better', 'theres', 'really']
df['review_new'] = df['review_new'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords_list)]))
print(df.head())

# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
data = df['review_new']

tf_idf_vectorizor = TfidfVectorizer(stop_words = 'english', #tokenizer = tokenize_and_stem,
                             max_features = 30000)
tf_idf = tf_idf_vectorizor.fit_transform(data)
tf_idf_norm = normalize(tf_idf)
tf_idf_array = tf_idf_norm.toarray()
pd.DataFrame(tf_idf_array, columns=tf_idf_vectorizor.get_feature_names()).head()


class Kmeans:
    """ K Means Clustering

    Parameters
    -----------
        k: int , number of clusters

        seed: int, will be randomly set if None

        max_iter: int, number of iterations to run algorithm, default: 200

    Attributes
    -----------
       centroids: array, k, number_features

       cluster_labels: label for each data point

    """

    def __init__(self, k, seed=None, max_iter=200):
        self.k = k
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)
        self.max_iter = max_iter

    def initialise_centroids(self, data):
        """Randomly Initialise Centroids

        Parameters
        ----------
        data: array or matrix, number_rows, number_features

        Returns
        --------
        centroids: array of k centroids chosen as random data points
        """

        initial_centroids = np.random.permutation(data.shape[0])[:self.k]
        self.centroids = data[initial_centroids]

        return self.centroids

    def assign_clusters(self, data):
        """Compute distance of data from clusters and assign data point
           to closest cluster.

        Parameters
        ----------
        data: array or matrix, number_rows, number_features

        Returns
        --------
        cluster_labels: index which minmises the distance of data to each
        cluster

        """

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        dist_to_centroid = pairwise_distances(data, self.centroids, metric='euclidean')
        self.cluster_labels = np.argmin(dist_to_centroid, axis=1)

        return self.cluster_labels

    def update_centroids(self, data):
        """Computes average of all data points in cluster and
           assigns new centroids as average of data points

        Parameters
        -----------
        data: array or matrix, number_rows, number_features

        Returns
        -----------
        centroids: array, k, number_features
        """

        self.centroids = np.array([data[self.cluster_labels == i].mean(axis=0) for i in range(self.k)])

        return self.centroids

    def convergence_calculation(self):
        """
        Calculates

        """
        pass

    def predict(self, data):
        """Predict which cluster data point belongs to

        Parameters
        ----------
        data: array or matrix, number_rows, number_features

        Returns
        --------
        cluster_labels: index which minmises the distance of data to each
        cluster
        """

        return self.assign_clusters(data)

    def fit_kmeans(self, data):
        """
        This function contains the main loop to fit the algorithm
        Implements initialise centroids and update_centroids
        according to max_iter
        -----------------------

        Returns
        -------
        instance of kmeans class

        """
        self.centroids = self.initialise_centroids(data)

        # Main kmeans loop
        for iter in range(self.max_iter):

            self.cluster_labels = self.assign_clusters(data)
            self.centroids = self.update_centroids(data)
            if iter % 100 == 0:
                print("Running Model Iteration %d " % iter)
        print("Model finished running")
        return self


# optimal number of clusters, 20
sklearn_pca = PCA(n_components = 2)
Y_sklearn = sklearn_pca.fit_transform(tf_idf_array)

number_clusters = range(3, 30)

kmeans = [KMeans(n_clusters=i, max_iter = 600) for i in number_clusters]
kmeans

score = [kmeans[i].fit(Y_sklearn).score(Y_sklearn) for i in range(len(kmeans))]
score

plt.plot(number_clusters, score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
# plt.title('Elbow Method')
plt.show()

# train the model
from sklearn.cluster import KMeans
n_clusters = 20
sklearn_pca = PCA(n_components = 2)
Y_sklearn = sklearn_pca.fit_transform(tf_idf_array)
kmeans = KMeans(n_clusters= n_clusters, max_iter=600, algorithm = 'auto')
%time fitted = kmeans.fit(Y_sklearn)
prediction = kmeans.predict(Y_sklearn)

plt.scatter(Y_sklearn[:, 0], Y_sklearn[:, 1],c=prediction ,s=50, cmap='viridis')

centers2 = fitted.cluster_centers_
plt.scatter(centers2[:, 0], centers2[:, 1],c='black', s=300, alpha=0.6);


# Top words in each Cluster
def get_top_features_cluster(tf_idf_array, prediction, n_feats):
    labels = np.unique(prediction)
    dfs = []
    for label in labels:
        id_temp = np.where(prediction==label) # indices for each cluster
        x_means = np.mean(tf_idf_array[id_temp], axis = 0) # returns average score across cluster
        sorted_means = np.argsort(x_means)[::-1][:n_feats] # indices with top 20 scores
        features = tf_idf_vectorizor.get_feature_names()
        best_features = [(features[i], x_means[i]) for i in sorted_means]
        df = pd.DataFrame(best_features, columns = ['features', 'score'])
        dfs.append(df)
    return dfs
dfs = get_top_features_cluster(tf_idf_array, prediction, 10)

# cluster column shows which sentence belongs to which cluster
df['cluster'] = kmeans.predict(Y_sklearn)
print(df.head())

# pie plot showing the frequency of clusters
ax = df['cluster'].value_counts().plot(kind='pie',
                                    figsize=(14,8),
                                    title="Frequency of Cluster")
ax.set_xlabel("Cluster")
ax.set_ylabel("Frequency")
plt.show()

# see top words of cluster 14, 0, 11
import seaborn as sns
plt.figure(figsize=(8,6))
sns.barplot(x = 'score' , y = 'features', orient = 'h' , data = dfs[14][:15])