import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from collections import Counter

# Load the CSV file
df = pd.read_csv('imdb_tvshows.csv')

# Ensure the CSV has the required columns
if 'About' not in df.columns or 'Title' not in df.columns:
    raise ValueError("CSV must contain 'About' and 'Title' columns")

# Fill missing values in the 'About' column with an empty string
df['About'].fillna('', inplace=True)

# Filter rows by minimal number of votes
df_filtered = df[df['Votes'] >= 200000]

# Load the sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Perform text embedding using the sentence transformer model
X = model.encode(df_filtered['About'].tolist())

# Reduce dimensionality to 2D using PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Perform KMeans clustering on the reduced dimensions
kmeans = KMeans(n_clusters=5, random_state=42)  # You can change the number of clusters
clusters = kmeans.fit_predict(X_reduced)

# Assign cluster labels to the dataframe
df_filtered['Cluster'] = clusters

# Extract key terms for each cluster
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_tfidf = vectorizer.fit_transform(df_filtered['About'])

# Get top terms for each cluster
n_terms = 5
cluster_names = []
for cluster_num in range(kmeans.n_clusters):
    cluster_indices = (clusters == cluster_num)
    cluster_texts = df_filtered.loc[cluster_indices, 'About']
    cluster_tfidf = vectorizer.transform(cluster_texts)
    mean_tfidf = cluster_tfidf.mean(axis=0).A1
    top_terms = [vectorizer.get_feature_names_out()[i] for i in mean_tfidf.argsort()[-n_terms:]]
    cluster_name = ", ".join(top_terms)
    cluster_names.append(cluster_name)

# Plot the series names on a 2D map with cluster names
plt.figure(figsize=(20, 12))
colors = [f'C{i}' for i in range(kmeans.n_clusters)]
for cluster_num in range(kmeans.n_clusters):
    cluster_indices = (clusters == cluster_num)
    cluster_points = X_reduced[cluster_indices]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[cluster_num], label=cluster_names[cluster_num])
    for i, series_name in enumerate(df_filtered.loc[cluster_indices, 'Title']):
        plt.text(cluster_points[i, 0] + 0.01, cluster_points[i, 1] + 0.01, series_name, fontsize=12)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('TV Series Embeddings in 2D Space with Clustering and Names')
plt.legend()
plt.grid(True)
plt.show()
