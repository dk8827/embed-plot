import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

# Load the CSV file
df = pd.read_csv('imdb_tvshows.csv')

# Ensure the CSV has the required columns
if 'About' not in df.columns or 'Title' not in df.columns:
    raise ValueError("CSV must contain 'About' and 'Title' columns")

# Fill missing values in the 'About' column with an empty string
df['About'] = df['About'].fillna('')

# Filter rows by minimal number of votes
df_filtered = df[df['Votes'] >= 200000].copy()

# Load the sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Perform text embedding using the sentence transformer model
X = model.encode(df_filtered['About'].tolist())

# Reduce dimensionality to 2D
tsne = TSNE(n_components=2, random_state=42)
X_reduced = tsne.fit_transform(X)

NUMBER_OF_CLUSTERS = 5

# Perform KMeans clustering on the reduced dimensions
kmeans = KMeans(n_clusters=NUMBER_OF_CLUSTERS, random_state=42, n_init=10)
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

    # Draw convex hull around the cluster points
    if len(cluster_points) > 2:  # Convex hull requires at least 3 points
        hull = ConvexHull(cluster_points)
        for simplex in hull.simplices:
            plt.plot(cluster_points[simplex, 0], cluster_points[simplex, 1], colors[cluster_num])

plt.xlabel('TSNE Component 1')
plt.ylabel('TSNE Component 2')
plt.title('TV Series Embeddings in 2D Space with Clustering and Names')
plt.legend()
plt.grid(True)

# Compute and log cosine distances
distances = cosine_distances(X)
titles = df_filtered['Title'].tolist()
abouts = df_filtered['About'].tolist()
distance_tuples = []

for i in range(len(titles)):
    for j in range(i + 1, len(titles)):
        distance_tuples.append((titles[i], titles[j], distances[i, j], abouts[i], abouts[j]))

# Sort by decreasing distance (closest first)
distance_tuples.sort(key=lambda x: x[2])

# Take only the top 20 pairs and bottom 20 pairs
distance_tuples = distance_tuples[:20] + distance_tuples[-20:]

# Log the distances
for seriesA, seriesB, distance, aboutA, aboutB in distance_tuples:
    print(f"Distance between '{seriesA}' and '{seriesB}': {distance:.4f}")
    print(f"'{seriesA}' About: {aboutA}")
    print(f"'{seriesB}' About: {aboutB}")
    print()

plt.show(block=True)  # Keep the plot open at the end of the script
