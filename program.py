import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('imdb_tvshows.csv')

# Ensure the CSV has the required columns
if 'About' not in df.columns or 'Title' not in df.columns:
    raise ValueError("CSV must contain 'About' and 'Title' columns")

# Fill missing values in the 'description' column with an empty string
df['About'].fillna('', inplace=True)

# Filter rows by minimal number of votes
df_filtered = df[df['Votes'] >= 300000]

# Perform text embedding using TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df_filtered['About'])

# Reduce dimensionality to 2D using PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X.toarray())

# Plot the series names on a 2D map
plt.figure(figsize=(10, 8))
for i, series_name in enumerate(df_filtered['Title']):
    plt.scatter(X_reduced[i, 0], X_reduced[i, 1])
    plt.text(X_reduced[i, 0] + 0.01, X_reduced[i, 1] + 0.01, series_name, fontsize=9)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('TV Series Embeddings in 2D Space')
plt.grid(True)
plt.show()
