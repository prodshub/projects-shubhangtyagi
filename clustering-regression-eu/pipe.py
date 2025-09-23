import pandas as pd
import kMeansCluster as kMeans
import numpy as np
from numpy.linalg import matrix_rank
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import leastsquaresbestfit as ls
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import geopandas as gpd
from sklearn.cluster import SpectralClustering
from sklearn.linear_model import LinearRegression

# Load the dataset
file_path = 'eu_common_2021_indicators.csv'
df = pd.read_csv(file_path, encoding='utf-8')
print(f"Dataset shape: {df.shape}")
# Remove rows with missing values
cleaned_df = df.dropna()
# Remove the 'country' column
cleaned_df = cleaned_df.drop(columns=['country'])
# Normalize the data except for the 'Total fertility rate'
# Identify columns to normalize
columns_to_normalize = cleaned_df.columns[cleaned_df.columns != 'TOTFERRT: Total fertility rate']
# Initialize the scaler
scaler = MinMaxScaler()
# Normalize the selected columns
cleaned_df[columns_to_normalize] = scaler.fit_transform(cleaned_df[columns_to_normalize])
# Display the head of the cleaned and normalized dataframe
print(cleaned_df.head())

#Elbow method to find optimal number of clusters
print("\n" + "="*50)
print("ELBOW METHOD FOR OPTIMAL K")
print("="*50)
kMeans.findK(cleaned_df, k_range=range(1, 10))
# Kmeans clustering with k=5
print("\n" + "="*50)
print("TESTING K-MEANS CLUSTERING")
print("="*50)
k = 5
X, C = kMeans.createMatricies1(cleaned_df, k)
print(f"Data shape: {X.shape}")
print(f"Initial centroids shape: {C.shape}")
A_final, C_final = kMeans.kMeans(X, C, k, max_iter=100)
   # Get cluster assignments
cluster_labels = np.argmax(A_final, axis=1)
cleaned_df['K-Cluster'] = cluster_labels
print(f"\nCluster distribution:")
print(pd.Series(cluster_labels).value_counts().sort_index())
# Spectral clustering
print("\n" + "="*50)
print("TESTING SPECTRAL CLUSTERING")
print("="*50)
n_clusters = 5
model = SpectralClustering(
    n_clusters=n_clusters,
    affinity='rbf',
    gamma=0.5, 
    random_state=42
)
clusters = model.fit_predict(cleaned_df)
# Add spectral clustering results to the original dataframe
#cleaned_df['Spectral_Cluster'] = clusters
cleaned_df.to_csv('eu_common_2021_indicators_with_clusters.csv', index=False)
print(f"\nCluster distribution:")
print(pd.Series(clusters).value_counts().sort_index())
feature_cols = [col for col in cleaned_df.columns if col != 'TOTFERRT: Total fertility rate' and col != 'K-Cluster']
for cluster_id in range(k):
    cluster_data = cleaned_df[cleaned_df['K-Cluster'] == cluster_id].copy()
    if len(cluster_data) > 0:
        print(f"\nCluster {cluster_id} ({len(cluster_data)} samples):")

        # Design matrix and label
        X = cluster_data[feature_cols].values
        y = cluster_data['TOTFERRT: Total fertility rate'].values

        model = LinearRegression()
        model.fit(X, y)

        # Intercept and coefficients
        intercept = model.intercept_
        coeffs = model.coef_

        # Rank check
        is_full_rank = matrix_rank(X) == X.shape[1]

        print(f"Design matrix shape: {X.shape}")
        print(f"Is full rank: {is_full_rank}")
        print(f"Coefficients shape: ({coeffs.shape[0] + 1}, 1)")

        # Compute average values
        cluster_means = cluster_data[feature_cols].mean()
        label_avg = cluster_data['TOTFERRT: Total fertility rate'].mean()

        print("Feature Averages and Coefficients:")
        print(f"{'Feature':<30}{'Coefficient':>15}")
        print("-" * 60)
        print(f"{'Intercept':<30}{'–':>15}{intercept:>15.4f}")  # Intercept
        prediction_at_mean = intercept

        for i, name in enumerate(feature_cols):
            avg_val = cluster_means[name]
            coef_val = coeffs[i]
            prediction_at_mean += avg_val * coef_val
            print(f"{name:<30}{coef_val:>15.4f}")

        print(f"{'Label average (Fertility Rate)':<35}{label_avg:>15.4f}")
        print(f"{'Prediction at mean features':<35}{prediction_at_mean:>15.4f}")
# Save the country-cluster mapping
# Plot K-Means
colors = ['red', 'blue', 'green', 'orange', 'purple']  # up to 5 clusters
cluster_colors = [colors[label] for label in cluster_labels]

plt.figure(figsize=(10, 6))
plt.scatter(df.iloc[:, 9], df.iloc[:, 12], c=cluster_colors, s=40)
plt.xlabel('GDP per Capita (USD)')
plt.ylabel('Fertility Rate')
plt.title('Spectral Clustering Results')
plt.show()
# Plot Spectral Clustering
plt.figure(figsize=(10, 6))
plt.scatter(df.iloc[:, 9], df.iloc[:, 12], c=clusters, cmap='viridis')
plt.xlabel('GDP per Capita (USD)')
plt.ylabel('Fertility Rate')
plt.title('Spectral Clustering Results')
plt.show()


