import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
def createMatricies1(df, k):
    """
    Create X, C matricies
    """
    X = df.values
    C = X[np.random.choice(X.shape[0], k, replace=False)]
    return X, C

def assignClusters(X, C, k):
    """
    Create D, A matricies
    """
    D = np.sqrt(((X[:, np.newaxis, :] - C[np.newaxis, :, :]) ** 2).sum(axis=2))
    labels = np.argmin(D, axis=1)
    
    # Create assignment matrix A
    A = np.zeros((X.shape[0], k))
    A[np.arange(X.shape[0]), labels] = 1
    
    return D, A
def kMeans(X, C, k, max_iter=100, tol=1e-6):
    """
    K-means algorithm
    """
    prev_centroids = C.copy()
    
    for i in range(max_iter):
        # Assign clusters
        D, A = assignClusters(X, C, k)
        # Update centroids
        for j in range(k):
            cluster_points = X[A[:, j] == 1]
            if len(cluster_points) > 0:
                C[j] = cluster_points.mean(axis=0)
        
        # Check for convergence
        if np.allclose(C, prev_centroids, atol=tol):
            print(f"Converged after {i+1} iterations")
            break
        prev_centroids = C.copy()
    
    # Final assignment
    D, A = assignClusters(X, C, k)
    return A, C

def computeInertia(X, C, A):
    """
    Compute total within-cluster sum of squared distances (inertia)
    """
    inertia = 0
    for i in range(X.shape[0]):
        c_idx = np.argmax(A[i])
        inertia += np.linalg.norm(X[i] - C[c_idx]) ** 2
    return inertia
def findK(df, k_range=range(1,10)):
    """
    Use elbow method to find optimal number of clusters
    """
    inertias = []
    for k in k_range:
        X, C = createMatricies1(df, k)
        D, A = assignClusters(X, C, k)
        A_final, C_final = kMeans(X, C, k, max_iter=100)
        inertia = computeInertia(X, C_final, A_final)
        inertias.append(inertia)
        plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertias, marker='o')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia (Total Within-Cluster SSE)")
    plt.title("Elbow Method for Optimal k")
    plt.grid(True)
    plt.show()

synthetic_data2 = {
    'x': [-6, -5, -5, -4, -6, -4, -5, -5,      # Cluster 1 around (-5, -5)
           0,  1,  0, -1,  1, -1,  0.5, -0.5,  # Cluster 2 around (0, 0)
           4,  5,  6,  5,  4,  6,  5.5, 4.5],  # Cluster 3 around (5, 5)
    'y': [-5, -6, -4, -5, -4, -6, -5.5, -5, 
           1,  0, -1,  0, -1,  1,  0.5, -0.5,
           5,  6,  5,  4,  6,  4,  5.5, 4.5]
}

