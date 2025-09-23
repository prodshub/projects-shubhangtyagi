import numpy as np  
from scipy.linalg import qr, solve_triangular

def create_matrices(df, label):
    """
    Create the A and y matrix from the dataframe.
    """
    # Drop the 'price' column
    X = df.drop(columns=[label])
    
    # Add a column of ones as the first column
    A = np.hstack((np.ones((X.shape[0], 1)), X.values))

    y = df[label].values.reshape(-1, 1)
    return A, y

def linear_regression_coefficients(A, y):
    """
    Calculate the linear regression coefficients using the normal equation.
    """
    # Calculate the coefficients
    coeff= np.linalg.pinv(A)@ y
    return coeff

def qr_regression_pivoting(A, y, tol=1e-10):
    Q, R, P = qr(A, mode='economic', pivoting=True)
    rank = np.sum(np.abs(np.diag(R)) > tol)
    R1 = R[:rank, :rank]
    Q1 = Q[:, :rank]
    y1 = Q1.T @ y

    # Solve R1 x = Q1^T y for the independent columns
    coeff_hat = solve_triangular(R1, y1)

    # Reconstruct full beta with zeros in place of dropped columns
    coeff = np.zeros((A.shape[1], 1))
    coeff[P[:rank]] = coeff_hat
    return coeff
def predict(new_data, coeff):
    """
    Predict label using new data and linear regression coefficients
    """
    new_input = np.hstack((np.ones((new_data.shape[0], 1)), new_data))
    prediction = new_input@coeff
    return prediction
def verify_full_rank(A):
    """Verify that matrix A has full column rank"""
    rank = np.linalg.matrix_rank(A)
    n_cols = A.shape[1]
    print(f"Matrix rank: {rank}, Number of columns: {n_cols}")
    print(f"Full rank: {'Yes' if rank == n_cols else 'No'}")
    return rank == n_cols













