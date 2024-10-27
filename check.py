import numpy as np
from sklearn.metrics import silhouette_score
from skimage.metrics import structural_similarity as ssim

#checks if two matrices are equal within a small tolerance
def matrices_are_equal_with_tolerance(mat1, mat2, tolerance=1e-4):
    return np.allclose(mat1, mat2, atol=tolerance)

def is_symmetric(matrix, tolerance=1e-4):
    return matrices_are_equal_with_tolerance(matrix, matrix.T, tolerance)

def mean_squared_error(A, B):
    return np.linalg.norm(A - B, np.inf)**2 / np.size(A)

def check_pseudo_inverse_properties_mse(A, A_ps_inv):
    print("\nInaccuracy A A^+ A = A")
    mse_1 = mean_squared_error(A, A @ A_ps_inv @ A)
    print(mse_1)
    
    print("Inaccuracy A^+ A A^+ = A^+")
    mse_2 = mean_squared_error(A_ps_inv, A_ps_inv @ A @ A_ps_inv)
    print(mse_2)

    print("Checking property: A^+ A is symmetric")
    condition_3 = is_symmetric(A_ps_inv @ A)
    print(f"Condition 3 (A^+ A symmetric): {condition_3}")
    
    print("Checking property: A A^+ is symmetric")
    condition_4 = is_symmetric(A @ A_ps_inv)
    print(f"Condition 4 (A A^+ symmetric): {condition_4}\n")

    return mse_1, mse_2, condition_3, condition_4

#added siluette check for the Y input matrix and 
#one found as the result of formula A @ X = Y
def compute_silhouette_score(Y_pred, Y):
    #flatten and concatenate Y and Y_pred for use in silhouette score calculation
    Y_flat = Y.flatten()
    Y_pred_flat = Y_pred.flatten()
    data = np.vstack((Y_flat.reshape(-1, 1), Y_pred_flat.reshape(-1, 1)))
    labels = np.concatenate((np.zeros(Y_flat.shape), np.ones(Y_pred_flat.shape))).ravel()

    #calculate silhouette score
    score = silhouette_score(data, labels)
    return score

#implement other measurings for comparison
def compute_image_metrics(Y_pred, Y):
    """Compute image comparison metrics: MSE and SSIM."""
    mse = mse = np.mean((Y_pred - Y) ** 2)
    data_range = 255.0 if Y.max() > 1 else 1.0
    ssim_score, _ = ssim(Y, Y_pred, data_range=data_range, full=True)
    return mse, ssim_score