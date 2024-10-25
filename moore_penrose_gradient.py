import numpy as np
import cv2
import matplotlib.pyplot as plt
from check import check_pseudo_inverse_properties_mse

def transposed_matrix(S):
    rows, cols = S.shape
    S_t = np.zeros((cols, rows), dtype=float)  
    for i in range(rows):
        for j in range(cols):
            S_t[j][i] = S[i][j] 
    return S_t
def resize_matrix_to_smaller(A, target_shape):
    return A[:target_shape[0], :target_shape[1]]

def Moore_Penrose_formula(A_t, E_matrix, A, d):
    mult = A @ A_t
    mult1 = d**2 * E_matrix
    mult2 = np.linalg.inv(mult + mult1)
    expr = A_t @ mult2
    return expr

def Moore_Penrose_method_gradient(A):
    max_iterations = 100
    epsilon = 1e-4  # tolerance for convergence
    step = 0.1      # learning rate for gradient descent
    rows, columns = A.shape
    E_matrix = np.eye(rows)
    
    # Initialize pseudo-inverse with Moore-Penrose formula
    A_ps_inv = Moore_Penrose_formula(transposed_matrix(A), E_matrix, A, step)
    
    for iteration in range(max_iterations):
        # Compute inaccuracy: the difference between A and its reconstruction
        inaccuracy = (A @ A_ps_inv @ A) - A
        loss = np.linalg.norm(inaccuracy, np.inf)
        
        if loss < epsilon:
            print(f"Converged in {iteration} iterations")
            break
        
        # Compute the gradient for update
        gradient = Moore_Penrose_formula(transposed_matrix(A), E_matrix, A, step)
        
        # Update the pseudo-inverse with the gradient
        A_ps_inv -= step * gradient

    return A_ps_inv

# Turning matrix into shades spectrum
def project_matrix_to_range(Yimage_MP, target_max=255):
    p, n = Yimage_MP.shape
    Yimage_projected_MP = np.zeros((p, n))
    
    ymax = np.max(Yimage_MP)
    ymin = np.min(Yimage_MP)
    
    if ymax == ymin:  # Avoid division by zero
        return np.full((p, n), target_max // 2, dtype=np.uint8)
    
    # Apply normalization to the range [0, target_max]
    for i in range(p):
        for j in range(n):
            Yimage_projected_MP[i, j] = target_max * (Yimage_MP[i, j] - ymin) / (ymax - ymin)
    
    return Yimage_projected_MP.astype(np.uint8)

def find_A_model_MP(X, Y, X_ps_inv, E_matrix):
    Z_Xt = E_matrix - X @ X_ps_inv
    Z_t_Xt = transposed_matrix(Z_Xt)
    V = np.zeros((Y.shape[0], X.shape[0]), dtype=float)
    A = (Y @ X_ps_inv) + (V @ Z_t_Xt)
    return A

def model_by_Moore_Penrose_gradient(X, Y):
    print("\n****** MP GRADIENT METHOD ******\n")

    X_ps_inv = Moore_Penrose_method_gradient(X)
    E = np.eye(X.shape[0])
    
    print(X_ps_inv)
    c1, c2, c3, c4 = check_pseudo_inverse_properties_mse(X, 255-X_ps_inv)
    
    A = find_A_model_MP(X, Y, X_ps_inv, E)
    Y_img = A @ X
    
    Yimage_projected_MP = project_matrix_to_range(Y_img)
    Yimage_projected_MP = 255 - Yimage_projected_MP

    cv2.imshow("Transformed Image", Yimage_projected_MP)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return c1, c2, c3, c4