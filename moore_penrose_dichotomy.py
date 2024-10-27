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
    A_resized = A[:target_shape[0], :target_shape[1]]
    return A_resized

def Moor_Penrose_formula(A_t, E_matrix, A, d):
    mult = A @ A_t
    
    #dividing operation for better control
    mult1 = d**2 * E_matrix
    mult2 = np.linalg.inv(mult + mult1)

    expr = A_t @ mult2
    return expr

def Moor_Penrose_method_dichotomy(A):
    # A_ps_inv - pseudo inverse A matrix
    epsilon = 1e-4    
    rows, columns = A.shape

    A_t = transposed_matrix(A)

    d = 0.1
    E_matrix = np.eye(columns)

    #resize to match the required dimensions for multiplication
    target_shape = (columns, A.shape[1]) 
    A = resize_matrix_to_smaller(A, target_shape)
    A_t = resize_matrix_to_smaller(A_t, (A.shape[1], columns))

    prev_guess = Moor_Penrose_formula(A_t, E_matrix, A, d)
    next_guess = Moor_Penrose_formula(A_t, E_matrix, A, d / 2)

    while np.linalg.norm(next_guess - prev_guess, np.inf) > epsilon:
        prev_guess = next_guess
        d = d / 2
        next_guess = Moor_Penrose_formula(A_t, E_matrix, A, d)
    
    A_ps_inv = next_guess
    #pseudo_inverse_check(A, A_ps_inv)
    return A, A_ps_inv, E_matrix

#turning matrix into shades spectrum
def project_matrix_to_range(Yimage_MP, target_max=255):
    p, n = Yimage_MP.shape
    Yimage_projected_MP = np.zeros((p, n))
    
    ymax = np.max(Yimage_MP)
    ymin = np.min(Yimage_MP)
    
    for i in range(p):
        for j in range(n):
            Yimage_projected_MP[i, j] = target_max * (Yimage_MP[i, j] - ymin) / (ymax - ymin)
    
    return Yimage_projected_MP.astype(np.uint8)

def find_A_model_MP(X, Y, X_ps_inv, E_matrix):
    # A = Y * X^+ + V * Z_t (X_t)
    Z_Xt = E_matrix - X @ X_ps_inv
    Z_t_Xt = transposed_matrix(Z_Xt)

    V = np.zeros((Y.shape[0], X.shape[0]), dtype=float)

    # resize all to the rows size of matrix Y

    A = (Y @ X_ps_inv) + (V @ Z_t_Xt)
    return A

def model_by_Moore_Penrose_dichotomy(X, Y):
    print("\n****** MP WITH DICHOTOMY METHOD ******\n")
    X_mp, X_ps_inv_mp, E_matrix_mp = Moor_Penrose_method_dichotomy(X)    
    resize_matrix_to_smaller(X_ps_inv_mp, (Y.shape[1], Y.shape[1]))

    X_ps_inv = X_ps_inv_mp 
    X = X_mp
    E = E_matrix_mp

    print(X_ps_inv)
    c1, c2, c3, c4 = check_pseudo_inverse_properties_mse(X, X_ps_inv)

    A = find_A_model_MP(X, Y, X_ps_inv, E)
    Y_img = A @ X

    # Transform the matrix back into an image
    Yimage_projected_MP = project_matrix_to_range(Y_img)

    cv2.imshow("Transformed Image", Yimage_projected_MP)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return c1, c2, c3, c4, Y_img