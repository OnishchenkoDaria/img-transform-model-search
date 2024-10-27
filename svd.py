import numpy as np
import cv2
from check import check_pseudo_inverse_properties_mse

#transposes a matrix
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

def pseudo_inverse_from_L(L):
    rows, columns = L.shape
    L_pseudo_inv = np.zeros((columns, rows))
    #make diagonal elements ^-1
    for i in range(min(rows, columns)):
        if L[i, i] != 0:
            L_pseudo_inv[i, i] = 1 / L[i, i]
    
    return L_pseudo_inv


def SVD_method(A):
    U, L_elements, Vt = np.linalg.svd(A, full_matrices=False)
    #make the matrix of the found elements in diagonal
    L = np.diag(L_elements)
    L_ps_inv = pseudo_inverse_from_L(L)

    A_ps_inv = Vt.T @ L_ps_inv @ U.T

    return A_ps_inv

# Normalizes the image matrix to a specific range
def project_matrix_to_range(Yimage_MP, target_max=255):
    p, n = Yimage_MP.shape
    Yimage_projected_MP = np.zeros((p, n))
    
    ymax = np.max(Yimage_MP)
    ymin = np.min(Yimage_MP)
    
    for i in range(p):
        for j in range(n):
            Yimage_projected_MP[i, j] = target_max * (Yimage_MP[i, j] - ymin) / (ymax - ymin)
    
    return Yimage_projected_MP.astype(np.uint8)

# Computes the transformation matrix A based on Moore-Penrose inverse
def find_A_model_MP(X, Y, X_ps_inv, E_matrix):
    Z_Xt = E_matrix - X @ X_ps_inv
    Z_t_Xt = transposed_matrix(Z_Xt)
    V = np.zeros((Y.shape[0], X.shape[0]), dtype=float)
    A = (Y @ X_ps_inv) + (V @ Z_t_Xt)
    return A

# Applies the Moore-Penrose inverse model to transform the image
def model_by_SVD(X, Y):
    print("\n****** SVD METHOD ******\n")

    X_ps_inv = SVD_method(X)
    print(X_ps_inv)
    
    c1, c2, c3, c4 = check_pseudo_inverse_properties_mse(X, X_ps_inv)

    A = find_A_model_MP(X, Y, X_ps_inv, np.eye(X.shape[0]))
    Y_img = A @ X

    # Project and normalize the result matrix to a clear image
    Yimage_projected_MP = project_matrix_to_range(Y_img)

    cv2.imshow("Transformed Image", Yimage_projected_MP)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return c1, c2, c3, c4, Y_img