import numpy as np
import cv2
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

def Greville_method(A):
    rows, columns = A.shape
    A_ps_inv = np.zeros((columns, rows), dtype=float)
    
    for i in range(columns):
        #base case for firset row - base for further iterations
        if i == 0:
            a_1 = A[:, i]  
            #first column != 0 (else it remains zeros)
            if np.linalg.norm(a_1) != 0:
                A_ps_inv = (a_1 / np.dot(a_1, a_1)).reshape(1, -1) # reshape in stead of T -> makes it into row
        else:
            # i>1 case
            a_i = A[:, i]
            d_i = A_ps_inv @ a_i  
            c_i = a_i - (A[:, :i] @ d_i)

            if np.linalg.norm(c_i) != 0:
                b_i = (c_i / np.dot(c_i, c_i)).reshape(1, -1)
                #inverse matrix is achieved using outer
                B_i = A_ps_inv - np.outer(d_i, b_i)
                A_ps_inv = np.vstack([B_i, b_i])
            else:
                b_i = ((1 + np.dot(d_i, d_i)) ** (-1)) * d_i.T @ A_ps_inv
                A_ps_inv = np.vstack([A_ps_inv, b_i])              
    
    return A_ps_inv

#turning matrix into shades spectrum
def project_matrix_to_range(Yimage_MP, target_max=255):
    rows, columns = Yimage_MP.shape
    Yimage_projected_MP = np.zeros((rows, columns))
    
    ymax = np.max(Yimage_MP)
    ymin = np.min(Yimage_MP)
    
    for i in range(rows):
        for j in range(columns):
            Yimage_projected_MP[i, j] = target_max * (Yimage_MP[i, j] - ymin) / (ymax - ymin)
    
    return Yimage_projected_MP.astype(np.uint8)

def find_A_model_MP(X, Y, X_ps_inv, E_matrix):
    Z_Xt = E_matrix - X @ X_ps_inv
    Z_t_Xt = transposed_matrix(Z_Xt)
    V = np.zeros((Y.shape[0], X.shape[0]), dtype=float)
    A = (Y @ X_ps_inv) + (V @ Z_t_Xt)
    return A

def model_by_Greville(X, Y):
    print("\n****** GREVILLE METHOD ******\n")

    X_ps_inv = Greville_method(X)
    print(X_ps_inv)
    resize_matrix_to_smaller(X_ps_inv, (Y.shape[1], Y.shape[1]))
    
    c1, c2, c3, c4 = check_pseudo_inverse_properties_mse(X, X_ps_inv)
    A = find_A_model_MP(X, Y, X_ps_inv, np.eye(X.shape[0]))
    Y_img = A @ X
    
    #project and normalize the result matrix to a clear image
    Yimage_projected_MP = project_matrix_to_range(Y_img)

    cv2.imshow("Transformed Image", Yimage_projected_MP)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return c1, c2, c3, c4

