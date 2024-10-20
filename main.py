import numpy as np
import cv2
import matplotlib.pyplot as plt

def E(n):
    E_matrix = np.eye(n)  
    return E_matrix

def transposed_matrix(S):
    rows, cols = S.shape
    S_t = np.zeros((cols, rows), dtype=float)  
    for i in range(rows):
        for j in range(cols):
            S_t[j][i] = S[i][j] 
    return S_t

def multiply_matrix(S, D):
    rows_S, cols_S = S.shape  
    rows_D, cols_D = D.shape  

    if cols_S != rows_D:
        raise ValueError("matrix dimensions do not match for multiplication")

    R = np.zeros((rows_S, cols_D), dtype=float)

    for i in range(rows_S):
        for j in range(cols_D):
            for k in range(cols_S):
                R[i][j] += S[i][k] * D[k][j]

    return R

def print_matrix(matrix, label="Matrix"):
    print(f"{label}:")
    for row in matrix:
        formatted_row = " ".join(f"{val:7.2f}" for val in row)
        print(formatted_row)
    print()

def read_img():
    #read the image & convert it to float32 for multiplication
    x = cv2.imread("x2.bmp", cv2.IMREAD_GRAYSCALE)
    y = cv2.imread("y2.bmp", cv2.IMREAD_GRAYSCALE)
    
    #display the image
    cv2.imshow("Image x", x)
    cv2.imshow("Image y", y)
    #wait for the user to press a key
    cv2.waitKey(0)
    #close all windows
    cv2.destroyAllWindows()
        
    return x.astype(np.float32), y.astype(np.float32)

def transform_linear(x):
    #flatten the image matrix into vector
    input_flat = x.astype(float).flatten()
    
    #add a row of 1 to fit size
    X = np.vstack([input_flat, np.ones(input_flat.shape)])
    print(f"{X}")
    return X

def resize_matrix_to_smaller(A, target_shape):
    A_resized = A[:target_shape[0], :target_shape[1]]
    return A_resized

def Moor_Penrose_formula(A_t, E_matrix, A, d):
    mult = A @ A_t
    
    mult1 = d**2 * E_matrix

    mult2 = np.linalg.inv(mult + mult1)

    expr = A_t @ mult2
    return expr

def Moor_Penrose_method(A):
    # A_ps_inv - pseudo inverse A matrix
    epsilon = 1e-4    
    rows, columns = A.shape

    A_t = transposed_matrix(A)
    print(f"rows, columns: {rows, columns}")

    d = 0.1
    E_matrix = E(columns)

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
    print(A_ps_inv)
    return 

def pseudo_inverse_check(A, A_ps_inv):
    result1 = A @ A_ps_inv @ A
    if not np.allclose(result1, A):
        return False

    result2 = A_ps_inv @ A @ A_ps_inv
    if not np.allclose(result2, A_ps_inv):
        return False

    return True

if __name__ == "__main__":
    X, Y = read_img()
    Moor_Penrose_method(X)
