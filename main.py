import numpy as np
import cv2
from sympy import symbols, limit

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
        raise ValueError("Matrix dimensions do not match for multiplication")

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
    # Read the image & convert it to float32 for multiplication
    input_image = cv2.imread("x2.bmp", cv2.IMREAD_GRAYSCALE).astype(np.float32)
    
    # Flatten the image matrix into vector
    input_flat = input_image.flatten()
    
    # Add a row of ones for the bias term
    X = np.vstack([input_flat, np.ones(input_flat.shape)])
    
    return X

def Moor_Penrose_formula(A_t, E_matrix, A, d):
    mult = multiply_matrix(A, A_t)
    #print_matrix(mult)
    
    mult1 = d**2 * E_matrix
    #print_matrix(mult1)

    mult2 = np.linalg.inv(mult + mult1)
    #rint_matrix(mult2)

    #d = symbols('d')
    expr = multiply_matrix(A_t, mult2)
    A_ps_inv = limit(expr, d**2, 0)
    return A_ps_inv

def Moor_Penrose_method(A):
    # A_ps_inv - pseudo inverse A matrix
    epsilon = 1e-4
    A_t = transposed_matrix(A)
    
    rows, columns = A.shape
    d = 0.1
    E_matrix = E(rows) #should be colums but too big matrix

    prev_guess = Moor_Penrose_formula(A_t, E_matrix, A, d)
    next_guess = Moor_Penrose_formula(A_t, E_matrix, A, d/2)

    while np.linalg.norm(next_guess - prev_guess, np.inf) > epsilon:
        prev_guess = next_guess
        d = d / 2
        next_guess = Moor_Penrose_formula(A_t, E_matrix, A, d)
    
    A_ps_inv = next_guess
    print(A_ps_inv)
    """ 
    # Calculate the inner expression
    inner_expr = np.linalg.inv(multiply_matrix(A, A_t) + d**2 * E_matrix)
    
    # Calculate the pseudo-inverse
    expr = multiply_matrix(inner_expr, A_t)
    A_ps_inv = limit(expr, d**2, 0)
    
    print_matrix(A_ps_inv, "Moor-Penrose")"""
    return 

if __name__ == "__main__":
    X = read_img()
    Moor_Penrose_method(X)
