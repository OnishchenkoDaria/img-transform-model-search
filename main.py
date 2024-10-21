import numpy as np
import cv2
import matplotlib.pyplot as plt

def transposed_matrix(S):
    rows, cols = S.shape
    S_t = np.zeros((cols, rows), dtype=float)  
    for i in range(rows):
        for j in range(cols):
            S_t[j][i] = S[i][j] 
    return S_t

#for accuracy check
def compare_matrices(A, B):
    if A.shape != B.shape:
        print("Matrices must have the same dimensions for comparison")
    
    #find max difference
    difference_matrix = np.abs(A - B)
    max_difference = np.max(difference_matrix)
    
    return max_difference

#reading input images X and Y
def read_img():
    #read the image & convert it to float32 for multiplication
    x_img = cv2.imread("x2.bmp", cv2.IMREAD_GRAYSCALE)
    y_img = cv2.imread("y2.bmp", cv2.IMREAD_GRAYSCALE)
    
    #display the image
    cv2.imshow("Image x", x_img)
    cv2.imshow("Image y", y_img)
    #wait for the user to press a key
    cv2.waitKey(0)
    #close all windows
    cv2.destroyAllWindows()
        
    x = x_img.astype(float)
    y = y_img.astype(float)

    print(f"x size: {x.shape} , y size: {y.shape}")
    return x, y

""" def transform_linear(x):
    #flatten the image matrix into vector
    input_flat = x.astype(float).flatten()
    
    #add a row of 1 to fit size
    X = np.vstack([input_flat, np.ones(input_flat.shape)])
    print(f"{X}")
    return X """

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
    print(A_ps_inv)
    #pseudo_inverse_check(A, A_ps_inv)
    return A, A_ps_inv, E_matrix

def pseudo_inverse_check(A, A_ps_inv):
    result1 = (A @ A_ps_inv) @ A
    differecne1 = compare_matrices(result1, A)
    print(differecne1)

    result2 = (A_ps_inv @ A) @ A_ps_inv
    differecne2 = compare_matrices(result2, A_ps_inv)
    print(differecne2)

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
    print(Z_t_Xt)

    V = np.zeros((Y.shape[0], X.shape[0]), dtype=float)

    print(Y.shape, X_ps_inv.shape, V.shape, Z_t_Xt.shape)
    # resize all to the rows size of matrix Y

    A = (Y @ X_ps_inv) + (V @ Z_t_Xt)
    print(A)
    return A

def model_by_Moore_Penrose(X, Y):
    X, X_ps_inv, E_matrix = Moor_Penrose_method(X)    

    A = find_A_model_MP(X, Y, X_ps_inv, E_matrix)
    Y_img = A @ X
    print(Y_img)

    Yimage_projected_MP = project_matrix_to_range(Y_img)

    cv2.imshow("Transformed Image", Yimage_projected_MP)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    X, Y = read_img()
    model_by_Moore_Penrose(X, Y)