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

# Reading input images X and Y
def read_img():
    # Read the image & convert it to float32 for multiplication
    x_img = cv2.imread("x2.bmp", cv2.IMREAD_GRAYSCALE)
    y_img = cv2.imread("y2.bmp", cv2.IMREAD_GRAYSCALE)
    
    # Display the image
    cv2.imshow("Image x", x_img)
    cv2.imshow("Image y", y_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        
    x = x_img.astype(float)
    y = y_img.astype(float)

    print(f"x size: {x.shape}, y size: {y.shape}")
    return x, y

def resize_matrix_to_smaller(A, target_shape):
    return A[:target_shape[0], :target_shape[1]]

def Moor_Penrose_formula(A_t, E_matrix, A, d):
    mult = A @ A_t
    mult1 = d**2 * E_matrix
    mult2 = np.linalg.inv(mult + mult1)
    expr = A_t @ mult2
    return expr

def Moore_Penrose_method(A):
    max_iterations = 100
    epsilon = 1e-4  # tolerance for convergence
    step = 0.1      # learning rate for gradient descent
    rows, columns = A.shape
        
    # Initialize pseudo-inverse with small random values within a limited range
    A_ps_inv = Moor_Penrose_formula(transposed_matrix(A), np.eye(rows), A, step)
    
    for iteration in range(max_iterations):
        # Compute inaccuracy: the difference between A and its reconstruction
        inaccuracy = (A @ A_ps_inv @ A) - A
        loss = np.linalg.norm(inaccuracy, np.inf)
        
        if loss < epsilon:
            print(f"Converged in {iteration} iterations")
            break
        
        # Compute the gradient for update
        gradient = Moor_Penrose_formula(transposed_matrix(A), np.eye(rows), A, step) # (A @ A_ps_inv - np.eye(rows)) @ A
        
        # Update the pseudo-inverse with the gradient (transpose to match shapes)
        A_ps_inv -= step * gradient

    return A_ps_inv
    #return np.linalg.pinv(A)

# Turning matrix into shades spectrum
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
    Z_Xt = E_matrix - X @ X_ps_inv
    Z_t_Xt = transposed_matrix(Z_Xt)
    V = np.zeros((Y.shape[0], X.shape[0]), dtype=float)
    A = (Y @ X_ps_inv) + (V @ Z_t_Xt)
    return A

def model_by_Moore_Penrose_gradient(X, Y):
    # Use the numpy pseudo-inverse method for now
    X_ps_inv = Moore_Penrose_method(X)
    
    E = np.eye(X.shape[0]) 
    
    A = find_A_model_MP(X, Y, X_ps_inv, E)
    Y_img = A @ X
    
    # Transform the matrix back into an image
    Yimage_projected_MP = project_matrix_to_range(Y_img)
    print(Yimage_projected_MP)
    Yimage_projected_MP = 255 - Yimage_projected_MP
    
    # Display the resulting image
    cv2.imshow("Transformed Image", Yimage_projected_MP)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    X, Y = read_img()
    model_by_Moore_Penrose_gradient(X, Y)
