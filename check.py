import numpy as np

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