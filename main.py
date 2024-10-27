import numpy as np
import cv2
import time
import prettytable as pt
from check import compute_silhouette_score, compute_image_metrics
#SVD 
from svd import model_by_SVD
#greville
from greville import model_by_Greville
#Moore-Penrose with dichotomy
from moore_penrose_dichotomy import model_by_Moore_Penrose_dichotomy
#Moore-Penrose with gradient
from moore_penrose_gradient import model_by_Moore_Penrose_gradient


#порівняти вхідний У і отримані через siluate score

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

def measure_time(model_function, X, Y, model_name):
    start_time = time.time()  #start time
    c1, c2, c3, c4, Y_img = model_function(X, Y)
    end_time = time.time()  #end time
    elapsed_time = end_time - start_time
    print(f"Execution time for {model_name}: {elapsed_time:.4f} seconds")
    mse, ssim = compute_image_metrics(Y_img, Y)
    silhouette = compute_silhouette_score(Y_img, Y)
    return elapsed_time, c1, c2, c3, c4, silhouette, mse, ssim


if __name__ == "__main__":
    X, Y = read_img()
    mpd_t, mpd_c1, mpd_c2, mpd_c3, mpd_c4, silhouette_mpd, mse_mpd, ssim_mpd = measure_time(model_by_Moore_Penrose_dichotomy, X, Y, "Moore-Penrose Dichotomy")
    mpg_t, mpg_c1, mpg_c2, mpg_c3, mpg_c4, silhouette_mpg, mse_mpg, ssim_mpg = measure_time(model_by_Moore_Penrose_gradient, X, Y, "Moore-Penrose Gradient")
    g_t, g_c1, g_c2, g_c3, g_c4, silhouette_g, mse_g, ssim_g = measure_time(model_by_Greville, X, Y, "Greville")
    svd_t, svd_c1, svd_c2, svd_c3, svd_c4, silhouette_svd, mse_svd, ssim_svd = measure_time(model_by_SVD, X, Y, "SVD")

    results = []

    results.append(("Moore-Penrose Dichotomy", mpd_t, silhouette_mpd, mse_mpd, ssim_mpd, mpd_c1, mpd_c2, mpd_c3, mpd_c4))
    results.append(("Moore-Penrose Gradient", mpg_t, silhouette_mpg, mse_mpg, ssim_mpg, mpg_c1, mpg_c2, mpg_c3, mpg_c4))
    results.append(("Greville", g_t, silhouette_g, mse_g, ssim_g, g_c1, g_c2, g_c3, g_c4))
    results.append(("SVD", svd_t, silhouette_svd, mse_svd, ssim_svd, svd_c1, svd_c2, svd_c3, svd_c4))

    # Display the results in a table
    table = pt.PrettyTable()
    table.field_names = ["Method", "Execution Time (seconds)", "Silhouette comparison", "MSE comparison", "SSIM comparison", "A * A^+ * A = A", "A^+ * A * A^+ = A^+", "A^+ A is symmetric", "A A^+ is symmetric"]

    # Add rows for each method
    for method_name, time_taken, silhouette_check,mse, ssim, cond1, cond2, cond3, cond4 in results:
        table.add_row([method_name, f"{time_taken:.4f}", f"{silhouette_check:.4f}", f"{mse:.4f}", f"{ssim:.4f}", cond1, cond2, cond3, cond4])

    print(table)