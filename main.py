import numpy as np
import cv2
import time
#SVD 
from svd import model_by_SVD
#greville
from greville import model_by_Greville
#Moore-Penrose with dichotomy
from moore_penrose_dichotomy import model_by_Moore_Penrose_dichotomy
#Moore-Penrose with gradient
from moore_penrose_gradient import model_by_Moore_Penrose_gradient


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
    model_function(X, Y)
    end_time = time.time()  #end time
    elapsed_time = end_time - start_time
    print(f"Execution time for {model_name}: {elapsed_time:.4f} seconds")


if __name__ == "__main__":
    X, Y = read_img()
    measure_time(model_by_Moore_Penrose_dichotomy, X, Y, "Moore-Penrose Dichotomy")
    measure_time(model_by_Moore_Penrose_gradient, X, Y, "Moore-Penrose Gradient")
    measure_time(model_by_Greville, X, Y, "Greville")
    measure_time(model_by_SVD, X, Y, "SVD")