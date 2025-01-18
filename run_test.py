import joblib
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt


x_test = imread("C:/Users/Administrator/Desktop/lung/op.jpg")


print("Original Image Shape:", x_test.shape)


if len(x_test.shape) == 3:  
    x_test = rgb2gray(x_test)  
    print("Converted to Grayscale. Shape:", x_test.shape)
elif len(x_test.shape) == 2:  
    print("Image is already Grayscale.")


x_test = resize(x_test, (96, 128), anti_aliasing=True)


print("Resized Image Shape:", x_test.shape)


x_test_flattened = x_test.flatten().reshape(1, -1)


x_test_flattened = x_test_flattened / 255.0


loaded_model = joblib.load('C:/Users/Administrator/Desktop/lung/lung_disease_model.pkl')


prediction = loaded_model.predict(x_test_flattened)


classes = {0: "Normal Lung", 1: "Diseased Lung"}
print("Prediction:", classes[prediction[0]])


plt.imshow(x_test)
plt.title(f"Prediction: {classes[prediction[0]]}")
plt.axis('off')  # Hide axis
plt.show()
