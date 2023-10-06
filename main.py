from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import cv2


img_path = './sample_images/normal_test4.jpg' #Put image path here or you can either use your camera as src


img = cv2.imread(img_path)
img = cv2.resize(img, (700,700))


# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
image = Image.open(img_path).convert("RGB")

# resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# turn the image into a numpy array
image_array = np.asarray(image)

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
data[0] = normalized_image_array

# Predicts the model
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

# Print prediction and confidence score
print("Class:", class_name[2:], end="")
print("Confidence Score:", confidence_score)

img = cv2.putText(img,f"{class_name[2:]}", (10,50), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255)) 
img = cv2.putText(img,f"Accuracy: {int(confidence_score*100)}%", (10,80), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255)) 
img = cv2.putText(img,"Created BY NAMAN", (10,690), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255)) 


while True:
    cv2.imshow("MRI scan Tumor Detector by NAMAN", img)
    keyboard_input = cv2.waitKey(1)
    if keyboard_input == 27: #press esc to exit
        cv2.destroyAllWindows()
        break