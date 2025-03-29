import cv2
import numpy as np
from skimage import restoration
from skimage import color
from skimage import exposure
from skimage import img_as_ubyte
from skimage import img_as_float

def reduce_noise(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

def remove_scratches(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5,5), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    binary = cv2.erode(binary, kernel, iterations=1)
    return cv2.inpaint(image, binary, 3, cv2.INPAINT_TELEA)

def deblur_image(image):
    return restoration.unsupervised_wiener(image, np.ones((5, 5)))[0]

def correct_color(image):
    image = img_as_float(image)
    image = color.rgb2lab(image)
    image[:,:,0] = exposure.equalize_hist(image[:,:,0])
    return img_as_ubyte(color.lab2rgb(image))

def upscale_image(image, scale=4):
    return cv2.resize(image, (image.shape[1] * scale, image.shape[0] * scale), interpolation=cv2.INTER_CUBIC)

def enhance_image(image_path):
    image = cv2.imread(image_path)
    image = reduce_noise(image)
    image = remove_scratches(image)
    image = deblur_image(image)
    image = correct_color(image)
    image = upscale_image(image)
    return image
