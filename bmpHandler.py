import os
import numpy as np
from PIL import Image

#takes an input bmp  file and converts down to a 28x28 bmp image file
def truncate(img_file_address, end_address):
    img = Image.open(img_file_address).resize((28, 28)).save(end_address)
    return 

    
#takes a truncated bmp file and converts to a numpy array
def imgtoarr(img_file_address):
    img = Image.open(img_file_address)

    pixels = list(img.getdata(0))
    twoDImages = []
    twoDPixels = []
    index = 0
    
    for i in range(28):
        row = []
        for j in range(28):
            row.append(pixels[index])
            index += 1
        twoDPixels.append(row)

    twoDImages.append(twoDPixels)
    arr = np.array(twoDImages)

    return arr

imgtoarr("truncated.bmp")