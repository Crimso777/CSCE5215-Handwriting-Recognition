import pandas as pd
import csv
from bs4 import BeautifulSoup as bs
import numpy as np
import cv2
import os
from os import listdir
from skimage.filters import threshold_otsu

from PIL import Image, ImageOps
import cv2

def process(image):
   rgb = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
   blur = cv2.GaussianBlur(image,(9,9),3)

   thresh = threshold_otsu(blur)
   img_otsu  = image < thresh
   filtered = filter_image(rgb, img_otsu)
   gray2 = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
   return gray2


def resize(image, desired_size):

   im = image
   old_size = im.shape[:2] # old_size is in (height, width) format
   ratio = float(desired_size)/max(old_size)
   new_size = tuple([int(x*ratio) for x in old_size])

# new_size should be in (width, height) format
   im = cv2.resize(im, (new_size[1], new_size[0])) 

   delta_w = desired_size - new_size[1]
   delta_h = desired_size - new_size[0]
   top, bottom = delta_h//2, delta_h-(delta_h//2)
   left, right = delta_w//2, delta_w-(delta_w//2)

   color = 0
   new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
       value=color)

   return new_im

def filter_image(image, mask):

    r = image[:,:,0] * mask
    g = image[:,:,1] * mask
    b = image[:,:,2] * mask

    return np.dstack([r,g,b])

def crop_proj(proj):
   left = 0
   right = len(proj)-1
   while proj[left] == 0 and left < len(proj):
      left = left + 1
   if left == len(proj):
      return []
   while proj[right] == 0:
      right = right - 1
   return proj[left:right]
# Create sort_contours() function to grab the contour of each digit from left to right
def sort_contours(cnts,reverse = False):
    if len(cnts) == 0:
        return cnts
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts
words = {}
with open("C:/Users/J54JWG3/Documents/Github/CSCE5215-Handwriting-Recognition/Datasets/Handwritten Words and Sentences/ASCII/words.txt") as file: 
   lines = file.readlines()
   print(len(lines))
   for line in lines:
      entry = line.split(' ')
      words[entry[0]] = entry[-1][:-1]
images = []
# get the path/directory
folder_dir = "C:/Users/J54JWG3/Documents/Github/CSCE5215-Handwriting-Recognition/Datasets/print words"
for doc in os.listdir(folder_dir):
   doc_dir = os.path.join(folder_dir, doc)
   if os.path.isdir(doc_dir):
      for image in os.listdir(doc_dir):
          image_path = os.path.join(doc_dir, image)
          images.append(image_path)
print(len(images))
segments =  []
i = 0
failures = 0
distance = 0
low_counts = 0
for path in images:
   word = words[os.path.basename(path).split('.')[0]]
   if not word.isalpha():
      continue
   image = cv2.imread(path)
# Scales, calculates absolute values, and converts the result to 8-bit.
#   img = cv2.convertScaleAbs(image, alpha=(255.0))
    
    # convert to grayscale and blur the image
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   blur = cv2.GaussianBlur(gray,(9,9),3)

   thresh = threshold_otsu(blur)
   img_otsu  = gray < thresh
   filtered = filter_image(image, img_otsu)
   gray2 = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
#   print(filtered.shape)
    
    # Applied inversed thresh_binary 
   binary = cv2.threshold(gray,  180, 255,
                       cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    ## Applied dilation 
   kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
   thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
   cont, _  = cv2.findContours(gray2 , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   proj_count = 0

   test_roi = image.copy()

# Initialize a list which will be used to append charater image
   crop_characters = []

# define standard width and height of character
   digit_w, digit_h = 28, 28
#   print(cont)
   for c in cont:
      (x, y, w, h) = cv2.boundingRect(c)
      ratio = h/w

               # Sperate number and gibe prediction
      curr_num = gray2[y:y+h,x:x+w]
#      curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
#      _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
      curr_num = resize(curr_num, 28)
      crop_characters.append(curr_num)

#   skewed= cv2.resize(binary , dsize=(100, binary.shape[0]))
   skewed = gray
#   print(skewed.shape)
#   print(binary.shape)
   vertical_projection = np.sum(skewed, axis = 0)
#   vertical_projection = crop_proj(vertical_projection)
   vertical_projection = np.append(vertical_projection, 0)
   sum = 0
   for i1 in range(len(vertical_projection)):
      sum = sum + vertical_projection[i1]
   average = sum / len(vertical_projection)
   for i1 in range(len(vertical_projection)):
      if vertical_projection[i1] < average*.2:
         vertical_projection[i1] = 0
    #      print(vertical_projection[i1])
   last = 0
   length = 0
   start = 0
   candidates = []
   for i1 in range(len(vertical_projection)):
      if last == 0 and vertical_projection[i1] > 0:      

#         if length == 4:
            proj_count = proj_count + 1
            last = vertical_projection[i1]
            start = i1
#         length = length + 1
      else:
         if last > 0 and vertical_projection[i1] == 0:      
            candidates.append((start, i1))
         last = vertical_projection[i1]
         length = 0
   cropped = []
#   print(skewed.shape)
   for seg in candidates:
#      print(seg)
      cropped.append(skewed[:, range(seg[0], seg[1])])
#   image_number = len(candidates)
   image_number = len(crop_characters)
   if len(crop_characters) == len(word):
      cv2.imshow('image', cv2.resize(crop_characters[0] , dsize=(100, 100)))
      cv2.waitKey(0)
      cv2.destroyAllWindows()
      segments.append([np.array(crop_characters), word])
      i = i + 1

#   image_number = len(filtered)
#   print(image_number)
#   if proj_count != 6:
   if image_number != len(word):
      if len(word)> image_number:
         low_counts = low_counts + 1
#      print(image_number, ", ", word, " ", len(word), " ", len(crop_characters))
      failures = failures + 1
      distance = distance + abs( image_number - len(word))


print(failures)
print(distance)
print(low_counts)
print(len(segments))
print(segments[0][0][0].shape)
print(type(segments[0][0][0]))
print(segments[0][0].shape)
