from __future__ import division
import numpy as np
import string
from PIL import Image, ImageFont, ImageDraw, ImageEnhance, ImageFilter
from scipy import io
import matplotlib.pyplot as plt
from StringIO import StringIO
from tempfile import NamedTemporaryFile
import pickle 
import glob

# Preprocess images
# Downsize to 250x250 pixels
# Add text at various locations
# Add labels to images.

def process_img(backgr_img, txt_img, final_img):
	# Resize img, add text and save labels
	ww = 128
	size = (ww, ww)
	img = Image.open(backgr_img).convert("RGBA")
	img_txt = Image.open(txt_img).convert("RGBA")
	width, height = img_txt.size
	width = int(width)
	height = int(height)
	width_half = int(np.rint(width/2))
	height_half = int(np.rint(height/2))
	img_txt = img_txt.resize((width_half, height_half))
	ww = int(128)
	ratio = max((width/ww), (height/ww))
	if ratio < 1:
		size_2 = [width_half, height_half]
	else:
		size_2 = [width/(1.2*ratio), height/(ratio*1.2)] # Then text is larger than image, need to downsize
		img_txt = img_txt.resize((int(np.rint(size_2[0])), int(np.rint(size_2[1]))))

	# Text is rotated inside boxes sometimes
	c_x = np.random.randint(low=0, high=ww-size_2[0])
	c_y = np.random.randint(low=0, high=ww-size_2[1])
	# Needs to take into account the scaling
	c_w = np.rint(size_2[0]) 
	c_h = np.rint(size_2[1]) 

	# put button on source image in position (0, 0)
	img.paste(img_txt, (c_x,c_y), img_txt)
	# The box argument is either a 2-tuple giving the upper left corner
	# This means that the c_x and c_y are upper left corner and not lower left corner

	img = img.filter(ImageFilter.GaussianBlur(1))

	# Labels as prob and bounding boxes
	labels_coord = [1, c_x, c_y, c_w, c_h] # cp, cx,cy,cw,ch

	save_loc = "log/img/directory"
	get_name = backgr_img.split("/")[-1]
	fn = save_loc + final_img  # saved with same name as txt img to allow finding match with word later

	img.save(fn)

	return labels_coord

def resize_img(img, loc):
	ww = 128
	size = (ww, ww)
	new_img = img.split("/")[-1]
	fn = loc + new_img

	img = Image.open(img)
	img = img.resize(size)
	img.save(fn)

def alter_img(backgr_img):
	# flips img
	img = Image.open(backgr_img).transpose(Image.FLIP_LEFT_RIGHT)
	#return img
	fn = backgr_img[:-4]+"processed"+".jpg"
	img.save(fn)

def alter_more_img(backgr_img):
	# vertical flip
	img = Image.open(backgr_img).transpose(Image.ROTATE_180)
	#return img
	fn = backgr_img[:-4]+"doubleprocessed"+".jpg"
	img.save(fn)

def save_obj(obj, name ):
	save_dir = "path/to/dir/for/final/img"
	with open(save_dir + name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


file_name = "path/to/IIIT5K/trainCharBound.mat"
characters = io.loadmat(file_name)
file_name2 = "path/to/IIIT5K/testCharBound.mat"
characters2 = io.loadmat(file_name2)
labels = {}
# filename is key and text value
for el in characters["trainCharBound"]:
	for e in el:
		filename = e[0][0].split("/")[1]
		text = e[1][0]
		labels[filename] = [text]

for el in characters2["testCharBound"]:
	for e in el:
		filename = e[0][0].split("/")[1]
		text = e[1][0]
		labels[filename] = [text]

train_images = "path/to/ImgVanilla" # images to paste text upon
train_images_resized = "path/to/ImgDownsized/" # save downsized img for further processing
text_images = "/home/maxnihr/Documents/Documents/Master2/JapanExchange/Project/IIIT5K/train"

file_names_backgr_img_original = []
file_names_backgr_img = []

# Get file names
for filename in glob.glob(train_images+'/*.jpg'):
    file_names_backgr_img_original.append(filename)

# Resize images and save them for further processing
t = 0
for image in file_names_backgr_img_original:
	resize_img(image, train_images_resized)
	t += 1

# Get new names for re-sized img
for filename in glob.glob(train_images_resized+'/*.jpg'):
    #im=#Image.open(filename)
    file_names_backgr_img.append(filename)

# Create more background images by modifying original img, not needed if you've already got a large number of images
for image in file_names_backgr_img:
	alter_img(image)

# Create even more images!
more_files = []
for filename in glob.glob(train_images_resized+'/*.jpg'):
    more_files.append(filename)

for image in more_files:
	alter_more_img(image)

# Add text to images

# read in updated list of images
file_names_backgr = []
for filename in glob.glob(train_images_resized+'/*.jpg'):
    file_names_backgr.append(filename)

file_names_text_img = []
# Read in text images
for filename in glob.glob(text_images+'/*.png'):
    file_names_text_img.append(filename)

# Save labels
for im, txt in zip(file_names_backgr,file_names_text_img):
	txt_key = txt.split("/")[-1]
	labels[txt_key].append(process_img(im, txt, txt_key))

save_obj(labels, "labels")
