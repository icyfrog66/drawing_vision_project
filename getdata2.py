# Creates data directory and train and val subdirectories for training and validation respectively.
# train contains 600 images per category (i.e. per subdirectory)
# val contains 400 images per category (i.e. per subdirectory)

from quickdraw import QuickDrawDataGroup
import PIL
import numpy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# Change these categories to download more categories
desired_categories = ["aircraft carrier", "angel", "birthday cake", "car", "hand", "leg", "purse", "shoe"]
for category in desired_categories:
	samples = QuickDrawDataGroup(category)

	train_directory_name = "data/train/" + str(category)
	val_directory_name = "data/val/" + str(category)

	if not os.path.exists(train_directory_name):
		os.makedirs(train_directory_name)
	if not os.path.exists(val_directory_name):
		os.makedirs(val_directory_name)

	i = 0
	for sample in samples.drawings:
		if i >= 600:
			sample.image.save(val_directory_name + "/pic" + str(i - 600) + ".png")
		else:
			sample.image.save(train_directory_name + "/pic" + str(i) + ".png")
		i = i + 1

# Quickdraw api
# https://quickdraw.readthedocs.io/en/latest/api.html
