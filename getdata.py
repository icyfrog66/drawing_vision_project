# Creates data directory and subdirectories for each category.
# Each category contains 1000 images

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
	directory_name = "data/" + str(category)
	if not os.path.exists(directory_name):
		os.makedirs(directory_name)
	for i in range(samples.drawing_count):
		drawing_i = samples.get_drawing(i)
		drawing_i.image.save(directory_name + "/pic" + str(i) + ".png")

# Quickdraw api
# https://quickdraw.readthedocs.io/en/latest/api.html
