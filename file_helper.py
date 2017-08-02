import sys
import os
from os import listdir
from os.path import isfile, join

class DisparityAndImagesPath:
	def __init__(self):
		self.imageLeftFilename = None
		self.imageRightFilename = None
		self.disparityLeftFilename = None
		self.disparityRightFilename = None
	def isValid(self):
		return (self.imageLeftFilename is not None and
			self.imageRightFilename is not None and
			self.disparityLeftFilename is not None and
			self.disparityRightFilename is not None)

def get_filenames(directory):
	all_files = []
	for root, dirs, files in os.walk(directory):
		replacedRoot = root.replace(directory,"")
		for file in files:
			all_files.append(os.path.join(replacedRoot,file))
	return all_files

def get_imgs_and_disps(img_dir, disp_dir):
	img_files = get_filenames(img_dir)
	disp_files = get_filenames(disp_dir)
	print("Num img files was " + str(len(img_files)))
	print("Num disp files was " + str(len(disp_files)))
	dataset = {}
	for filename in img_files:
		tokens = filename.split("\\")
		key = ""
		#ignore the last two tokens
		for token_idx in range(len(tokens) - 2):
			key += tokens[token_idx] + "\\"
		key += tokens[len(tokens)-1].replace(".webp","")
		if key not in dataset.keys():
			dataset[key] = DisparityAndImagesPath()
		item = dataset[key]
		if "right" in tokens[len(tokens) - 2]:
			item.imageRightFilename = os.path.join(img_dir, filename)
		else:
			item.imageLeftFilename = os.path.join(img_dir, filename)
	for filename in disp_files:
		tokens = filename.split("\\")
		key = ""
		#ignore the last two tokens
		for token_idx in range(len(tokens) - 2):
			key += tokens[token_idx] + "\\"
		key += tokens[len(tokens)-1].replace(".pfm","")
		if key not in dataset.keys():
			dataset[key] = DisparityAndImagesPath()
		item = dataset[key]
		if "right" in tokens[len(tokens) - 2]:
			item.disparityRightFilename = os.path.join(disp_dir, filename)
		else:
			item.disparityLeftFilename = os.path.join(disp_dir, filename)
	valid_dataset = []
	for item in dataset.values():
		if item.isValid():
			valid_dataset.append(item)
	print("Number of valid data items " + str(len(valid_dataset)))
	assert(len(valid_dataset) == len(img_files)/2)
	assert(len(valid_dataset) == len(disp_files)/2)
	return valid_dataset