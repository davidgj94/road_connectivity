from pathlib import Path
import os.path
import os
import argparse
import numpy as np
import re
import pdb

cities = ['Vegas', 'Shanghai']

def filter_img_list(img_list, city):
	new_img_list = []
	img_numbers = []
	for img_name in img_list:
		if city in img_name:
			new_img_list.append(img_name)
			img_numbers.append(int(re.search(r"img(\d+)", img_name)[0].replace('img', '')))
	return new_img_list, img_numbers


def make_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--list_dir', type=str, required=True)
	parser.add_argument('--dataset_dir', type=str, required=True)
	return parser

if __name__ == "__main__":

	args = make_parser().parse_args()

	txt_path_template = os.path.join(args.list_dir, '{}-old.txt')

	gt_dir = os.path.join(args.dataset_dir, 'gt')
	gt_img_list = [os.path.basename(str(glob)).split('.')[0] for glob in Path(gt_dir).glob("*.png")]

	for part in ['train', 'val']:
		txt_path = txt_path_template.format(part)
		img_list = np.loadtxt(txt_path, dtype=str).tolist()
		new_img_list = []
		for city in cities:
			gt_city_list, gt_city_numbers = filter_img_list(gt_img_list, city)
			_, img_city_numbers = filter_img_list(img_list, city)
			for idx, number in enumerate(gt_city_numbers):
				if number in img_city_numbers:
					new_img_list.append(gt_city_list[idx])

		new_txt_path = txt_path.replace('-old', '')
		if os.path.exists(new_txt_path):
			os.remove(new_txt_path)
		np.savetxt(new_txt_path, np.array(new_img_list, dtype=str), fmt='%s')
	
	img_list_train = np.loadtxt(txt_path_template.format('train').replace('-old', ''), dtype=str).tolist()
	img_list_val = np.loadtxt(txt_path_template.format('val').replace('-old', ''), dtype=str).tolist()

	print("{} images train".format(len(img_list_train)))
	print("{} images val".format(len(img_list_val)))
	if len(set(img_list_val) & set(img_list_train)) > 0:
		print("DANGER: VAL IMAGES IN TRAIN PARTITION!!!!!")