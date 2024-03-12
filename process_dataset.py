import os, sys
import numpy as np
import pandas as pd
import csv
import argparse

def parse_arguments(args):

	parser = argparse.ArgumentParser()
	# parser.add_argument('--dataset_dir', type=str, default="../dataset/EXR_images")
	# parser.add_argument('--augments', type=int, default=3)
	# parser.add_argument('--out_dir', type=str, default="../dataset/images")
	parser.add_argument('--new_csv', type=str, default="../dataset/dataset.csv")
	parser.add_argument('--old_csv', type=str, default="../dataset/old_dataset.csv")
	# parser.add_argument('--env_dir', type=str, default="../dataset/env_maps")
	# parser.add_argument('--edge_dir', type=str, default="../dataset/edge_maps")
	# parser.add_argument('--ldr_dir', type=str, default="../dataset/ldr_images")
	# parser.add_argument('--train_split', type=int, default=80)
	# parser.add_argument('--val_split', type=int, default=10)


	return parser.parse_known_args()


# def process_dataset(dataset_dir, csv_file, num_augs):
# 	#get list of images in dataset_dir
# 	img_list = get_file_list(dataset_dir)
# 	#create csv file
# 	with open(csv_file, 'w') as csvDataFile:
# 		writer = csv.writer(csvDataFile)
# 		header = ["filename"]
# 		writer.writerow(header)
# 		#cycle thorugh img_list
# 		for img_file in img_list:
# 			#read image
# 			#hdr = im.imread(img_file, 'EXR-FI')
# 			hdr = np.asarray(Image.open(img_file))
# 			#resize
# 			#hdr = cv2.resize(hdr, (512,256), interpolation=cv2.INTER_CUBIC)
# 			#save to CSV file
# 			img_filename, img_extension = os.path.splitext(os.path.basename(img_file))
# 			writer.writerow([str(img_filename)])
# 			#save image
# 			if num_augs > 0:
# 				#augment by rotating
# 				completed = []
# 				for i in range(num_augs):
# 					#add random augmentations
# 					check = False
# 					while check == False:
# 						# add random vertical flips
# 						r = random.randrange(0,5,1) #20% chance of vertically flipping
# 						#add random rotations
# 						rot = random.randrange(60,300,60)	#random roation in degrees between 30 and 330
# 						if rot not in completed:
# 							completed.append(rot)
# 							check = True

# 					img_aug = rotate_equirect(hdr,(rot,0,0))
# 					if r == 1:
# 						img_aug = cv2.flip(img_aug, 1)
# 						print("{}: rotation {} flip: Vertical".format(i,rot))
# 						img_out = "{}_{}_V".format(img_filename,rot)
# 					else:
# 						print("{}: rotation {}".format(i,rot))
# 						img_out = "{}_{}".format(img_filename,rot)

# 					writer.writerow([str(img_out)])

# 		#save df to csv file
# 		csvDataFile.close()

def in_out_door(old_csv,new_csv):
	old_df = pd.read_csv(old_csv)
	img_list = old_df['filename'].tolist()
	#create csv file
	with open(new_csv, 'w') as csvDataFile:
		writer = csv.writer(csvDataFile)
		header = ["filename","label"]
		writer.writerow(header)
		#cycle thorugh img_list
		for img_file in img_list:
			if "scene" in img_file:
				label = 1
			else:
				label = 0
			writer.writerow([str(img_file),label])

		#save df to csv file
		csvDataFile.close()

if __name__ == "__main__":
	args, _ = parse_arguments(sys.argv)

	in_out_door(args.old_csv,args.new_csv)