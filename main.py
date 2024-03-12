import os, sys
from os.path import join
import numpy as np
import imageio as im
import lpips
import argparse
import yaml
import matplotlib.pyplot as plt
from time import time
#pytorch
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset,DataLoader,TensorDataset
from torch.autograd import Variable
from torchvision import transforms
from torchvision.transforms import ToTensor, Normalize, Resize, CenterCrop
from torchvision.transforms.functional import InterpolationMode
#in repo
from models.build4 import build_model
from utils.spherical_lpips import LPIPS as SP_LPIPS
from utils.utils import tonemapping
from models.Discriminator_ml import MsImageDis
from models.env_map_net import EnvModel
from dataset import gamma_correct_torch, dataset_ldr2hdr
from utils.loss import cosine_blur
#metrics
from pytorch_msssim import ssim
from utils.metrics import PSNR, my_FID, angular_error

from blip_models.blip import blip_decoder

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

"""
	To do:
"""


class sceneDetection():
	def __init__(self,args,config):
		#set vairables
		self.args = args
		self.epochs = args.epochs
		self.path = args.path
		self.loss_func = args.loss_func

		#Initialise model
		self.gen = build_model(config).cuda()

		# Load pre-trained weight
		self.start_epoch = 0
		if args.load_pretrain and os.path.exists(args.save_weight_dir+"Gen_former"):
			print('Loading model weight...')
			#load generator
			data = torch.load(join(args.save_weight_dir, 'Gen_former'))
			self.gen.load_state_dict(data['generator'])
			self.start_epoch = data['epoch']
			print("From Epoch: ", self.start_epoch)
			#load saved losses
			if os.path.exists(args.save_weight_dir+"gen_data.txt"):
				file = open(args.save_weight_dir+"gen_data.txt","rb")
				self.gen_data = np.load(file, allow_pickle=True)
				file.close
		else:
			self.gen_data = np.zeros((self.args.epochs))

		#multi-GPU
		self.device_count = torch.cuda.device_count()
		if self.device_count == 2:
			print("Let's use", torch.cuda.device_count(), "GPUs!")
			self.gen = nn.DataParallel(self.gen)
			workers = self.device_count
		elif self.device_count == 1:
			workers = 0

		#Optimiser
		self.opt_gen = optim.Adam(self.gen.parameters(), lr=args.lr / 2, betas=(0, 0.9), weight_decay=args.weight_decay)
		if self.dsc_type != "None" and args.mode < 2:
			self.opt_dis = optim.Adam(self.dis.parameters(), lr=args.lr * 2, betas=(0, 0.9), weight_decay=args.weight_decay)


		#Load data
		print('Loading data...')
		self.train_data = dataset_sceneDetection(args, root=args.train_data_dir)
		self.train_loader = DataLoader(self.train_data, batch_size=args.train_batch_size*self.device_count, shuffle=True, num_workers=workers)
		print('train data: %d images' % (len(self.train_loader.dataset)))

		if args.validation:
			self.validate_data = dataset_sceneDetection(args, root=args.validate_data_dir)
			self.validate_loader = DataLoader(self.validate_data, batch_size=args.train_batch_size*self.device_count, shuffle=True, num_workers=workers)
			print('validation data: %d images' % (len(self.validate_loader.dataset)))

		if args.mode > 0:
			self.eval_data = dataset_sceneDetection(args,root=args.test_data_dir)
			self.eval_loader = DataLoader(self.eval_data, batch_size=args.train_batch_size*self.device_count, shuffle=False, num_workers=workers)
			print('test data: %d image pairs' % (len(self.eval_loader.dataset)))

		if args.mode < 2:
			self.setup_loss_functions()

	def save(self,epoch,loss_graph):
		self.gen.eval()

		torch.save({"generator": self.gen.state_dict(),
					"epoch": epoch
			}, join(self.args.save_weight_dir, 'Gen'))

		file = open(self.args.save_weight_dir+"gen_data.txt","wb")
		np.save(file,loss_graph)
		file.close

		if self.dsc_type != "None" and args.mode < 2:
			torch.save(self.dis.state_dict(), join(self.args.save_weight_dir, 'Dis'))
			self.dis.eval()

	def setup_loss_functions(self):
		## Define Loss functions
		self.L2_loss = nn.MSELoss().cuda()
		self.mae = nn.L1Loss().cuda()
		self.BCE = nn.BCELoss().cuda()

	def calc_losses(self, pred, gt):
		losses = np.zeros((6))

		#Pixel Reconstruction Loss
		if self.loss_func == "L1":
			pixel_rec_loss = self.mae(pred,gt)
		elif self.L2_loss == "L2":
			pixel_rec_loss = self.L2_loss(pred,gt)
		else:
			pixel_rec_loss = self.BECLoss(pred, gt)

		pixel_rec_loss = pixel_rec_loss * self.args.PR_lw

		self.opt_gen.zero_grad()
		pixel_rec_loss.backward()
		self.opt_gen.step()

		return pixel_rec_loss

	def print_images(self, main_dir, image, num=0):
		print_dir = join(self.args.save_dir, main_dir)
		os.makedirs(print_dir, exist_ok=True)

		#normalise images
		ones = torch.ones_like(I_pred)
		image = gamma_correct_torch(image+ones,gamma=self.gamma, alpha=self.alpha,inverse=1)

		if num == 0:
			num = image.shape[0]
		for i in range(num):
			image = image[i].permute((1,2,0)).detach().cpu().numpy()

			pred_ldr = tonemapping(image)

			if i == 0:
				pre_hist, bin_edges = np.histogram(pred_ldr, bins=30,range=(-1,2))
				plt.hist(bin_edges[:-1], bin_edges, weights=pre_hist)
				plt.savefig(join(print_dir,f'{i}_hist.png'))

			im.imwrite(join(print_dir, f'{name}.png'), np.clip(pred_ldr,0,1))


	def plot_train_graph(self,train_loss,val_loss=None):
		e = range(0,self.args.epochs,1)
		fig, ax = plt.subplots()
		ax.set_xlabel("Epoch")
		ax.set_ylabel("Loss")

		ax.plot(e,train_loss, color='r',label="Train Loss")
		if self.args.validation:
		    ax.plot(e,val_loss, color='blue',linestyle='-',label="Val Loss")

		ax.legend(loc=0)
		plt.title("Scene Detection Loss "+ self.args.name)
		plt.savefig(self.args.save_dir + "loss_plot.png")

		plt.close()

	def train(self):
		self.gen.train()
		if self.dsc_type != "None":
			self.dis.train()

		#setup loss logs
		tot_loss = np.zeros((self.args.epochs))
		if self.args.validation:
			val_loss = np.zeros((self.args.epochs))
		if self.args.load_pretrain:
			tot_loss[:self.gen_data.shape[1]] = self.gen_data

		epoch = self.start_epoch
		for epoch in range(self.start_epoch,self.epochs):
			print(f"----Start training[{epoch+1}]----")

			for batch_idx, (image, label, name) in enumerate(self.train_loader):

				pred_label = self.gen(image)

				tot_loss[epoch] = self.calc_losses(pred_label, label)

				# #write first few files to output
				# if epoch == (self.args.epochs-1) and (batch_idx == 0):
				# 	self.print_images('train',pred_ldr,6)

			print_loss = tot_loss[:,epoch] / len(self.train_loader)
			print(f"Epoch {epoch}/{self.args.epochs}:Loss: {print_loss[2]}")

			self.save(epoch, tot_loss)
			self.gen.train()
			if self.dsc_type != "None":
				self.dis.train()

			#run validation set
			if self.args.validation:
				all_val_loss = self.validate(epoch)
				val_loss[epoch] = all_val_loss
				self.gen.train()
				if self.dsc_type != "None":
					self.dis.train()

		self.save(epoch, tot_loss)
		tot_loss /= len(self.train_loader)

		## plot graph
		self.plot_train_graph(tot_loss,val_loss if self.args.validation else None)


	def validate(self,epoch=0):
		self.gen.eval()
		if self.dsc_type != "None":
			self.dis.eval()

		for batch_idx, (image, label, name) in enumerate(self.validate_loader):

			pred_label = self.gen(image)

			##Compute losses
			tot_loss = calc_losses(pred_label, label)

			# #write first few files to output
			# if epoch == (self.args.epochs-1) and (batch_idx == 0):
			# 	self.print_images('val',pred_ldr,6)

		tot_loss /= len(self.validate_loader)
		print(f"Validation set: Loss: {tot_loss}")

		return tot_loss

	def eval(self):
		print("Testing...")
		self.gen.eval()

		accuracy

		eve = eval_metrics()

		com_total = 0
		for batch_idx, (image, label, name) in enumerate(self.eval_loader):
			with torch.no_grad():
				t_start = time()

				pred_labels = self.gen(image)

				comsum = time() - t_start
				com_total += comsum
				acc_labels = torch.where(pred_labels < 0.5,0.0,1.0)
				accuracy += torch.where(pred_labels==label,1.0,0.0).mean()

		accuracy /= len(self.eval_loader)
		avg = com_total / len(self.eval_loader)
		print(f"Average Inference Time: {avg}s")
		print(f"Accuracy: {accuracy}")

	def print_EXR(self,loader):
		self.gen.eval()

		os.makedirs(join(self.args.save_dir, 'GT_EXRs'), exist_ok=True)
		os.makedirs(join(self.args.save_dir, 'EXRs'), exist_ok=True)

		com_total = 0
		for batch_idx, (gt, name, ldr_img) in enumerate(loader):
			with torch.no_grad():
				t_start = time()

				I_pred,_,_ = self.gen(ldr_img)


				ones = torch.ones_like(I_pred)
				I_pred = gamma_correct_torch(I_pred+ones,gamma=self.gamma, alpha=self.alpha,inverse=1)
				gt = gamma_correct_torch(gt+ones,gamma=self.gamma, alpha=self.alpha,inverse=1)
				if self.IO:
					I_pred = torch.cat((I_pred[...,int(self.args.width/2):],I_pred[...,:int(self.args.width/2)]),dim=3)
					gt = torch.cat((gt[...,int(self.args.width/2):],gt[...,:int(self.args.width/2)]),dim=3)

				comsum = time() - t_start
				com_total += comsum

			for i in range(gt.size(0)):
				im.imwrite(join(self.args.save_dir, 'EXRs', '%s.exr' % (name[i])), I_pred[i].permute((1,2,0)).detach().cpu().numpy())
				im.imwrite(join(self.args.save_dir, 'GT_EXRs', '%s.exr' % (name[i])), gt[i].permute((1,2,0)).detach().cpu().numpy())


		avg = com_total / len(loader)
		print(f"Average Inference Time: {avg}s")


class eval_metrics():
	def __init__(self):
		self.psnr = PSNR(1).cuda()
		self.L2_loss = nn.MSELoss()
		self.fid = my_FID()
		self.ang_err = angular_error()

	def eval(self, gen_img, gt_img):
		SSIM = torch.mean(ssim(gen_img, gt_img, data_range=gt_img.max()-gt_img.min(), size_average=False))
		psnr = self.psnr(gen_img.permute((0,2,3,1)),gt_img.permute((0,2,3,1)))
		rmse = torch.sqrt(self.L2_loss(gen_img,gt_img))
		fid = self.fid(gen_img,gt_img)
		ae = self.ang_err(gen_img,gt_img)

		return SSIM, psnr, rmse, fid, ae	

if __name__ == '__main__':

	SAVE_WEIGHT_DIR = '/outpainting/checkpoints/'#former_resize_4-3/'
	SAVE_LOG_DIR = '/outpainting/logs_all/'#logs_former_resize_4-3/'
	TRAIN_DATA_DIR = '/dataset/trainset.csv'
	TEST_DATA_DIR = '/dataset/testset.csv' #'/outpainting/data/'
	SAVE_DIR = '/outpainting/save/'

	def get_args():
		parser = argparse.ArgumentParser()
		#general
		parser.add_argument('--path', type=str, default='./', help="Path of directory")
		parser.add_argument('--mode', type=int, default=1, help="0: Train, 1: Train&Eval, 2: Eval")
		parser.add_argument('--name', type=str, default='./', help="Name of test")
		parser.add_argument('--model_type', type=str, default='fixup', help="Type of model")
		parser.add_argument('--load_pretrain', type=int, default=1, help="Load from checkpoint or pretrain")
		#Parameters
		parser.add_argument('--train_batch_size', type=int, help='batch size of training data', default=40)
		parser.add_argument('--test_batch_size', type=int, help='batch size of testing data', default=16)
		parser.add_argument('--validation', type=int, default=0, help="Run validation test set")
		parser.add_argument('--epochs', type=int, help='number of epoches', default=30)
		parser.add_argument('--lr', type=float, help='learning rate', default=2e-4)
		parser.add_argument('--weight_decay', type=float, help='weight decay for L2 regularization', default=1e-4)
		#Network Settings
		parser.add_argument('--height',type=int,default=128,help="Max feature channel size")
		parser.add_argument('--width',type=int,default=256,help="Max feature channel size")

        parser.add_argument('--block_depths', type=str, help='SWIN block depths (must be 4 numbers e.g. 1234', default=3352)
        parser.add_argument('--in_chans', type=int, help='3=just image, 6=w/ Env Map', default=3)
        parser.add_argument('--window_size', type=int, help='Window Size', default=8)
        parser.add_argument('--embed_dim', type=int, help='Number of embedding dimensions', default=96)
        parser.add_argument('--use_checkpoint', type=int, help='Training checkpoint traning', default=0)

		#Loss Functions
		parser.add_argument('--PR_lw',type=float, help='L1 loss weight',default=20)
		parser.add_argument('--loss_func',type=str, help='L1 loss weight',default="BCE")
		#Directories
		parser.add_argument('--load_weight_dir', type=str, help='directory of pretrain model weights',
							default=SAVE_WEIGHT_DIR)
		parser.add_argument('--save_weight_dir', type=str, help='directory of saving model weights',
							default=SAVE_WEIGHT_DIR)
		parser.add_argument('--log_dir', type=str, help='directory of saving logs', default=SAVE_LOG_DIR)
		parser.add_argument('--save_dir', type=str, help='directory of saving results', default=SAVE_DIR)
		parser.add_argument('--train_data_dir', type=str, help='directory of training data', default=TRAIN_DATA_DIR)
		parser.add_argument('--validate_data_dir', type=str, help='directory of validation data', default=TEST_DATA_DIR)
		parser.add_argument('--test_data_dir', type=str, help='directory of testing data 1', default=TEST_DATA_DIR)
		parser.add_argument('--dataset_dir', type=str, help='directory of dataset', default='/dataset/scenery/all')

		opts = parser.parse_args()
		return opts

	args = get_args()
	print("\nArguments:\n")
	for arg in vars(args):
		print(" {} : {} ".format(arg, getattr(args,arg) or ''))

    config = {}
    config['TYPE'] = args.model_type
    config['PATCH_SIZE'] = 4
    config['IN_CHANS'] = args.in_chans
    config['EMBED_DIM'] = args.embed_dim
#    config['SWIN.DEPTHS'] = [2, 2, 6, 2]	#old
    config['DEPTHS'] = [int(args.block_depths[0]), int(args.block_depths[1]), int(args.block_depths[2]), int(args.block_depths[3])]
    config['NUM_HEADS'] = [3, 6, 12, 24]
    config['WINDOW_SIZE'] = args.window_size
    config['MLP_RATIO'] = 4.
    config['QKV_BIAS'] = True
    config['QK_SCALE'] = None
    config['DROP_RATE'] = 0.0
    config['DROP_PATH_RATE'] = 0.2
    config['PATCH_NORM'] = args.patch_norm
    config['TRAIN.USE_CHECKPOINT'] = False


	args.log_dir = args.path + args.log_dir + args.name +"/"
	args.load_weight_dir = args.path + args.load_weight_dir
	args.save_weight_dir = args.path + args.save_weight_dir + args.name +"/"
	args.train_data_dir = args.path + args.train_data_dir
	args.validate_data_dir = args.path + args.validate_data_dir
	args.test_data_dir = args.path + args.test_data_dir
	args.save_dir = args.path + args.save_dir  + "save_"+args.name +"/"

	os.makedirs(args.save_weight_dir, exist_ok=True)
	if args.validation:
		os.makedirs(join(args.save_weight_dir,"val"), exist_ok=True)
	os.makedirs(args.save_dir, exist_ok=True)

	model = sceneDetection(args,config)

	# Train & test the model
	if args.mode < 2:
		model.train()

	# Evaluate
	if args.mode > 0 and args.mode < 3:
		model.eval()

	#print model
	if args.mode == 3:
		#model.print_EXR(model.train_loader)
		#model.print_EXR(model.validate_loader)
		model.print_EXR(model.eval_loader)


	#print saved training graph
	if args.mode == 4:
		model.args.validation = 0
		model.plot_train_graph(model.gen_data)
