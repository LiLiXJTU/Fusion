# Training a NestFuse network
# auto-encoder

import os
import numpy as np
from torch.utils.data import DataLoader

from Load_Dataset import LoadDatasets
from nets.UNet import UNet
from util.save import save_checkpoint
from Loss.DiceBceLoss import WeightedDiceBCE
import sys
import time
from tqdm import tqdm, trange
import scipy.io as scio
import random
import torch
from torch.optim import Adam
from torch.autograd import Variable
import utils
from args_fusion import args
import pytorch_msssim
from nets.model import Model
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
EPSILON = 1e-5


def main():
	img_flag = False
	alpha_list = [700]
	w_all_list = [[6.0, 3.0]]
	train_data, val_data = LoadDatasets(args.train_path,args.val_path)
	train_loader = DataLoader(train_data, args.batch_size, shuffle=False)
	val_loader = DataLoader(val_data, args.batch_size, shuffle=False)
	model = Model()
	# nest_model = NestFuse_light2_nodense(nb_filter, input_nc, output_nc, deepsupervision)
	# model = Fusion_network(nb_filter, f_type)
	#model = UNet(1, 1)
	optimizer = Adam(model.parameters(), args.lr)
	max_dice = 0.0
	for epoch in range(args.epochs):
		print(epoch+1,'/',args.epochs)
		print('start training....')
		model.train(True)
		model.cuda()
		for w_w in w_all_list:
			w1, w2 = w_w
			for alpha in alpha_list:
				train_Fusion(train_loader,optimizer,model,alpha, w1, w2, Train=True)
		# evaluate on validation set
		print('start validation....')
		with torch.no_grad():
			model.eval()
			for w_w in w_all_list:
				w1, w2 = w_w
				for alpha in alpha_list:
					val_loss, val_dice = train_Fusion(val_loader,optimizer,model, alpha, w1, w2,Train = False)
		#       Save best model
		# =============================================================
		if val_dice > max_dice:
			#print(max_dice, val_dice)
			max_dice = val_dice
			save_checkpoint({'epoch': epoch,
						 'best_model': True,
						'model':args.model,
						 'state_dict': model.state_dict(),
						 'val_loss': val_loss,
						 'optimizer': optimizer.state_dict()}, args.model_path)


def train_Fusion(data_loader,optimizer,model,alpha, w1, w2,Train):
	mse_loss = torch.nn.MSELoss()
	ssim_loss = pytorch_msssim.msssim

	dices = []
	Loss_ssim = []
	Loss_all = []
	dice_sum = 0
	loss_sum = 0
	for i, (batch, names) in enumerate(data_loader, 1):
		print([i],'/',[len(data_loader)])
		CT, SUV,label = batch['CT_image'], batch['SUV_image'], batch['label']
		img_CT, img_SUV, label = CT.cuda(),SUV.cuda(), label.cuda()
		# get fusion image
		# encoder
		# 4 layers
		en_CT = model.nest_model.encoder(img_CT)
		en_SUV = model.nest_model.encoder(img_SUV)
		# fusion
		fusion = model.fusion_model(en_CT, en_SUV)
		# decoder
		outputs = model.nest_model.decoder_eval(fusion)
		# resolution loss: between fusion image and visible image
		x_CT = img_CT.clone()
		######################### LOSS FUNCTION #########################
		loss1_value = 0.
		loss2_value = 0.
		for output in outputs:
			#是否需要归一化？
			output = (output - torch.min(output)) / (torch.max(output) - torch.min(output) + EPSILON)
			output = output * 255
			#将CT也归一化？
			x_CT = (x_CT - torch.min(x_CT)) / (torch.max(x_CT) - torch.min(x_CT) + EPSILON)
			x_CT = x_CT * 255
			# ---------------------- LOSS IMAGES ------------------------------------
			# detail loss
			# ssim_loss_temp1 = ssim_loss(output, x_ir, normalize=True)
			#ssim_loss_temp2 = ssim_loss(output, x_SUV, normalize=True)
			#考虑是和CT还是SUV做损失
			ssim_loss_temp2 = ssim_loss(output, x_CT, normalize=True)
			loss1_value += 1 - ssim_loss_temp2

			# feature loss
			w_CT = [w1, w1, w1, w1]
			w_SUV = [w2, w2, w2, w2]
			w_fea = [1, 10, 100, 1000]
			for j in range(4):
				en_CT_layer = en_CT[j]
				en_SUV_layer = en_SUV[j]
				en_fusion_layer = fusion[j]
				loss2_value += w_fea[j]*mse_loss(en_fusion_layer, w_CT[j]*en_CT_layer + w_SUV[j]*en_SUV_layer)
			loss2_value = loss2_value*0.000005
			# seg loss
			preds = model.nest_model.decoder_train(fusion)
			seg_loss = WeightedDiceBCE()(preds, label.float())

		loss1_value /= len(outputs)
		loss2_value /= len(outputs)
		# total loss
		out_loss = loss1_value + loss2_value + seg_loss
		if Train:
			optimizer.zero_grad()
			out_loss.backward()
			optimizer.step()

		loss_sum = loss_sum + len(label) * out_loss
		if i == len(data_loader):
			train_loss_avg = loss_sum / (args.batch_size * (i - 1) + len(label))
		else:
			train_loss_avg = loss_sum / (i * args.batch_size)

		######################### Dice #########################
		train_dice = WeightedDiceBCE()._show_dice(preds, label.float())
		print(train_dice)
		dices.append(train_dice)
		dice_sum = dice_sum + len(label) * train_dice
		if i == len(data_loader):
			train_dice_avg = dice_sum / (args.batch_size * (i - 1) + len(label))
		else:
			train_dice_avg = dice_sum / (i * args.batch_size)

		if Train:
			print('train_loss_avg:', "{:.4f}".format(train_loss_avg.item()), 'train_dice_avg',
					  "{:.4f}".format(train_dice_avg.item()))
		else:
			print('val_loss_avg:', "{:.4f}".format(train_loss_avg.item()), 'val_dice_avg',
					  "{:.4f}".format(train_dice_avg.item()))
	return train_loss_avg, train_dice_avg


if __name__ == "__main__":
	main()
