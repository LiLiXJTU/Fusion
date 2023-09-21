import os
from torch.utils.data import DataLoader
from Load_Dataset import LoadDatasets
from util.save import save_checkpoint
from tqdm import tqdm, trange
import torch
from torch.optim import Adam
from nets.UNet import UNet
from args_fusion import args
from Loss.DiceBceLoss import WeightedDiceBCE
EPSILON = 1e-5
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
	train_data, val_data = LoadDatasets(args.train_path,args.val_path)
	train_loader = DataLoader(train_data, args.batch_size, shuffle=False)
	val_loader = DataLoader(val_data, args.batch_size, shuffle=False)
	model = UNet(1, 1)
	optimizer = Adam(model.parameters(), args.lr)
	max_dice = 0.0
	for epoch in range(args.epochs):
		print(epoch+1,'/',args.epochs)
		print('start training....')
		model.train(True)
		model.cuda()
		train(train_loader,optimizer,model,Train=True)
		# evaluate on validation set
		print('start validation....')
		with torch.no_grad():
			model.eval()
			val_loss, val_dice = train(val_loader,optimizer,model,Train = False)
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

def train(data_loader,optimizer,model,Train):
	Loss_all = []
	dices = []
	dice_sum = 0
	loss_sum = 0
	for i, (batch, names) in enumerate(data_loader, 1):
		print([i],'/',[len(data_loader)])
		CT, SUV,label = batch['CT_image'], batch['SUV_image'], batch['label']
		# images, masks = sampled_batch['t_image'], sampled_batch['label']
		img_CT, img_SUV, label = CT.cuda(),SUV.cuda(), label.cuda()
		# get fusion image

		preds = model(img_CT)

		######################### LOSS FUNCTION #########################
		out_loss = WeightedDiceBCE()(preds, label.float())
		if Train:
			optimizer.zero_grad()
			out_loss.backward()
			optimizer.step()

		loss_sum = loss_sum + len(label) * out_loss
		if i == len(data_loader):
			train_loss_avg = loss_sum / (args.batch_size*(i-1) + len(label))
		else:
			train_loss_avg = loss_sum / (i * args.batch_size)

		######################### Dice #########################
		train_dice = WeightedDiceBCE()._show_dice(preds, label.float())
		dices.append(train_dice)
		dice_sum = dice_sum + len(label) * train_dice
		if i == len(data_loader):
			train_dice_avg = dice_sum / (args.batch_size * (i - 1) + len(label))
		else:
			train_dice_avg = dice_sum / (i * args.batch_size)

		if Train:
			print('train_loss_avg:',"{:.4f}".format(train_loss_avg.item()),'train_dice_avg',"{:.4f}".format(train_dice_avg.item()))
		else:
			print('val_loss_avg:', "{:.4f}".format(train_loss_avg.item()), 'val_dice_avg', "{:.4f}".format(train_dice_avg.item()))
	return train_loss_avg, train_dice_avg

if __name__ == "__main__":
	main()
