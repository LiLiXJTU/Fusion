class args():
	# training args
	epochs = 150 #"number of training epochs, default is 2"
	batch_size = 4 #"batch size for training, default is 4"
	model_path = './models'
	train_path = '/mnt/sda/li/SUV_CT_SEG/Train/'
	val_path = '/mnt/sda/li/SUV_CT_SEG/Val/'
	# train_path = r'D:\data\body\Train\SUV\Train/'
	# val_path = r'D:\data\body\Train\SUV/'
	model = 'Fusion'
	HEIGHT = 224
	WIDTH = 224

	save_fusion_model = "./models/train/fusionnet/"
	save_loss_dir = './models/train/loss_fusionnet/'

	image_size = 224 #"size of training images, default is 256 X 256"
	cuda = 1 #"set it to 1 for running on GPU, 0 for CPU"
	seed = 42 #"random seed for training"

	lr = 1e-5 #"learning rate, default is 0.001"
	log_interval = 10 #"number of images after which the training loss is logged, default is 500"
	resume_fusion_model = None
	# nest net model
	resume_nestfuse = './models/nestfuse/nestfuse_gray_1e2.model'
	fusion_model = './models/rfn_twostage/'



