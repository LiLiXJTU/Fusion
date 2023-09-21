import torch.nn as nn
from net import NestFuse_light2_nodense, Fusion_network
class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        nc = 1
        input_nc = nc
        output_nc = nc
        nb_filter = [64, 112, 160, 208, 256]
        f_type = 'res'
        deepsupervision = False
        self.nest_model = NestFuse_light2_nodense(nb_filter, input_nc, output_nc, deepsupervision)
        self.fusion_model = Fusion_network(nb_filter, f_type)
    def forward(self, img, en_CT, en_SUV):
        self.nest_model = self.nest_model.forward(img)
        self.fusion_model = self.fusion_model.forward(en_CT, en_SUV)
