import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision as tv
import torchvision.transforms as tfs
import PIL
import numpy as np
import torch.nn.functional as F
import os
import cv2
from skimage import io,morphology,measure
import math
import matplotlib.pyplot as plt
import signal
from sklearn import metrics
import time
import cv2
import argparse
import glob
import importlib
import inspect
import json
import os
import shutil

import matplotlib
import numpy as np
import torch
from docopt import docopt
from tensorboardX import SummaryWriter
from torch.nn import DataParallel  # TODO: switch to DistributedDataParallel
from torch.utils.data import DataLoader
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from config import Config
from dataloader.train_loader import FileLoader
from misc.utils import rm_n_mkdir
from run_utils.engine import RunEngine

from models.hovernet.net_desc import HoVerNet
class Dataset_train(Dataset):
    def __init__(self, path_pos, path_neg):
        super(Dataset_train, self).__init__()
        self.path_pos = path_pos
        self.path_neg = path_neg
        self.list_pos = os.listdir(self.path_pos)
        self.list_neg = os.listdir(self.path_neg)
        self.list_pos.sort()
        self.list_neg.sort()
        self.num_pos = len(self.list_pos)
        self.num_neg = len(self.list_neg)
        self.sizeset = 512

    def __getitem__(self, index):

        if index < self.num_pos:
            image,image_numpy = self.read(self.path_pos, self.list_pos[index], self.sizeset)
            label = torch.tensor([1])
        elif index < self.num_pos + self.num_neg:
            image,image_numpy = self.read(self.path_neg, self.list_neg[index - self.num_pos], self.sizeset)
            label = torch.tensor([0])

        return image, label,image_numpy

    def __len__(self):
        return self.num_pos + self.num_neg

    def read(self, path, name, image_size):
        image_path = os.path.join(path, name)
        image_numpy = io.imread(image_path)[:,:,0:3]
        # img = PIL.Image.fromarray(image_numpy)
        # resize = tfs.Resize((image_size, image_size))
        # img = resize(img)
        # image_numpy = np.array(img)
        # image_numpy = cv2.cvtColor(image_numpy, cv2.COLOR_BGR2GRAY)
        tensorimage = torch.from_numpy(image_numpy).type(torch.FloatTensor).cuda()

        for i in range(tensorimage.shape[2]):
            average = tensorimage[:, :, i].mean()
            tensorimage[:, :, i] = (tensorimage[:, :, i] - average) / 255

        # tensorimage = (tensorimage.unsqueeze(2) - tensorimage.mean())/255


        return tensorimage.permute(2, 0, 1),image_numpy
# class Dataset_test(Dataset):
#     def __init__(self, path_pos, path_neg, path_gdt, path_gtv):
#         super(Dataset_test, self).__init__()
#         self.path_pos = path_pos
#         self.path_neg = path_neg
#         self.path_gdt = path_gdt
#         self.path_gtv = path_gtv
#         self.list_pos = os.listdir(self.path_pos)
#         self.list_neg = os.listdir(self.path_neg)
#         self.list_gdt = os.listdir(self.path_gdt)
#         self.list_gtv = os.listdir(self.path_gtv)
#         self.list_pos.sort()
#         self.list_neg.sort()
#         self.list_gdt.sort()
#         self.list_gtv.sort()
#         self.num_pos = len(self.list_pos)
#         self.num_neg = len(self.list_neg)
#         self.sizeset = 512
#
#     def __getitem__(self, index):
#
#         if index < self.num_pos:
#             image = self.read(self.path_pos, self.list_pos[index], self.sizeset, True).permute(2, 0, 1)
#             label = torch.from_numpy(np.array([1,0])).type(torch.FloatTensor).cuda()
#         else:
#             image = self.read(self.path_neg, self.list_neg[index-self.num_pos], self.sizeset, True).permute(2, 0, 1)
#             label = torch.from_numpy(np.array([0,1])).type(torch.FloatTensor).cuda()
#         return image, label
#
#     def __len__(self):
#         return self.num_pos + self.num_neg
#
#     def read(self, path, name, image_size, normalize):
#         image_path = os.path.join(path, name)
#         image_numpy = io.imread(image_path)
#         # img = PIL.Image.fromarray(image_numpy)
#         # resize = tfs.Resize((image_size, image_size))
#         # img = resize(img)
#         # image_numpy = np.array(img)
#         tensorimage = torch.from_numpy(image_numpy).type(torch.FloatTensor).cuda()
#         if normalize == True:
#             # image_numpy = cv2.cvtColor(tensorimage.to("cpu").numpy(), cv2.COLOR_BGR2GRAY)
#             # tensorimage = torch.from_numpy(image_numpy).type(torch.FloatTensor).cuda()
#             # tensorimage = (tensorimage.unsqueeze(2) - tensorimage.mean()) / 255
#             for i in range(tensorimage.shape[2]):
#                 average = tensorimage[:, :, i].mean()
#                 tensorimage[:, :, i] = (tensorimage[:, :, i] - average) / 255
#
#         return tensorimage
# import sys
# # sys.path.append("..")
# sys.path.append("PytorchLRP")
# from jrieke.utils import load_nifti, save_nifti
# from innvestigator import InnvestigateModel
# from settings import settings
# from jrieke import interpretation
# from nmm_mask_areas import all_areas

# import numpy as np
# import pickle
# import jrieke.models as models
# from PytorchLRP import jrieke
# from PytorchLRP import utils
# from PytorchLRP import inverter_util


# from PytorchLRP.innvestigator import InnvestigateModel
def train(hvnet):
    path_pos = "/data1/wyj/M/datasets/MyNP/positive/"
    path_neg = "/data1/wyj/M/datasets/MyNP/negative/"
    useLRP = False
    print("parameters initializing......")
    net = hvnet.cuda()

    # net.load_state_dict(torch.load('logs/newep001.pth', map_location='cpu'))
    print(net)
    if useLRP == True:
        target_layers = [net.d3]
        cam = GradCAM(model=net, target_layers=target_layers, use_cuda=True)
    # model_prediction, heatmap = inn_model.innvestigate(in_tensor=data)

    batch_size_set = 1
    dataset = Dataset_train(path_pos, path_neg)
    dataload = DataLoader(dataset, batch_size=batch_size_set, shuffle=True, num_workers=0)
    dataset_size = len(dataload.dataset)

    lr = 1e-6
    wd = 0.0005
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)

    loss_fn = F.multilabel_soft_margin_loss
    num_epoch = 30

    print('dataset size:', dataset_size)
    print("batch size:", batch_size_set)

    print("{:*^50}".format("training start"))
    for epoch in range(0,num_epoch):
        # print('Epoch {}/{}'.format(epoch + 1, num_epoch))
        # print('-' * 20)

        epoch_loss = 0
        step = 0

        for batch in dataload:
            image0, label0 ,imnumpy= batch
            image = image0.cuda()
            label = label0.cuda()

            pred= net(image)
            if useLRP==True:
                import matplotlib.pyplot as plt
                targets = None
                grayscale_cam = cam(input_tensor=image, targets=targets)
                grayscale_cam = grayscale_cam[0, :]
                oriim=image0[0,:,:,:].cpu().permute(1,2,0).numpy()



                plt.subplot(2, 2, 1)
                plt.imshow(imnumpy[0, :, :, :])
                plt.subplot(2, 2, 2)
                plt.imshow(grayscale_cam)
                plt.subplot(2, 2, 3)
                plt.imshow(oriim)
                plt.subplot(2, 2, 4)
                plt.imshow(grayscale_cam>0.01)
                plt.show()
            # print(f'{pred},,  {label}')
            loss = loss_fn(pred, label)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            step += 1
            # print("%d/%d,train_loss:%0.4f" % (step, math.ceil(dataset_size // dataload.batch_size), loss.item()))
        epochs = epoch + 1
        average_loss = epoch_loss / math.ceil(dataset_size // dataload.batch_size)
        print("epoch %d loss:%0.4f average loss:%0.4f" % (epochs, epoch_loss, average_loss))
        torch.save(net.state_dict(), 'logs/newep%03d.pth' % (epochs))

if __name__ == "__main__":
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    # device = torch.device("cpu")
    print("device:", device)

    hvnet=HoVerNet()
    train(hvnet)
