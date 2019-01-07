import os
import sys
import torch
import cv2
import configparser
from fastai.torch_imports import resnet34
from fastai.transforms import tfms_from_model
from fastai.conv_learner import ConvnetBuilder
from fastai.core import V, VV, to_np, is_listy
from fastai.dataset import FilesIndexArrayDataset
from fastai.dataloader import DataLoader
from fastai.model import get_prediction
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def load_img(img_pth):
	flags = cv2.IMREAD_UNCHANGED + cv2.IMREAD_ANYDEPTH + cv2.IMREAD_ANYCOLOR
	img = cv2.imread(str(img_pth), flags).astype(np.float32) / 255
	if img is None:
	    raise OSError(f'File not recognized by opencv: {img_pth}')

	return img


config = configparser.ConfigParser()
config.read('setup.cfg')
section = config.default_section
#pixel_spacing = config[section]['pixel_spacing']
model_weight_path =  config[section]['model_weight_path']


sz = 224
bs = 128
model = resnet34



trn_tfms, val_tfms = tfms_from_model(model, sz)
preprocess = val_tfms
model = ConvnetBuilder(model, 2, False, False, pretrained=False).model
sd = torch.load(model_weight_path)
model.load_state_dict(sd)
model.eval();


if __name__ == "__main__":
	img_pth = sys.argv[1]
	output_file = sys.argv[2]

	img = load_img(img_pth)
	preprocessed_img = preprocess(img)
	preprocessed_img = V(preprocessed_img).unsqueeze(0)
	logprob = model(preprocessed_img)
	logprob = to_np(logprob)
	prob = np.exp(logprob)
	prob = prob[0]

	with open(output_file, 'w') as f:
		f.write(str(prob[1]))

