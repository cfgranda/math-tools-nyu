import numpy as np
import torch
import cv2
import torch.utils.data as udata
from random import randint


class Dataset(udata.Dataset):
	def __init__(self,  list_of_image_path, resize_to_256 = False):
		super(Dataset, self).__init__()

		self.n_data = len(list_of_image_path)

		self.data_list = [None]*self.n_data;
		for i in range(self.n_data):

			f = list_of_image_path[i];
			Img = cv2.imread(f)
			Img = self.normalize(np.float32(Img[:,:,0]))
			if resize_to_256:
			    Img = cv2.resize(Img, (256, 256)) 
			self.data_list[i] = Img;

	def __len__(self):
		return self.n_data

	def __getitem__(self, index):
		data = self.data_list[index]
		return torch.Tensor(data).unsqueeze(0);

	def normalize(self, data):
		return data/255.

	