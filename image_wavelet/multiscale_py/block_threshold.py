import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class denoising_single_orient(nn.Module):

	def __init__(self, window_size = 5):
		super(denoising_single_orient, self).__init__()
	   
		self.bias = nn.Parameter(torch.tensor(np.random.randn()), requires_grad = True);
# 		self.alpha = nn.Parameter(torch.tensor(np.random.randn()), requires_grad = True);
		self.alpha = 1

		self.window_size = window_size;
		assert( (self.window_size%2) == 1);
		padding = int( (self.window_size-1)//2);
		self.conv_layer = torch.nn.Conv2d(1, 1, kernel_size = self.window_size, stride=1, 
											padding=padding, dilation=1, groups=1, bias=False)
		self.conv_layer.weight.data.fill_(1);
		self.conv_layer.weight.data.requires_grad = False;

		self.large_number_for_sigmoid = 500.0;

	def forward(self, x, eval_mode):
	   
		if self.bias < 0:
			self.bias.data.zero_()

		squared_x = x**2;

		squared_x = self.conv_layer(squared_x);    

		thresholded = F.sigmoid( self.large_number_for_sigmoid * (self.alpha*squared_x - self.bias) );

		out = thresholded * x;
		return(out)

class denoising_neural_network(nn.Module):
	def __init__(self, max_level, max_orient, window_size):
		super(denoising_neural_network, self).__init__()
					
	   
		self.max_level = max_level;
		self.max_orient = max_orient;
		
		self.neural_net_dict = {};
		for i in range(1, self.max_level+1):
			self.neural_net_dict[str(i)] = [None] * (self.max_orient);

		for level in range(1, self.max_level+1):
			for orient in range(self.max_orient):
				self.neural_net_dict[str(level)][orient] = denoising_single_orient(window_size = window_size);

				self.neural_net_dict[str(level)][orient] = self.neural_net_dict[str(level)][orient];
	   
			self.neural_net_dict[str(level)] = nn.ModuleList(self.neural_net_dict[str(level)]);
		self.neural_net_dict = nn.ModuleDict(self.neural_net_dict)
		

	def forward(self, input_list, eval_mode = False):
		assert(len(input_list) == self.max_level+1);

		output_list = [None] * (self.max_level+1);
								
		for level in range(1, self.max_level+1):

			temp_coeff_tensor = torch.zeros_like( input_list[level] );
		

			for orient in range(self.max_orient):
				temp_coeff_tensor[:, orient:orient+1] = self.neural_net_dict[str(level)][orient](input_list[level][:, orient:orient+1], eval_mode );
								
			output_list[level] = temp_coeff_tensor
							
		output_list[0] = output_list[1][:, :1];

		return(output_list)
