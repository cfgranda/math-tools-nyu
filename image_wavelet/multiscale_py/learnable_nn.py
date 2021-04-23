import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class denoising_single_orient(nn.Module):

	def __init__(self, kernel_size=1, input_dimension=1, n_hidden = 3, intermedeate_layer_dimension = 16):
		super(denoising_single_orient, self).__init__()
		
		if(kernel_size > 1):
			padding = int((kernel_size -1)/2);
		else:
			padding = 0;
			
		self.n_hidden = n_hidden - 1;

		self.relu_layer  = nn.ReLU();
		self.first_layer = nn.Conv1d(input_dimension, intermedeate_layer_dimension, 
									 kernel_size, stride = 1, padding = padding, bias = True);
		self.last_layer = nn.Conv1d(intermedeate_layer_dimension, input_dimension, kernel_size, 
									stride = 1, padding = padding, bias = True);

		self.hidden_layer_list = [None] * (self.n_hidden);

		for i in range(self.n_hidden):
			self.hidden_layer_list[i] = nn.Conv1d(intermedeate_layer_dimension, intermedeate_layer_dimension, 
												  kernel_size, stride = 1, padding = padding, bias = True);
		self.hidden_layer_list = nn.ModuleList(self.hidden_layer_list);

		self.batchnorm_list = [None] * (self.n_hidden);
		for i in range(self.n_hidden):
			self.batchnorm_list[i] = nn.BatchNorm1d(intermedeate_layer_dimension);
		self.batchnorm_list = nn.ModuleList(self.batchnorm_list);

	def forward(self, x):
		out = self.first_layer(x);
		out = self.relu_layer(out);

		for i in range(self.n_hidden):
			out = self.hidden_layer_list[i](out);
			out = self.batchnorm_list[i](out)
			out = self.relu_layer(out);

		out = self.last_layer(out);
		
		return(out)

class denoising_neural_network(nn.Module):
	def __init__(self, max_level, max_orient):
		super(denoising_neural_network, self).__init__()
					
	   
		self.max_level = max_level;
		self.max_orient = max_orient;
		
		self.neural_net_dict = {};
		for i in range(1, self.max_level+1):
			self.neural_net_dict[str(i)] = [None] * (self.max_orient);

		for level in range(1, self.max_level+1):
			for orient in range(self.max_orient):
				self.neural_net_dict[str(level)][orient] = denoising_single_orient();

				self.neural_net_dict[str(level)][orient] = self.neural_net_dict[str(level)][orient];
	   
			self.neural_net_dict[str(level)] = nn.ModuleList(self.neural_net_dict[str(level)]);
		self.neural_net_dict = nn.ModuleDict(self.neural_net_dict)
		

	def forward(self, input_list):
		assert(len(input_list) == self.max_level+1);

		output_list = [None] * (self.max_level+1);
								
		for level in range(1, self.max_level+1):

			temp_coeff_tensor = torch.zeros_like( input_list[level] );
			
			for orient in range(self.max_orient):

				temp_coeff_tensor[:, orient:orient+1] = self.neural_net_dict[str(level)][orient](input_list[level][:, orient:orient+1]);
								
			output_list[level] = temp_coeff_tensor
							
		output_list[0] = output_list[1][:, :1];

		return(output_list)
