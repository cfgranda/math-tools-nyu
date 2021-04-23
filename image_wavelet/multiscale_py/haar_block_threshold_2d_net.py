import torch
import transform_network
import block_threshold
import torch.nn as nn

class haar_block_threshold_2d(nn.Module):

	def __init__(self, kernel_size = 2, max_orient = 2, max_level = 4, window_size = 5,
						custom_wavelet = 'haar', learn_transform_filters = False):
		super(haar_block_threshold_2d, self).__init__()
	   
		self.transform_net = transform_network.transform_network(kernel_size = kernel_size,
																	 max_orient = max_orient, 
																	 max_level = max_level, 
																	 learn_transform_filters = learn_transform_filters);
		
		self.transform_net.custom_filter_initialisation(custom_wavelet);


		self.denoising_net = block_threshold.denoising_neural_network(max_level = max_level, 
																	max_orient = max_orient,
																	window_size = window_size);

	def forward(self, x):
	   
		inp_coeff_list = self.transform_net.forward_wavelet_transform(x);

		denoised_coeff_list = self.denoising_net(inp_coeff_list)
		
		rec_img = self.transform_net.inverse_wavelet_transform(denoised_coeff_list, inp_dimension = x[0,0].size()  );
		
		return(rec_img)