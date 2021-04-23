import torch
import transform_network
import hard_threshold
import torch.nn as nn

class haar_hard_threshold_2d(nn.Module):

	def __init__(self, kernel_size = 2, max_orient = 2, max_level = 4, 
						custom_wavelet = 'haar', learn_transform_filters = False):
		super(haar_hard_threshold_2d, self).__init__()
	   
		self.transform_net = transform_network.transform_network(kernel_size = kernel_size,
																	 max_orient = max_orient, 
																	 max_level = max_level, 
																	 learn_transform_filters = learn_transform_filters);
		
		self.transform_net.custom_filter_initialisation(custom_wavelet);


		self.denoising_net = hard_threshold.denoising_neural_network(max_level = max_level, 
																	max_orient = max_orient);

	def forward(self, x):
	   
		inp_coeff_list = self.transform_net.forward_wavelet_transform(x);

		denoised_coeff_list = self.denoising_net(inp_coeff_list)
		
		rec_img = self.transform_net.inverse_wavelet_transform(denoised_coeff_list, inp_dimension=x.shape[-1] );
		
		return(rec_img)