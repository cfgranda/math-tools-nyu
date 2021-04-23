import torch
import transform_network_1D
import soft_threshold
import torch.nn as nn

class haar_soft_threshold_1d(nn.Module):

	def __init__(self, kernel_size = 2, max_orient = 2, max_level = 4, 
						custom_wavelet = 'haar', learn_transform_filters = False):
		super(haar_soft_threshold_1d, self).__init__()
	   
		self.transform_net = transform_network_1D.transform_network_1D(kernel_size = kernel_size,
																	 max_orient = max_orient, 
																 	 max_level = max_level, 
																 	 learn_transform_filters = learn_transform_filters);
		self.transform_net.custom_filter_initialisation(custom_wavelet);


		self.denoising_net = soft_threshold.denoising_neural_network(max_level = max_level, 
																	max_orient = max_orient);

	def forward(self, x):
	   
		inp_coeff_list = self.transform_net.forward_wavelet_transform(x);

		denoised_coeff_list = self.denoising_net(inp_coeff_list)
		
		rec_img = self.transform_net.inverse_wavelet_transform(denoised_coeff_list, inp_dimension=x.shape[-1] );
		
		return(rec_img)