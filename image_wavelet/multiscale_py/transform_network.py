import torch
# import torch_device
import torch.nn as nn
import pywt
import numpy as np

class transform_network(nn.Module):

	def __init__(self, kernel_size = 5, max_orient = 8, max_level = 4, learn_transform_filters = True,
				 input_dim = 1):

		super(transform_network, self).__init__()

		# assert(kernel_size % 2  == 1);

		self.kernel_size = kernel_size;
		self.max_level = max_level;
		self.max_orient = max_orient;
		self.learn_transform_filters = learn_transform_filters;
	   
		self.forward_filters = nn.Parameter(torch.randn(self.max_orient, 1, self.kernel_size, self.kernel_size), 
													requires_grad = self.learn_transform_filters);

		self.inverse_filters = nn.Parameter(torch.randn(self.max_orient, 1, self.kernel_size, self.kernel_size), 
													requires_grad = self.learn_transform_filters);

		self.forward_filters.data[0, 0] = torch.ones(self.kernel_size, self.kernel_size);
											
		# self.normalize_filters();

		self.d = int((self.kernel_size - 1)/2);



	def custom_filter_initialisation(self, wavelet_init):

		self.max_orient = 4;

		self.forward_filters = nn.Parameter(torch.randn(self.max_orient, 1, self.kernel_size, self.kernel_size), 
													requires_grad = self.learn_transform_filters);

		self.inverse_filters = nn.Parameter(torch.randn(self.max_orient, 1, self.kernel_size, self.kernel_size), 
													requires_grad = self.learn_transform_filters);

		def filter_outer_product(x1, x2):
			return(x1.unsqueeze(0) * x2.unsqueeze(1));

		w=pywt.Wavelet(wavelet_init)
		self.kernel_size = len(w.dec_hi);
		dec_hi = torch.Tensor(w.dec_hi[::-1]) 
		dec_lo = torch.Tensor(w.dec_lo[::-1])
		rec_hi = torch.Tensor(w.rec_hi)
		rec_lo = torch.Tensor(w.rec_lo)

		self.forward_filters.data[0, 0], self.forward_filters.data[1, 0] = filter_outer_product(dec_lo, dec_lo), filter_outer_product(dec_lo, dec_hi);
		self.forward_filters.data[2, 0], self.forward_filters.data[3, 0] = filter_outer_product(dec_hi, dec_lo), filter_outer_product(dec_hi, dec_hi);
		self.inverse_filters.data[0, 0], self.inverse_filters.data[1, 0] = filter_outer_product(rec_lo, rec_lo), filter_outer_product(rec_lo, rec_hi);
		self.inverse_filters.data[2, 0], self.inverse_filters.data[3, 0] = filter_outer_product(rec_hi, rec_lo), filter_outer_product(rec_hi, rec_hi);

	def normalize_filters(self):

		def proj_to_mean_zero(x):
			x.data = x.data - torch.mean(x.data);

		def make_norm_one(x):
			norm_filter = torch.norm(x.data);
			if(norm_filter > 1):
				x.data = x.data/norm_filter;

		for i in range(1, self.max_orient):
			proj_to_mean_zero(self.forward_filters[i, 0]);

		for i in range(self.max_orient):
			make_norm_one(self.forward_filters[i, 0]);
			make_norm_one(self.inverse_filters[i, 0]);
		

	
	def forward_wavelet_transform(self, vimg):
		padding = (self.d, self.d, self.d, self.d);
		# padding = (0,0,0,0)
		result = [0]*(self.max_level+1);
		level = self.max_level;
		h = vimg.size(2)
		w = vimg.size(3)
		while(level>0 and h>1 and w>1):
			padded = torch.nn.functional.pad(vimg, padding)
			res = torch.nn.functional.conv2d(padded, self.forward_filters, stride=2)
			result[level] = res;
			vimg = res[:,:1];
			level = level-1
			h = vimg.size(2)
			w = vimg.size(3)
			assert(h>1 and w>1)
		result[0] = res[:,:1,:,:]
		return result


	def inverse_wavelet_transform(self, coefficient_list, inp_dimension):
		def padding_for_transpose_conv(x):
			input_padding = 0;
			output_padding = 0;
			if(x > 0):
				output_padding = x;
			if(x < 0):
				if(np.abs(x) % 2 == 0):
					input_padding = np.abs(x)/2;
				else:
					output_padding = 1;
					input_padding = (np.abs(x) + 1)/2;
			return(int(input_padding), int(output_padding))

		rec_img = coefficient_list[0];
		for level in range(1, self.max_level+1):

			if(level == self.max_level ):
				padding_calc = np.array(inp_dimension) - 2*( np.array(coefficient_list[level][0,0].size()) - 1) - self.kernel_size;
			else:
				padding_calc = np.array(coefficient_list[level+1][0,0].size()) - 2*( np.array(coefficient_list[level][0,0].size()) - 1) - self.kernel_size;

			input_padding, output_padding = [0,0], [0,0];
			input_padding[0], output_padding[0] = padding_for_transpose_conv(padding_calc[0]);
			input_padding[1], output_padding[1] = padding_for_transpose_conv(padding_calc[1]);
			input_padding, output_padding = tuple(input_padding), tuple(output_padding)
			full_coeff_stack = torch.cat(( rec_img, coefficient_list[level][:,1:] ), dim=1);
			rec_img = torch.nn.functional.conv_transpose2d(full_coeff_stack, self.inverse_filters, stride=2, 
										padding = input_padding, output_padding = output_padding);

		return rec_img



	def forward(self, x):

		inp_coeff_list = self.forward_wavelet_transform(x);
		
		rec_img = self.inverse_wavelet_transform(inp_coeff_list, inp_dimension=x[0,0].size() );
		return rec_img
		