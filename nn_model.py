from __future__ import print_function
import torch
from torch import nn 


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

class NN_Large_Predictor(nn.Sequential):

	def make_encoder(self, enc_dims, dec_dims):
		num_enc = len(enc_dims)
		num_dec = len(dec_dims)
		encoder = []
		decoder = []
		for i in range(num_enc-1):
			if i != 0:
				encoder += [nn.ReLU(True), nn.Linear(enc_dims[i], enc_dims[i+1])]
			else:
				encoder.append(nn.Linear(enc_dims[i], enc_dims[i+1]))
		for i in range(num_dec-1):
			if i != 0:
				decoder += [nn.ReLU(True), nn.Linear(dec_dims[i], dec_dims[i+1])] 
			else:
				decoder.append(nn.Linear(dec_dims[i], dec_dims[i+1]))
		return torch.nn.Sequential(*(encoder)), torch.nn.Sequential(*(decoder))

	def __init__(self, **config):
		super(NN_Large_Predictor, self).__init__()
		self.input_dim = config['input_dim']
		self.encode_fc1_dim = config['encode_fc1_dim']
		self.encode_fc2_dim = config['encode_fc2_dim']
		self.decode_fc1_dim = config['decode_fc1_dim']
		self.decode_fc2_dim = config['decode_fc2_dim']
		self.cat_flag = False
		self.num_flag = False

		#self.ffn_input = config['features_size'] + config['encode_fc2_dim']
		self.ffn_input = config['encode_fc2_dim']

		if config['cat_features_size']>0 and config['extra_features'] == True:
			self.cat_flag = True
			self.cat_input_dim = config['cat_features_size']
			self.cat_encode_fc1_dim = config['cat_encode_fc1_dim']
			self.cat_encode_fc2_dim = config['cat_encode_fc2_dim']
			self.cat_decode_fc1_dim = config['cat_decode_fc1_dim']
			self.cat_decode_fc2_dim = config['cat_decode_fc2_dim']
			cat_enc_dims = [self.cat_input_dim, self.cat_encode_fc1_dim, self.cat_encode_fc2_dim]
			cat_dec_dims = [self.cat_encode_fc2_dim, self.cat_decode_fc1_dim, self.cat_decode_fc2_dim]
			self.cat_encoder, self.cat_decoder = self.make_encoder(cat_enc_dims, cat_dec_dims)

			self.ffn_input += cat_enc_dims[-1]

		if config['num_features_size']>0 and config['extra_features'] == True:
			self.num_flag = True
			self.num_input_dim = config['num_features_size']
			self.num_encode_fc1_dim = config['num_encode_fc1_dim']
			self.num_encode_fc2_dim = config['num_encode_fc2_dim']
			self.num_decode_fc1_dim = config['num_decode_fc1_dim']
			self.num_decode_fc2_dim = config['num_decode_fc2_dim']
			num_enc_dims = [self.num_input_dim, self.num_encode_fc1_dim, self.num_encode_fc2_dim]
			num_dec_dims = [self.num_encode_fc2_dim, self.num_decode_fc1_dim, self.num_decode_fc2_dim]
			self.num_encoder, self.num_decoder = self.make_encoder(num_enc_dims, num_dec_dims)

			self.ffn_input += 92 #num_enc_dims[-1] ###### REMOVE THIS
			# TODO
			#self.ffn_input += num_enc_dims[-1]

		self.predict_dim = config['predict_dim']
		self.predict_out_dim = config['predict_out_dim']
		
		self.dropout_ratio = config['dropout']
		self.ffn_num_layers = config['ffn_num_layers']
		print("input dim:", self.input_dim, "encode dim:", self.encode_fc1_dim)

		enc_dims = [self.input_dim, self.encode_fc1_dim, self.encode_fc2_dim]
		dec_dims = [self.encode_fc2_dim, self.decode_fc1_dim, self.decode_fc2_dim]
		self.encoder, self.decoder = self.make_encoder(enc_dims, dec_dims)

		'''if self.feature_input_dim:
			self.features_ffn = nn.Sequential(
				nn.Linear(self.feature_input_dim, self.feature_fc1_dim),
				nn.Sigmoid()
			)'''

		self.create_predictor()

	def create_predictor(self):
		
		dropout = nn.Dropout(self.dropout_ratio)
		activation = nn.ReLU()
		batchnorm = nn.BatchNorm1d(self.ffn_input) ##### REMOVE THIS

		if self.ffn_num_layers == 1:
			predictor = [
				batchnorm,
                dropout,
                nn.Linear(self.ffn_input, self.predict_out_dim)
            ]
		else:
			predictor = [
				batchnorm,
                dropout,
                nn.Linear(self.ffn_input, self.predict_dim)
            ]
			predictor.extend([
                activation,
                dropout,
                nn.Linear(self.predict_dim, self.predict_dim//2),
            ])
			for _ in range(self.ffn_num_layers - 3):
				predictor.extend([
                    activation,
                    dropout,
                    nn.Linear(self.predict_dim//2, self.predict_dim//2),
                ])
			predictor.extend([
                activation,
                dropout,
                nn.Linear(self.predict_dim//2, self.predict_out_dim),
            ])

		# Create predictor model
		self.predictor = nn.Sequential(*predictor)
	def forward(self, v_D, cat_features = None, num_features=None):
		'''
		:param v_D: batch_size x eta, multi-hot vector
		:return: recon, score, code
		'''
		Z_D = self.encoder(v_D)
		# # decode
		v_D_hat = self.decoder(Z_D)
		recon  = torch.sigmoid(v_D_hat)

		cat_recon = None
		num_recon = None
		#if features_batch is not None:
		if self.cat_flag:
			assert cat_features != None 
			#Z_D_extended = torch.cat([Z_D, features_batch], dim=1)
			cat_e = self.cat_encoder(cat_features.to(device))
			cat_hat = self.cat_decoder(cat_e)
			cat_recon = torch.sigmoid(cat_hat)
			Z_D = torch.cat([Z_D, cat_e], dim=1)  
		
		if self.num_flag:
			# TODO
			'''
			num_e = self.num_encoder(num_features.to(device))
			num_hat = self.num_decoder(num_e)
			num_recon = torch.sigmoid(num_hat)
			Z_D = torch.cat([Z_D, num_e], dim=1)
			'''
			Z_D = torch.cat([Z_D, num_features.to(device)], dim=1)
			#'''

		score = self.predictor(Z_D)
		return  recon, cat_recon, num_recon, score, Z_D