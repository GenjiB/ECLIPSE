from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset
import numpy as np
import json
import math
from dataloaders.rawvideo_util import RawVideoExtractor
import glob
import torchvision
import torch
import torchaudio
from ipdb import set_trace
from modules.ast_models import ASTModel
from modules import resnet_models


### VGGSound
from scipy import signal
import soundfile as sf

## VGGish
from . import mel_features
from . import vggish_params


from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image

import shutil

class ActivityNet_DataLoader(Dataset):
	def __init__(
			self,
			subset,
			data_path,
			features_path,
			tokenizer,
			opt,
			max_words=30,
			feature_framerate=1.0,
			max_frames=100,
			image_resolution=224,
			frame_order=0,
			slice_framepos=0,
	):
		
		self.opt = opt
		self.data_path = data_path
		self.features_path = features_path
		self.feature_framerate = feature_framerate
		self.max_words = max_words
		self.max_frames = max_frames
		self.tokenizer = tokenizer
		# 0: ordinary order; 1: reverse order; 2: random order.
		self.frame_order = frame_order
		assert self.frame_order in [0, 1, 2]
		# 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
		self.slice_framepos = slice_framepos
		assert self.slice_framepos in [0, 1, 2]

		self.subset = subset
		assert self.subset in ["train", "val"]

		video_id_path_dict = {}
		
		video_id_path_dict["train"] = os.path.join(self.data_path, "train_ids.json")
		video_id_path_dict["val"] = os.path.join(self.data_path, "val_ids.json")

		video_json_path_dict = {}
		video_json_path_dict["train"] = os.path.join(self.data_path, "train.json")
		video_json_path_dict["val"] = os.path.join(self.data_path, "val_1.json")

		pseudo_video_id_list, video_id_list = self._get_video_id_single(video_id_path_dict[self.subset])
		pseudo_caption_dict = self._get_captions_single(video_json_path_dict[self.subset])

		
		print("video id list: {}".format(len(video_id_list)))
		print("pseudo caption dict: {}".format(len(pseudo_caption_dict.keys())))

		video_dict = {}
		for root, dub_dir, video_files in os.walk(self.features_path):
			for video_file in video_files:
				video_id_ = ".".join(video_file.split(".")[:-1])
				video_id_ = video_id_[2:] # yb modify
				if video_id_ not in video_id_list:
					continue
				file_path_ = os.path.join(root, video_file)
				video_dict[video_id_] = file_path_
		self.video_dict = video_dict
		print("video dict: {}".format(len(video_dict)))
		

		self.pseudo_video_id_list = pseudo_video_id_list
		self.video_id_list = video_id_list
		self.pseudo_caption_dict = pseudo_caption_dict

		# Get iterator video ids
		self.video_id2idx_dict = {pseudo_video_id: id for id, pseudo_video_id in enumerate(self.pseudo_video_id_list)}
		# Get all captions
		self.iter2video_pairs_dict = {}
		for pseudo_video_id, video_id in zip(self.pseudo_video_id_list, self.video_id_list):
			if pseudo_video_id not in self.pseudo_caption_dict or video_id not in self.video_dict:
				continue
			caption = self.pseudo_caption_dict[pseudo_video_id]
			n_caption = len(caption['start'])
			for sub_id in range(n_caption):
				self.iter2video_pairs_dict[len(self.iter2video_pairs_dict)] = (pseudo_video_id, sub_id)

		self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
		self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
							  "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}


		self.my_normalize = Compose([
			Resize((224,224), interpolation=Image.BICUBIC),
            # CenterCrop(224),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])



		# self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		# self.my_normalize = torchvision.transforms.Compose([self.normalize,])

		# self.AST = ASTModel(label_dim=512, fstride=16, tstride=16, input_fdim=128,
		#                           input_tdim=1024, imagenet_pretrain=True,
		#                           audioset_pretrain=True, model_size='base384').cuda()


		### VGGsound extract ###
		# self.resnet= resnet_models.AVENet(self.opt) 
		# self.resnet = resnet_models.resnet18(num_classes=309, pool='avgpool')
		# checkpoint = torch.load('../VGGSound/H.pth.tar')
		# self.resnet.load_state_dict(checkpoint['model_state_dict'])
		# self.resnet = self.resnet.cuda()
		### VGGsound end ####

		# self.mean = []
		# self.std = []
		# self.all_list = glob.glob('/playpen-iop/yblin/act_val_frames/*/*.jpg')
		self.all_list = glob.glob('/playpen-iop/yblin/act_val_frames/v_Zc7uU4Qwolc/*.jpg')
		
		self.norm_mean = -5.4450
		self.norm_std = 2.9610
		self.my_len = self.__len__()
		self.save_path = []
		

		
	def __len__(self):

		return len(self.all_list)

	def _get_video_id_from_pseduo(self, pseudo_video_id):
		video_id = pseudo_video_id[2:]
		return video_id

	def _get_video_id_single(self, path):
		pseudo_video_id_list = []
		video_id_list = []
		print('Loading json: {}'.format(path))
		with open(path, 'r') as f:
			json_data = json.load(f)

		for pseudo_video_id in json_data:
			if pseudo_video_id in pseudo_video_id_list:
				print("reduplicate.")
			else:
				video_id = self._get_video_id_from_pseduo(pseudo_video_id)
				pseudo_video_id_list.append(pseudo_video_id)
				video_id_list.append(video_id)
		return pseudo_video_id_list, video_id_list

	def _get_captions_single(self, path):
		pseudo_caption_dict = {}
		with open(path, 'r') as f:
			json_data = json.load(f)

		for pseudo_video_id, v_ in json_data.items():
			pseudo_caption_dict[pseudo_video_id] = {}
			duration = v_["duration"]
			pseudo_caption_dict[pseudo_video_id]["start"] = np.array([0], dtype=object)
			pseudo_caption_dict[pseudo_video_id]["end"] = np.array([int(math.ceil(float(duration)))], dtype=object)
			pseudo_caption_dict[pseudo_video_id]["text"] = np.array([" ".join(v_["sentences"])], dtype=object)
		return pseudo_caption_dict

	def _get_text(self, pseudo_video_id, sub_id):
		caption = self.pseudo_caption_dict[pseudo_video_id]
		k = 1
		r_ind = [sub_id]

		starts = np.zeros(k, dtype=np.long)
		ends = np.zeros(k, dtype=np.long)
		pairs_text = np.zeros((k, self.max_words), dtype=np.long)
		pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
		pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

		yb_word = ''
		for i in range(k):
			ind = r_ind[i]
			yb_word = yb_word + caption['text'][ind] 
			start_, end_ = caption['start'][ind], caption['end'][ind]

			words = self.tokenizer.tokenize(caption['text'][ind])
			starts[i], ends[i] = start_, end_

			words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
			total_length_with_CLS = self.max_words - 1
			if len(words) > total_length_with_CLS:
				words = words[:total_length_with_CLS]
			words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

			input_ids = self.tokenizer.convert_tokens_to_ids(words)
			input_mask = [1] * len(input_ids)
			segment_ids = [0] * len(input_ids)
			while len(input_ids) < self.max_words:
				input_ids.append(0)
				input_mask.append(0)
				segment_ids.append(0)
			assert len(input_ids) == self.max_words
			assert len(input_mask) == self.max_words
			assert len(segment_ids) == self.max_words

			pairs_text[i] = np.array(input_ids)
			pairs_mask[i] = np.array(input_mask)
			pairs_segment[i] = np.array(segment_ids)

		return pairs_text, pairs_mask, pairs_segment, starts, ends,yb_word

	def _get_rawvideo(self, idx, s, e):
		video_mask = np.zeros((len(s), self.max_frames), dtype=np.long)
		max_video_length = [0] * len(s)

		# Pair x L x T x 3 x H x W
		video = np.zeros((len(s), self.max_frames, 1, 3,
						  self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float)
		video_path = self.video_dict[idx]
		try:
			for i in range(len(s)):
				start_time = int(s[i])
				end_time = int(e[i])
				start_time = start_time if start_time >= 0. else 0.
				end_time = end_time if end_time >= 0. else 0.
				if start_time > end_time:
					start_time, end_time = end_time, start_time
				elif start_time == end_time:
					end_time = end_time + 1

				# Should be optimized by gathering all asking of this video
				raw_video_data = self.rawVideoExtractor.get_video_data(video_path, start_time, end_time)
				raw_video_data = raw_video_data['video']

				if len(raw_video_data.shape) > 3:
					raw_video_data_clip = raw_video_data
					# L x T x 3 x H x W
					raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
					if self.max_frames < raw_video_slice.shape[0]:
						if self.slice_framepos == 0:
							video_slice = raw_video_slice[:self.max_frames, ...]
						elif self.slice_framepos == 1:
							video_slice = raw_video_slice[-self.max_frames:, ...]
						else:
							sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
							video_slice = raw_video_slice[sample_indx, ...]
					else:
						video_slice = raw_video_slice

					video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)

					slice_len = video_slice.shape[0]
					max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
					if slice_len < 1:
						pass
					else:
						video[i][:slice_len, ...] = video_slice
				else:
					print("video path: {} error. video id: {}, start: {}, end: {}".format(video_path, idx, start_time, end_time))
		except Exception as excep:
			print("video path: {} error. video id: {}, start: {}, end: {}, Error: {}".format(video_path, idx, s, e, excep))
			raise excep

		for i, v_length in enumerate(max_video_length):
			video_mask[i][:v_length] = [1] * v_length

		# if video_mask.sum() != 64:
		# 	set_trace()
		# 	print('25')
		print('video length', raw_video_slice.shape[0])
		return video, video_mask


	def _get_rawvideo_yb(self, idx, s, e):
		video_mask = np.zeros((len(s), self.max_frames), dtype=np.long)


		video = torch.zeros((len(s), self.max_frames, 1, 3, self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=torch.double)

		video_folder = self.video_dict[idx].split('.')[:-1][0].replace('Activity_Videos','Activity_Videos_Frame')
		# video_folder = video_folder.replace('playpen-iop','playpen-storage')

		# total_num_frames = len(glob.glob(video_folder+'/*.jpg'))
		### demo -->
		video_folder = '/playpen-iop/yblin/act_val_frames'
		total_num_frames = len(glob.glob(video_folder+'/v_'+idx+'/*.jpg'))
		## <---
		

		# pass multiple captions
		for i in range(len(s)):
				start_time = int(s[i])
				end_time = int(e[i])
				start_time = start_time if start_time >= 0. else 0.
				end_time = end_time if end_time >= 0. else 0.
				if start_time > end_time:
					start_time, end_time = end_time, start_time
				elif start_time == end_time:
					end_time = end_time + 1
		
		tmp_img_all = []
		tmp_ori_img_all = []
		video_ori = []
		# load images 
		if self.max_frames < (total_num_frames/3) or True:
			# my_end = (end_time-1)*3 if (end_time-1)*3 < total_num_frames else total_num_frames

			sample_indx = np.linspace(start_time*3+1, total_num_frames , num=total_num_frames, dtype=int)
			# sample_indx = np.linspace(start_time*3+1, total_num_frames , num=self.opt.max_frames, dtype=int)
			for tmp_idx in sample_indx:
				try:
					tmp_img = torchvision.io.read_image(video_folder+'/v_'+idx+'/'+ str("{:04d}".format(tmp_idx))+ '.jpg').float()/255
					# tmp_img = torchvision.io.read_image(video_folder+'/'+ str("{:04d}".format(tmp_idx))+ '.jpg').double()/255
				except:
					print('loading images error')
					exit()
					
				tmp_ori_img_all.append(tmp_img)

				tmp_img = self.my_normalize(tmp_img)
				tmp_img_all.append(tmp_img.unsqueeze(0))
				video_mask[:] = 1
	
			video = torch.stack(tmp_img_all, dim=0).unsqueeze(0)
			video_ori = torch.stack(tmp_ori_img_all, dim=0).unsqueeze(0)
		else:
			for tmp_idx in range(0,int(total_num_frames/3)):
				tmp_img = torchvision.io.read_image(video_folder+'/'+ str("{:04d}".format(tmp_idx*3+1))+ '.jpg')/255
				tmp_img = self.my_normalize(tmp_img)
				tmp_img_all.append(tmp_img.unsqueeze(0))
			vid_length = len(tmp_img_all)
			video_mask[:, :vid_length] = 1

			video[:,:vid_length] = torch.stack(tmp_img_all, dim=0).unsqueeze(0)
			# for tmp_idx in range(1, total_num_frames):
				

		
		return video,video_ori, video_mask

	def _wav2fbank(self, filename, filename2=None, idx=None):
		# mixup
		if filename2 == None:
			waveform, sr = torchaudio.load(filename)
			waveform = waveform - waveform.mean()
		# mixup
		else:
			waveform1, sr = torchaudio.load(filename)
			waveform2, _ = torchaudio.load(filename2)

			waveform1 = waveform1 - waveform1.mean()
			waveform2 = waveform2 - waveform2.mean()

			if waveform1.shape[1] != waveform2.shape[1]:
				if waveform1.shape[1] > waveform2.shape[1]:
					# padding
					temp_wav = torch.zeros(1, waveform1.shape[1])
					temp_wav[0, 0:waveform2.shape[1]] = waveform2
					waveform2 = temp_wav
				else:
					# cutting
					waveform2 = waveform2[0, 0:waveform1.shape[1]]

			# sample lambda from uniform distribution
			#mix_lambda = random.random()
			# sample lambda from beta distribtion
			mix_lambda = np.random.beta(10, 10)

			mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
			waveform = mix_waveform - mix_waveform.mean()
		

		## yb: align ##
		if waveform.shape[0] > 16000*(self.opt.yb_audio_length+0.1):
			sample_indx = np.linspace(0, waveform.shape[0] -16000*(self.opt.yb_audio_length+0.1), num=self.opt.max_audio_frames, dtype=int)
			waveform = waveform[:,sample_indx[idx]:sample_indx[idx]+int(16000*self.opt.yb_audio_length)]
		## align end ##

		fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
												  window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)

		fbank = (fbank - self.norm_mean) / (self.norm_std * 2)

		target_length = int(1024 * (self.opt.yb_audio_length/10)) ## for audioset: 10s
		# target_length = 512 ## 5s
		# target_length = 256 ## 2.5s
		n_frames = fbank.shape[0]

		p = target_length - n_frames

		# cut and pad
		if p > 0:
			m = torch.nn.ZeroPad2d((0, 0, 0, p))
			fbank = m(fbank)
		elif p < 0:
			fbank = fbank[0:target_length, :]

		if filename2 == None:
			return fbank, 0
		else:
			return fbank, mix_lambda


	def _get_audio(self, idx, s, e):
		
		audio_mask = torch.zeros((1, self.opt.max_audio_frames), dtype=np.long)


		audio = torch.zeros((self.opt.max_audio_frames, 1024, 128), dtype=torch.double) 



		# audio_folder = self.video_dict[idx].split('.')[:-1][0].replace('Activity_Videos','Audio_features_all')


		audio_folder = self.video_dict[idx].split('.')[:-1][0].replace('Activity_Videos',self.opt.audio_pt)
		# audio_folder = self.video_dict[idx].split('.')[:-1][0].replace('Activity_Videos','audio_raw')

		# audio_folder = self.video_dict[idx].split('.')[:-1][0].replace('Activity_Videos','Audio_Segment')
		# 
		# audio_folder = self.video_dict[idx].split('.')[:-1][0].replace('Activity_Videos','Audio_features_all_10s_aligned_stride_16')


		audio_folder_bk = audio_folder.replace('audio_raw','VGGSound_Audio_features_20s_aligned')
		self.save_path = audio_folder_bk
		# audio_folder = audio_folder.replace('playpen-iop','playpen-storage')
		
		total_num_wav = len(glob.glob(audio_folder+'/*.wav'))
		total_num_pt = len(glob.glob(audio_folder+'/*.pt'))

		# print('Read: '+ idx)

		total_fbank = []
		# if self.opt.max_audio_frames < total_num_wav: # for frame-wise fusion
		
		if self.my_len == 4816 or True:
			sample_indx = np.linspace(0, total_num_pt-1, num=self.opt.max_audio_frames, dtype=int)
			for tmp_idx in sample_indx:
				# try:
				# fbank, mix_lambda = self._wav2fbank(audio_folder+'/'+ str("{:04d}".format(tmp_idx))+ '.wav')
				# fbank = fbank.unsqueeze(0)

				# # fbank = (fbank - (-4.2677393)) / ((4.5689974) * 2) # audioset normalization
				# fbank = (fbank + 4.26) / (4.57 * 2) 
				# yb: pre-defined audio feature

				fbank = torch.load(audio_folder+'/'+ str("{:04d}".format(tmp_idx))+ '.pt', map_location=torch.device('cpu'))

				## yb: in case shpae wrong
				
				# fbank = fbank.view(1,fbank.size(-2),fbank.size(-1))

				# print(audio_folder+'/'+ str("{:04d}".format(tmp_idx))+ '.pt')
				total_fbank.append(fbank)
				# except:
				# 	print("Check audio loading")
					# continue
					# 

		else:
			for tmp_idx in range(self.opt.max_audio_frames):	#total_num_wav self.opt.max_audio_frames
				# if not os.path.exists(audio_folder_bk+'/'+ str("{:04d}".format(tmp_idx))+ '.pt'):
					
					### loader for VGGish
					# try:

					# 	samples, samplerate = sf.read(audio_folder+'/'+ '0000.wav')

					# 	if samples.shape[0] > 16000*(self.opt.yb_audio_length+0.1):
					# 		sample_indx = np.linspace(0, samples.shape[0] -16000*(self.opt.yb_audio_length+0.1), num=self.opt.max_audio_frames, dtype=int)
					# 		samples = samples[sample_indx[tmp_idx]:sample_indx[tmp_idx]+int(16000*self.opt.yb_audio_length)]

					# 	else:
					# 		# repeat in case audio is too short
					# 		samples = np.tile(samples,int(self.opt.yb_audio_length))[:int(16000*self.opt.yb_audio_length)]

					# 	samples[samples > 1.] = 1.
					# 	samples[samples < -1.] = -1.

					# 	log_mel = mel_features.log_mel_spectrogram(
					# 				samples,
					# 				audio_sample_rate=vggish_params.SAMPLE_RATE,
					# 				log_offset=vggish_params.LOG_OFFSET,
					# 				window_length_secs=vggish_params.STFT_WINDOW_LENGTH_SECONDS,
					# 				hop_length_secs=vggish_params.STFT_HOP_LENGTH_SECONDS,
					# 				num_mel_bins=vggish_params.NUM_MEL_BINS,
					# 				lower_edge_hertz=vggish_params.MEL_MIN_HZ,
					# 				upper_edge_hertz=vggish_params.MEL_MAX_HZ)
					# 	# Frame features into examples.
					# 	features_sample_rate = 1.0 / vggish_params.STFT_HOP_LENGTH_SECONDS
					# 	example_window_length = int(round(
					# 		vggish_params.EXAMPLE_WINDOW_SECONDS * features_sample_rate))
					# 	example_hop_length = int(round(
					# 		vggish_params.EXAMPLE_HOP_SECONDS * features_sample_rate))
					# 	log_mel_examples = mel_features.frame(
					# 		log_mel,
					# 		window_length=example_window_length,
					# 		hop_length=example_hop_length)

					# 	log_mel_examples = torch.tensor(log_mel_examples)[:, None, :, :].float()

					# 	total_fbank.append(log_mel_examples.unsqueeze(0))
						
						

					### loader for VGGSound
					try:
							
						samples, samplerate = sf.read(audio_folder+'/'+ '0000.wav')

						if samples.shape[0] > 16000*(self.opt.yb_audio_length+0.1):
							sample_indx = np.linspace(0, samples.shape[0] -16000*(self.opt.yb_audio_length+0.1), num=self.opt.max_audio_frames, dtype=int)
							samples = samples[sample_indx[tmp_idx]:sample_indx[tmp_idx]+int(16000*self.opt.yb_audio_length)]

						else:
							# repeat in case audio is too short
							samples = np.tile(samples,int(self.opt.yb_audio_length))[:int(16000*self.opt.yb_audio_length)]

						samples[samples > 1.] = 1.
						samples[samples < -1.] = -1.

						frequencies, times, spectrogram = signal.spectrogram(samples, samplerate, nperseg=512,noverlap=353)
						spectrogram = np.log(spectrogram+ 1e-7)

						mean = np.mean(spectrogram)
						std = np.std(spectrogram)
						spectrogram = np.divide(spectrogram-mean,std+1e-9)

						total_fbank.append(torch.tensor(spectrogram).unsqueeze(0).float())

					# try:
					# 	# fbank, mix_lambda = self._wav2fbank(audio_folder+'/'+ str("{:04d}".format(tmp_idx))+ '.wav')
						
					# 	fbank, mix_lambda = self._wav2fbank(audio_folder+'/'+ '0000.wav', idx=tmp_idx)


						
					# 	total_fbank.append(fbank.unsqueeze(0))

					# 	#### yb: for extract audio only
					# 	# with torch.no_grad():
					# 	# 	audio_feature = self.AST(fbank.unsqueeze(0).cuda()) # 1x1024x128
					# 	# # 
					# 	# if not os.path.exists(audio_folder_bk):
					# 	# 	os.mkdir(audio_folder_bk)
					# 	# torch.save(audio_feature.cpu(), audio_folder_bk+'/'+ str("{:04d}".format(tmp_idx))+ '.pt')
					# 	# print('save file:', audio_folder_bk+'/'+ str("{:04d}".format(tmp_idx))+ '.pt')
					# 	#### end ###

					# 	# total_fbank.append(fbank.unsqueeze(0).cpu())
					except:
						print('Too short: '+ audio_folder+'/'+ str("{:04d}".format(tmp_idx))+ '.wav')
						# print("skip too short")
						continue
					
		
		

		# audio[:total_fbank.size(0)] = total_fbank
		# audio_mask[0, :total_fbank.size(0)] = 1
		# return audio, audio_mask
		total_fbank = torch.vstack(total_fbank)
		return total_fbank, audio_mask
		

	def _get_rawvideo_clip_aug(self, idx, s, e):
		video_mask = np.zeros((len(s), self.max_frames), dtype=np.long)


		video = torch.zeros((len(s), self.max_frames, 1, 3, self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=torch.double)

		video_folder = self.video_dict[idx].split('.')[:-1][0].replace('Activity_Videos','Activity_Videos_Frame')
		
		###
		video_folder = video_folder.replace('playpen-iop','playpen-storage')
		video_folder = video_folder.replace('v1-2','')
		###

		total_num_frames = len(glob.glob(video_folder+'/*.jpg'))


		# pass multiple captions
		for i in range(len(s)):
				start_time = int(s[i])
				end_time = int(e[i])
				start_time = start_time if start_time >= 0. else 0.
				end_time = end_time if end_time >= 0. else 0.
				if start_time > end_time:
					start_time, end_time = end_time, start_time
				elif start_time == end_time:
					end_time = end_time + 1
		
		tmp_img_all = []
		# load images 
		if self.max_frames < total_num_frames:
			sample_indx = np.linspace(1, total_num_frames , num=self.max_frames+1, dtype=int)

			for idx_itr in range(0, self.max_frames):
				tmp_idx = np.random.randint(sample_indx[idx_itr], sample_indx[idx_itr+1])
				tmp_img = torchvision.io.read_image(video_folder+'/'+ str("{:04d}".format(tmp_idx))+ '.jpg').double()/255
				tmp_img = self.my_normalize(tmp_img)
				tmp_img_all.append(tmp_img.unsqueeze(0))
				video_mask[:] = 1

			video = torch.stack(tmp_img_all, dim=0).unsqueeze(0)
		else:
			for tmp_idx in range(0,int(total_num_frames/3)):
				tmp_img = torchvision.io.read_image(video_folder+'/'+ str("{:04d}".format(tmp_idx*3+1))+ '.jpg')/255
				tmp_img = self.my_normalize(tmp_img)
				tmp_img_all.append(tmp_img.unsqueeze(0))
			vid_length = len(tmp_img_all)
			video_mask[:, :vid_length] = 1

			video[:,:vid_length] = torch.stack(tmp_img_all, dim=0).unsqueeze(0)
			# for tmp_idx in range(1, total_num_frames):
		
		return video, video_mask

	def _get_rawvideo_fixed_length(self, idx, s, e):
		video_mask = np.zeros((len(s), self.max_frames), dtype=np.long)


		video = torch.zeros((len(s), self.max_frames, 1, 3, self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=torch.double)

		video_folder = self.video_dict[idx].split('.')[:-1][0].replace('Activity_Videos','Activity_Videos_Frame')
		total_num_frames = len(glob.glob(video_folder+'/*.jpg'))


		# pass multiple captions
		for i in range(len(s)):
				start_time = int(s[i])
				end_time = int(e[i])
				start_time = start_time if start_time >= 0. else 0.
				end_time = end_time if end_time >= 0. else 0.
				if start_time > end_time:
					start_time, end_time = end_time, start_time
				elif start_time == end_time:
					end_time = end_time + 1
		
		tmp_img_all = []
		# load images 
		if self.max_frames < (total_num_frames/3) and (total_num_frames/3) > (self.opt.fixed_length+3):
			my_end = (end_time-1)*3 if (end_time-1)*3 < total_num_frames else total_num_frames

			offset = np.random.randint((total_num_frames/3) - self.opt.fixed_length)
			sample_indx = np.linspace((start_time + offset)*3+1, (start_time + offset + self.opt.fixed_length)*3  , num=self.max_frames, dtype=int)

			
			for tmp_idx in sample_indx:
				tmp_img = torchvision.io.read_image(video_folder+'/'+ str("{:04d}".format(tmp_idx))+ '.jpg').double()/255
				tmp_img = self.my_normalize(tmp_img)
				tmp_img_all.append(tmp_img.unsqueeze(0))
				video_mask[:] = 1
	
			video = torch.stack(tmp_img_all, dim=0).unsqueeze(0)


		elif self.max_frames < (total_num_frames/3):
			my_end = (end_time-1)*3 if (end_time-1)*3 < total_num_frames else total_num_frames

			sample_indx = np.linspace(start_time*3+1, my_end , num=self.max_frames, dtype=int)
			for tmp_idx in sample_indx:
				try:
					tmp_img = torchvision.io.read_image(video_folder+'/'+ str("{:04d}".format(tmp_idx))+ '.jpg').double()/255
				except:
					print('error when loading images')
					exit()
					set_trace()
				tmp_img = self.my_normalize(tmp_img)
				tmp_img_all.append(tmp_img.unsqueeze(0))
				video_mask[:] = 1
	
			video = torch.stack(tmp_img_all, dim=0).unsqueeze(0)
		else:
			for tmp_idx in range(0,int(total_num_frames/3)):
				tmp_img = torchvision.io.read_image(video_folder+'/'+ str("{:04d}".format(tmp_idx*3+1))+ '.jpg')/255
				tmp_img = self.my_normalize(tmp_img)
				tmp_img_all.append(tmp_img.unsqueeze(0))
			vid_length = len(tmp_img_all)
			video_mask[:, :vid_length] = 1

			video[:,:vid_length] = torch.stack(tmp_img_all, dim=0).unsqueeze(0)
			# for tmp_idx in range(1, total_num_frames):
				

		
		return video, video_mask
	
	def __getitem__(self, feature_idx):


		cur_path = self.all_list[feature_idx]
		tmp_img = torchvision.io.read_image(cur_path).float()/255
		tmp_img_norm = self.my_normalize(tmp_img)

		name = cur_path.split('/')[-2]
		cur_num = float(os.path.splitext(cur_path)[0].split('/')[-1])
		total_img = len(glob.glob('/playpen-iop/yblin/act_val_frames/'+name+'/*.jpg'))
		audio_idx = math.ceil((cur_num/total_img)*32)-1

		audio = torch.load('/playpen-storage/yblin/v1-2/VGGSound_Audio_features_10s_aligned/'+name+'/'+str("{:04d}".format(audio_idx))+'.pt', map_location=torch.device('cpu'))
		


		my_data = {'path': cur_path, 'img':tmp_img_norm,
		'img_ori':tmp_img, 'audio':audio, 'img_mask':torch.ones((1, 1), dtype=np.long)}

		return my_data