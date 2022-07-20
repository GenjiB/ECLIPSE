from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from collections import defaultdict
import json
import random
from dataloaders.rawvideo_util import RawVideoExtractor

import torch
from ipdb import set_trace

import glob


from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
import torchvision


class MSRVTT_DataLoader(Dataset):
	"""MSRVTT dataset loader."""
	def __init__(
			self,
			csv_path,
			features_path,
			tokenizer,
			max_words=30,
			feature_framerate=1.0,
			max_frames=100,
			image_resolution=224,
			frame_order=0,
			slice_framepos=0,
	):
		self.data = pd.read_csv(csv_path)
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

		self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
		self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
							  "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

		self.my_normalize = Compose([
			Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

	def __len__(self):
		return len(self.data)

	def _get_text(self, video_id, sentence):
		choice_video_ids = [video_id]
		n_caption = len(choice_video_ids)

		k = n_caption
		pairs_text = np.zeros((k, self.max_words), dtype=np.long)
		pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
		pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

		for i, video_id in enumerate(choice_video_ids):
			words = self.tokenizer.tokenize(sentence)

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

		return pairs_text, pairs_mask, pairs_segment, choice_video_ids

	def _get_rawvideo_yb(self, my_id):
		video_mask = np.zeros((1, self.max_frames), dtype=np.long)


		video = torch.zeros((1, self.max_frames, 1, 3, self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=torch.double)

		video_folder = '/playpen-iop/yblin/MSRVTT/frames/'+ my_id[0]
		# video_folder = video_folder.replace('playpen-iop','playpen-storage')

		total_num_frames = len(glob.glob(video_folder+'/*.jpg'))

		
		tmp_img_all = []
		# load images 
		if True:
			sample_indx = np.linspace(1, total_num_frames, num=self.max_frames, dtype=int)
			for tmp_idx in sample_indx:
				try:
					tmp_img = torchvision.io.read_image(video_folder+'/'+ str("{:04d}".format(tmp_idx))+ '.jpg').double()/255
				except:
					print('loading images error')
					exit()
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
	
	def _get_rawvideo(self, choice_video_ids):
		video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
		max_video_length = [0] * len(choice_video_ids)

		# Pair x L x T x 3 x H x W
		video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
						  self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float)

		for i, video_id in enumerate(choice_video_ids):
			# Individual for YoucokII dataset, due to it video format
			video_path = os.path.join(self.features_path, "{}.mp4".format(video_id))
			if os.path.exists(video_path) is False:
				video_path = video_path.replace(".mp4", ".webm")

			raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
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
				print("video path: {} error. video id: {}".format(video_path, video_id))

		for i, v_length in enumerate(max_video_length):
			video_mask[i][:v_length] = [1] * v_length

		return video, video_mask

	def __getitem__(self, idx):
		video_id = self.data['video_id'].values[idx]
		sentence = self.data['sentence'].values[idx]

		pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, sentence)
		# video, video_mask = self._get_rawvideo(choice_video_ids)
		video, video_mask = self._get_rawvideo_yb(choice_video_ids)
		
		return pairs_text, pairs_mask, pairs_segment, video, video_mask, video, video_mask

class MSRVTT_TrainDataLoader(Dataset):
	"""MSRVTT train dataset loader."""
	def __init__(
			self,
			csv_path,
			json_path,
			features_path,
			tokenizer,
			max_words=30,
			feature_framerate=1.0,
			max_frames=100,
			unfold_sentences=False,
			image_resolution=224,
			frame_order=0,
			slice_framepos=0,
	):
		self.csv = pd.read_csv(csv_path)
		self.data = json.load(open(json_path, 'r'))
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

		self.unfold_sentences = unfold_sentences
		self.sample_len = 0
		if self.unfold_sentences:
			train_video_ids = list(self.csv['video_id'].values)
			self.sentences_dict = {}
			for itm in self.data['sentences']:
				if itm['video_id'] in train_video_ids:
					self.sentences_dict[len(self.sentences_dict)] = (itm['video_id'], itm['caption'])
			self.sample_len = len(self.sentences_dict)
		else:
			num_sentences = 0
			self.sentences = defaultdict(list)
			s_video_id_set = set()
			for itm in self.data['sentences']:
				self.sentences[itm['video_id']].append(itm['caption'])
				num_sentences += 1
				s_video_id_set.add(itm['video_id'])

			# Use to find the clips in the same video
			self.parent_ids = {}
			self.children_video_ids = defaultdict(list)
			for itm in self.data['videos']:
				vid = itm["video_id"]
				url_posfix = itm["url"].split("?v=")[-1]
				self.parent_ids[vid] = url_posfix
				self.children_video_ids[url_posfix].append(vid)
			self.sample_len = len(self.csv)

		self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
		self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
							  "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
		self.my_normalize = Compose([
			Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

	def __len__(self):
		return self.sample_len

	def _get_text(self, video_id, caption=None):
		k = 1
		choice_video_ids = [video_id]
		pairs_text = np.zeros((k, self.max_words), dtype=np.long)
		pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
		pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

		for i, video_id in enumerate(choice_video_ids):
			if caption is not None:
				words = self.tokenizer.tokenize(caption)
			else:
				words = self._get_single_text(video_id)

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

		return pairs_text, pairs_mask, pairs_segment, choice_video_ids

	def _get_single_text(self, video_id):
		rind = random.randint(0, len(self.sentences[video_id]) - 1)
		caption = self.sentences[video_id][rind]
		words = self.tokenizer.tokenize(caption)
		return words

	def _get_rawvideo(self, choice_video_ids):
		video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
		max_video_length = [0] * len(choice_video_ids)

		# Pair x L x T x 3 x H x W
		video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
						  self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float)

		for i, video_id in enumerate(choice_video_ids):
			# Individual for YoucokII dataset, due to it video format
			video_path = os.path.join(self.features_path, "{}.mp4".format(video_id))
			if os.path.exists(video_path) is False:
				video_path = video_path.replace(".mp4", ".webm")

			raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
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
				print("video path: {} error. video id: {}".format(video_path, video_id))

		for i, v_length in enumerate(max_video_length):
			video_mask[i][:v_length] = [1] * v_length

		return video, video_mask


	def _get_rawvideo_yb(self, my_id):
		video_mask = np.zeros((1, self.max_frames), dtype=np.long)


		video = torch.zeros((1, self.max_frames, 1, 3, self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=torch.double)

		video_folder = '/playpen-iop/yblin/MSRVTT/frames/'+ my_id[0]
		# video_folder = video_folder.replace('playpen-iop','playpen-storage')

		total_num_frames = len(glob.glob(video_folder+'/*.jpg'))

		
		tmp_img_all = []
		# load images 
		if True:
			sample_indx = np.linspace(1, total_num_frames, num=self.max_frames, dtype=int)
			for tmp_idx in sample_indx:
				try:
					tmp_img = torchvision.io.read_image(video_folder+'/'+ str("{:04d}".format(tmp_idx))+ '.jpg').double()/255
				except:
					print('loading images error')
					exit()
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
	def _get_audio(self, idx, s, e):
		
		audio_mask = torch.zeros((1, self.opt.max_audio_frames), dtype=np.long)


		audio = torch.zeros((self.opt.max_audio_frames, 1024, 128), dtype=torch.double) 



		audio_folder = self.video_dict[idx].split('.')[:-1][0].replace('frames',self.opt.audio_pt)

		audio_folder_bk = audio_folder.replace('audio_raw','VGGSound_Audio_features_10s_aligned')
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
				fbank = torch.load(audio_folder+'/'+ str("{:04d}".format(tmp_idx))+ '.pt', map_location=torch.device('cpu'))

				total_fbank.append(fbank)

		else:
			for tmp_idx in range(self.opt.max_audio_frames):	#total_num_wav self.opt.max_audio_frames
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

					except:
						print('Too short: '+ audio_folder+'/'+ str("{:04d}".format(tmp_idx))+ '.wav')
						# print("skip too short")
						continue
					
		
		

		# audio[:total_fbank.size(0)] = total_fbank
		# audio_mask[0, :total_fbank.size(0)] = 1
		# return audio, audio_mask
		total_fbank = torch.vstack(total_fbank)
		return total_fbank, audio_mask
	def __getitem__(self, idx):
		if self.unfold_sentences:
			video_id, caption = self.sentences_dict[idx]
		else:
			video_id, caption = self.csv['video_id'].values[idx], None
		
		pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, caption)

		# video, video_mask = self._get_rawvideo(choice_video_ids)
		video, video_mask = self._get_rawvideo_yb(choice_video_ids)


		text_pt = torch.load('/playpen-iop/yblin/v1-2/train_features_b32_f96/1599_text.pt')
		vis_pt = torch.load('/playpen-iop/yblin/v1-2/train_features_b32_f96/1599_vis.pt')
		return pairs_text, pairs_mask, pairs_segment, video, video_mask, video, video_mask, text_pt, vis_pt, idx
			
