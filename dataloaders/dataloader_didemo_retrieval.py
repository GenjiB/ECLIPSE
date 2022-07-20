from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset
import numpy as np
import json
from dataloaders.rawvideo_util import RawVideoExtractor
import glob
from ipdb import set_trace
import torch
import torchvision

import torchaudio

### VGGSound
from scipy import signal
import soundfile as sf

from torchvision.utils import save_image

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image



class DiDeMo_DataLoader(Dataset):
	def __init__(
			self,
			subset,
			data_path,
			features_path,
			opt,
			tokenizer,
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
		print('run loader:', self.subset)
		assert self.subset in ["train", "val", "test"]
		
		video_id_path_dict = {}
		video_id_path_dict["train"] = os.path.join(self.data_path, "train_list.txt")
		video_id_path_dict["val"] = os.path.join(self.data_path, "val_list.txt")
		video_id_path_dict["test"] = os.path.join(self.data_path, "test_list.txt")

		video_json_path_dict = {}
		video_json_path_dict["train"] = os.path.join(self.data_path, "train_data.json")
		video_json_path_dict["val"] = os.path.join(self.data_path, "val_data.json")
		video_json_path_dict["test"] = os.path.join(self.data_path, "test_data.json")
		
		with open(video_id_path_dict[self.subset], 'r') as fp:
			video_ids = [itm.strip() for itm in fp.readlines()]
		   

		caption_dict = {}
		with open(video_json_path_dict[self.subset], 'r') as f:
			json_data = json.load(f)
		for itm in json_data:
			
			description = itm["description"]
			times = itm["times"]
			video = itm["video"]
			
			if video not in video_ids:
				continue

			# each video is split into 5-second temporal chunks
			# average the points from each annotator
			start_ = np.mean([t_[0] for t_ in times]) * 5
			end_ = (np.mean([t_[1] for t_ in times]) + 1) * 5
			
			if video in caption_dict:
				caption_dict[video]["start"].append(start_)
				caption_dict[video]["end"].append(end_)
				caption_dict[video]["text"].append(description)
			else:
				caption_dict[video] = {}
				caption_dict[video]["start"] = [start_]
				caption_dict[video]["end"] = [end_]
				caption_dict[video]["text"] = [description]


		for k_ in caption_dict.keys():
			caption_dict[k_]["start"] = [0]
			# trick to save time on obtaining each video length
			# [https://github.com/LisaAnne/LocalizingMoments/blob/master/README.md]:
			# Some videos are longer than 30 seconds. These videos were truncated to 30 seconds during annotation.
			caption_dict[k_]["end"] = [31]
			caption_dict[k_]["text"] = [" ".join(caption_dict[k_]["text"])]

		video_dict = {}
		

		# self.all_raw_list = glob.glob(self.features_path+'/*.mp4')
		for root, dub_dir, video_files in os.walk(self.features_path):
			
			for video_file in video_files:
				# yb add --->
				video_file = os.path.splitext(video_file)[0]  # <--
				

				video_id_ = video_file
				if video_id_ not in video_ids:
					continue
				file_path_ = os.path.join(root, video_file)
				video_dict[video_id_] = file_path_

		self.caption_dict = caption_dict
		self.video_dict = video_dict

		video_ids = list(set(video_ids) & set(self.caption_dict.keys()) & set(self.video_dict.keys()))

		
		# Get all captions
		self.iter2video_pairs_dict = {}
		for video_id in self.caption_dict.keys():
			if video_id not in video_ids:
				continue
			caption = self.caption_dict[video_id]
			n_caption = len(caption['start'])
			for sub_id in range(n_caption):
				self.iter2video_pairs_dict[len(self.iter2video_pairs_dict)] = (video_id, sub_id)


		self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
		self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
							  "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
		# self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		# self.normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
		# self.crop = torchvision.transforms.CenterCrop(224)
		self.my_normalize = Compose([
			Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

		self.norm_mean = -5.4450
		self.norm_std = 2.9610

	def __len__(self):
		return len(self.iter2video_pairs_dict)

	def _get_text(self, video_id, sub_id):
		caption = self.caption_dict[video_id]
		k = 1
		r_ind = [sub_id]

		starts = np.zeros(k, dtype=np.long)
		ends = np.zeros(k, dtype=np.long)
		pairs_text = np.zeros((k, self.max_words), dtype=np.long)
		pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
		pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

		for i in range(k):
			ind = r_ind[i]
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

		return pairs_text, pairs_mask, pairs_segment, starts, ends

	def _get_rawvideo(self, idx, s, e):
		video_mask = np.zeros((len(s), self.max_frames), dtype=np.long)
		max_video_length = [0] * len(s)

		# Pair x L x T x 3 x H x W
		video = np.zeros((len(s), self.max_frames, 1, 3,
						  self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float)
		video_path = self.video_dict[idx] + '.mp4'

		# video_path = '/playpen-iop/yblin/didemo_data/raw_low_sacle/' + idx.split('.')[0] + '.mp4'
		
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

				cache_id = "{}_{}_{}".format(video_path, start_time, end_time)
				# Should be optimized by gathering all asking of this video

				raw_video_data = self.rawVideoExtractor.get_video_data(video_path, start_time, end_time)
				raw_video_data = raw_video_data['video']

				if len(raw_video_data.shape) > 3:
					raw_video_data_clip = raw_video_data
					# L x T x 3 x H x W
					
					raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
					# print('dur: ', end_time-start_time, 'start: ',start_time,'end: ', end_time , 'extract frame:', raw_video_slice.shape[0], idx)
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
			pass
			# raise e

		for i, v_length in enumerate(max_video_length):
			video_mask[i][:v_length] = [1] * v_length

		return video, video_mask

	def _get_rawvideo_yb(self, idx, s, e):
		# vid_name = self.all_list[idx].split('/')[-1][:-4]
		video_mask = np.zeros((len(s), self.max_frames), dtype=np.long)


		video = torch.zeros((len(s), self.max_frames, 1, 3, self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=torch.double)


		
		video_folder = self.video_dict[idx]
		video_folder = video_folder.replace('raw_video','video_frame')


		video_folder = video_folder #+'.mp4'
		video_folder = os.path.splitext(video_folder)[0]


		total_num_frames = len(glob.glob(video_folder+'/*.jpg'))

		if total_num_frames == 0:
			print('error path:', video_folder)

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

		my_end = min(end_time, int(total_num_frames/3))

		dur = end_time - start_time

		if self.max_frames < dur or True:

			sample_indx = np.linspace(start_time, my_end-1 , num=self.max_frames, dtype=int)
			
			for tmp_idx in sample_indx:

				tmp_img = torchvision.io.read_image(video_folder+'/'+ str("{:04d}".format(tmp_idx*3+1))+ '.jpg').double()/255
				# tmp_img = torch.load(video_folder+'/'+ str("{:04d}".format(tmp_idx+1))+ '.jpg')
				tmp_img = self.my_normalize(tmp_img)
				tmp_img_all.append(tmp_img.unsqueeze(0))
				video_mask[:] = 1
	
			video = torch.stack(tmp_img_all, dim=0).unsqueeze(0)
		else:
			new_end = min(total_num_frames, end_time)

			for tmp_idx in range(start_time, new_end):
				# tmp_img = torchvision.io.read_image(video_folder+'/'+ str("{:04d}".format(tmp_idx+1))+ '.jpg')/255
				
				tmp_img = torch.load(video_folder+'/'+ str("{:04d}".format(tmp_idx+1))+ '.pt')
				# 3 x 224 x224
				# tmp_img = self.my_normalize(tmp_img)
				tmp_img_all.append(tmp_img.unsqueeze(0))
			vid_length = len(tmp_img_all)

			# print('yb frames:', vid_length)
			video_mask[:, :vid_length] = 1


			video[:,:vid_length] = torch.stack(tmp_img_all, dim=0).unsqueeze(0)

			# for tmp_idx in range(1, total_num_frames):
				

		
		return video, video_mask

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
		


		## didemo only ---->
		waveform = waveform[:16000*30] 
		#### <-----

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


	def _get_audio(self, vid_name, s, e):


		# vid_name = vid_name.split('.')[0]
		audio_folder = '/playpen-storage/yblin/didemo_data/raw_audio2/'
		
		# audio_folder_bk = audio_folder.replace('raw_audio2','didemo_AST_Audio_features_10s_aligned')
		audio_folder_bk = audio_folder.replace('raw_audio2','didemo_VGGSound_Audio_features_10s_aligned')

		self.save_path = audio_folder_bk + '/' + vid_name
		# audio_folder = audio_folder.replace('playpen-iop','playpen-storage')
		

		total_num_wav = len(glob.glob(audio_folder+'/*.wav'))
		total_num_pt = len(glob.glob(self.save_path +'/*.pt'))

		# print('Read: '+ idx)
		audio_mask = torch.zeros((1, self.opt.max_audio_frames), dtype=np.long)

		total_fbank = []
		# if self.opt.max_audio_frames < total_num_wav: # for frame-wise fusion
		
		if True :
			sample_indx = np.linspace(0, total_num_pt-1, num=self.opt.max_audio_frames, dtype=int)
			for tmp_idx in sample_indx:
				# try:
				
				fbank = torch.load(self.save_path+'/'+ str("{:04d}".format(tmp_idx))+ '.pt', map_location=torch.device('cpu'))
				total_fbank.append(fbank)
				# except:
				# 	print("Check audio loading")
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
						
						

					### loader for VGGSound ---->
					# try:
					# 	if os.path.exists(audio_folder+'/'+ vid_name +'.mp4.wav'):
					# 		samples, samplerate = sf.read(audio_folder+'/'+ vid_name +'.mp4.wav')
					# 	else: ## incase no sound
					# 		# print('no sound:'+audio_folder+'/'+ vid_name +'.mp4.wav')
					# 		samples = np.zeros((1600000,))
					# 		samplerate = 16000
					# 	## didemo only ---->
					# 	samples = samples[:16000*35] 
					# 	#<-----
					# 	if samples.shape[0] > 16000*(self.opt.yb_audio_length+0.1):
					# 		sample_indx = np.linspace(0, samples.shape[0] -16000*(self.opt.yb_audio_length+0.1), num=self.opt.max_audio_frames, dtype=int)
					# 		samples = samples[sample_indx[tmp_idx]:sample_indx[tmp_idx]+int(16000*self.opt.yb_audio_length)]

					# 	else:
					# 		# repeat in case audio is too short
					# 		samples = np.tile(samples,int(self.opt.yb_audio_length))[:int(16000*self.opt.yb_audio_length)]

					# 	samples[samples > 1.] = 1.
					# 	samples[samples < -1.] = -1.

						
					# 	frequencies, times, spectrogram = signal.spectrogram(samples, samplerate, nperseg=512,noverlap=353)
					# 	spectrogram = np.log(spectrogram+ 1e-7)

					# 	mean = np.mean(spectrogram)
					# 	std = np.std(spectrogram)
					# 	spectrogram = np.divide(spectrogram-mean,std+1e-9)

					# 	total_fbank.append(torch.tensor(spectrogram).unsqueeze(0).float())
		
					#### <-----


					###### AST ---------->
					# try:
					# fbank, mix_lambda = self._wav2fbank(audio_folder+'/'+ str("{:04d}".format(tmp_idx))+ '.wav')
					# fbank, mix_lambda = self._wav2fbank(audio_folder+'/'+ '0000.wav', idx=tmp_idx)

					if os.path.exists(audio_folder+'/'+ vid_name +'.mp4.wav'):
						fbank, mix_lambda = self._wav2fbank(audio_folder+'/'+ vid_name +'.mp4.wav')
						
					else: ## incase no sound
						# print('no sound:'+audio_folder+'/'+ vid_name +'.mp4.wav')
						fbank = torch.zeros(1024,128)
	
					total_fbank.append(fbank.unsqueeze(0))

					##### <----------------
					# except:
					# 	print('Bug: '+audio_folder+'/'+ vid_name +'.wav')
					# 	# /playpen-iop/yblin/didemo_data/raw_audio//83129246@N00_5202800204_4f1da42afc.wav
					# 	# sf.read(' /playpen-iop/yblin/didemo_data/raw_audio//83129246@N00_5202800204_4f1da42afc.wav')

					# 	# print("skip too short")
					# 	continue
					
		
		

		# audio[:total_fbank.size(0)] = total_fbank
		# audio_mask[0, :total_fbank.size(0)] = 1
		# return audio, audio_mask
		total_fbank = torch.vstack(total_fbank)
		return total_fbank, audio_mask
	def __getitem__(self, feature_idx):

		# for idx_all in range(len(self.iter2video_pairs_dict)):
		# 	video_id, sub_id = self.iter2video_pairs_dict[idx_all]
		# 	pairs_text, pairs_mask, pairs_segment, starts, ends = self._get_text(video_id, sub_id)
		# 	video, video_mask = self._get_rawvideo(video_id, starts, ends)
		# 	root_dir = '/playpen-iop/yblin/didemo_data/video_frame2'
		# 	for idx in range(sum(video_mask[0])):
		# 		save_path = root_dir + '/' + os.path.splitext(video_id)[0]
		# 		if not os.path.exists(save_path):
		# 			os.mkdir(save_path)
		# 		print(idx_all,'saving:', save_path+'/'+str("{:04d}".format(idx+1))+ '.jpg')
		# 		save_image(torch.tensor(video[0,idx]), fp=save_path+'/'+str("{:04d}".format(idx+1))+ '.jpg')
		# set_trace()




		video_id, sub_id = self.iter2video_pairs_dict[feature_idx]

		pairs_text, pairs_mask, pairs_segment, starts, ends = self._get_text(video_id, sub_id)


		# video, video_mask = self._get_rawvideo(video_id, starts, ends)

		video, video_mask = self._get_rawvideo_yb(video_id, starts, ends)



		# save_image(torch.tensor(video[0,0,0]), fp='ori.jpg')
		# tmp_img = torchvision.io.read_image(video_folder+'/'+ str("{:04d}".format(tmp_idx+1))+ '.jpg')/255

		# torchvision.io.read_image('ori.jpg')
		# 

		audio, audio_mask = self._get_audio(video_id, starts, ends)

	





		# root_dir = '/playpen-iop/yblin/didemo_data/video_frame2'
		# for idx in range(sum(video_mask[0])):
		# 	save_path = root_dir + '/' + os.path.splitext(video_id)[0]
		# 	if not os.path.exists(save_path):
		# 		os.mkdir(save_path)
		# 	save_image(torch.tensor(video[0,idx]), fp=save_path+'/'+str("{:04d}".format(idx+1))+ '.jpg')




		# from torchvision.utils import make_grid
		# 
		# grid_img = make_grid(video_yb.squeeze(0).squeeze(1), nrow=8)
		# save_image(grid_img, fp='/playpen/yblin/tmp_shit/'+video_id+'yb.png')

		# grid_img = make_grid(torch.tensor(video).squeeze(0).squeeze(1), nrow=8)
		# save_image(grid_img, fp='/playpen/yblin/tmp_shit/'+video_id+'official.png')



		if self.subset == 'train':
			text_pt = torch.load('/playpen-storage/yblin/v1-2/train_features_b32_f96/1599_text.pt')
			vis_pt = torch.load('/playpen-storage/yblin/v1-2/train_features_b32_f96/1599_vis.pt')
			return pairs_text, pairs_mask, pairs_segment, video, video_mask, audio, audio_mask, text_pt, vis_pt, feature_idx
		else:
			return pairs_text, pairs_mask, pairs_segment, video, video_mask, audio, audio_mask #, self.save_path