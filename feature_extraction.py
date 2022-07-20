from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from gpuinfo import GPUInfo
import os
from ipdb import set_trace

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["NCCL_DEBUG"]="INFO"
os.environ["NCCL_SOCKET_IFNAME"]="docker0"

# os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"
# os.environ['MASTER_PORT'] = '30678'
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

mygpu = GPUInfo.get_info()[0]
# if len(mygpu) == 0 :
#     os.environ['MASTER_PORT'] = '9527'
#     os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"
# elif mygpu['N/A'][0] == '4':
#     os.environ['MASTER_PORT'] = '9527'
#     os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
	
# else:
#     os.environ['MASTER_PORT'] = '5278'
#     os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"


num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
import torch
import numpy as np
import random
import torch.nn.functional as F



# os.environ["NCCL_P2P_DISABLE"] = '1'
# os.environ["NCCL_IB_DISABLE"] = '1'


	

# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
from metrics import compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim
import time
import argparse
import wandb
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling import CLIP4Clip
from modules.optimization import BertAdam

from util import parallel_apply, get_logger
from dataloaders.data_dataloaders import DATALOADER_DICT
from thop import profile
from thop import clever_format
from einops import rearrange
# import torchprof
from deepspeed.profiling.flops_profiler.profiler import FlopsProfiler
from deepspeed.profiling.flops_profiler.profiler import get_model_profile
from torchvision.utils import save_image
import cv2


from modules import resnet_models

from torchvggish.vggish import VGGish

model_urls = {
    'vggish': 'https://github.com/harritaylor/torchvggish/'
              'releases/download/v0.1/vggish-10086976.pth',
    'pca': 'https://github.com/harritaylor/torchvggish/'
           'releases/download/v0.1/vggish_pca_params-970ea276.pth'
}

from modules.ast_models import ASTModel



torch.distributed.init_process_group(backend="nccl")



from torch.cuda.amp import autocast,GradScaler
import warnings
warnings.filterwarnings('ignore')
global logger

def get_args(description='CLIP4Clip on Retrieval Task'):
	parser = argparse.ArgumentParser(description=description)
	parser.add_argument("--do_pretrain", action='store_true', help="Whether to run training.")
	parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
	parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")

	parser.add_argument('--train_csv', type=str, default='data/.train.csv', help='')
	parser.add_argument('--val_csv', type=str, default='data/.val.csv', help='')
	parser.add_argument('--data_path', type=str, default='data/caption.pickle', help='data pickle file path')
	parser.add_argument('--features_path', type=str, default='data/videos_feature.pickle', help='feature path')

	parser.add_argument('--num_thread_reader', type=int, default=1, help='')
	parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
	parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
	parser.add_argument('--batch_size', type=int, default=256, help='batch size')
	parser.add_argument('--batch_size_val', type=int, default=3500, help='batch size eval')
	parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay')
	parser.add_argument('--n_display', type=int, default=100, help='Information display frequence')
	parser.add_argument('--video_dim', type=int, default=1024, help='video feature dimension')
	parser.add_argument('--seed', type=int, default=42, help='random seed')
	parser.add_argument('--max_words', type=int, default=20, help='')
	parser.add_argument('--max_frames', type=int, default=100, help='')
	parser.add_argument('--max_audio_frames', type=int, default=16, help='')
	parser.add_argument('--feature_framerate', type=int, default=1, help='')
	parser.add_argument('--margin', type=float, default=0.1, help='margin for loss')
	parser.add_argument('--hard_negative_rate', type=float, default=0.5, help='rate of intra negative sample')
	parser.add_argument('--negative_weighting', type=int, default=1, help='Weight the loss for intra negative')
	parser.add_argument('--n_pair', type=int, default=1, help='Num of pair to output from data loader')

	parser.add_argument("--output_dir", default=None, type=str, required=True,
						help="The output directory where the model predictions and checkpoints will be written.")
	parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
	parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
	parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
	parser.add_argument("--warmup_proportion", default=0.1, type=float,
						help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
	parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
						help="Number of updates steps to accumulate before performing a backward/update pass.")
	parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")

	parser.add_argument("--cache_dir", default="", type=str,
						help="Where do you want to store the pre-trained models downloaded from s3")

	parser.add_argument('--fp16', action='store_true',
						help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
	parser.add_argument('--fp16_opt_level', type=str, default='O1',
						help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
							 "See details at https://nvidia.github.io/apex/amp.html")

	parser.add_argument("--task_type", default="retrieval", type=str, help="Point the task `retrieval` to finetune.")
	parser.add_argument("--datatype", default="msrvtt", type=str, help="Point the dataset to finetune.")

	parser.add_argument("--world_size", default=0, type=int, help="distribted training")
	parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
	parser.add_argument("--rank", default=0, type=int, help="distribted training")
	parser.add_argument('--coef_lr', type=float, default=1., help='coefficient for bert branch.')
	parser.add_argument('--use_mil', action='store_true', help="Whether use MIL as Miech et. al. (2020).")
	parser.add_argument('--sampled_use_mil', action='store_true', help="Whether MIL, has a high priority than use_mil.")

	parser.add_argument('--text_num_hidden_layers', type=int, default=12, help="Layer NO. of text.")
	parser.add_argument('--visual_num_hidden_layers', type=int, default=12, help="Layer NO. of visual.")
	parser.add_argument('--cross_num_hidden_layers', type=int, default=4, help="Layer NO. of cross.")

	parser.add_argument('--loose_type', action='store_true', help="Default using tight type for retrieval.")
	parser.add_argument('--expand_msrvtt_sentences', action='store_true', help="")

	parser.add_argument('--train_frame_order', type=int, default=0, choices=[0, 1, 2],
						help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")
	parser.add_argument('--eval_frame_order', type=int, default=0, choices=[0, 1, 2],
						help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")

	parser.add_argument('--freeze_layer_num', type=int, default=0, help="Layer NO. of CLIP need to freeze.")
	parser.add_argument('--slice_framepos', type=int, default=0, choices=[0, 1, 2],
						help="0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.")
	parser.add_argument('--linear_patch', type=str, default="2d", choices=["2d", "3d"],
						help="linear projection of flattened patches.")
	parser.add_argument('--sim_header', type=str, default="meanP",
						choices=["meanP", "seqLSTM", "seqTransf", "tightTransf"],
						help="choice a similarity header.")

	parser.add_argument("--pretrained_clip_name", default="ViT-B/32", type=str, help="Choose a CLIP version")
	

	parser.add_argument("--audio_cluster", default=16, type=int, help="num k for bilinear  pooling")

	parser.add_argument("--wandb", default=0, type=int, help="wandb log")
	parser.add_argument("--model_name", default=None, type=str, help="name")

	parser.add_argument("--fixed_length", default=64, type=int, help="the sampling length")
	
	#########yb ###
	parser.add_argument("--audio_pt", default='Audio_features_all', type=str, help="audio length pt")

	parser.add_argument("--yb_coff_dis", default=1, type=float, help="test")
	parser.add_argument("--yb_coff_loss", default=1, type=float, help="test")

	parser.add_argument("--yb_factor_aduio", default=1, type=float, help="test")
	parser.add_argument("--yb_factor_audio_decay", default=1, type=float, help="test")

	parser.add_argument("--yb_sample_num", default=128, type=int, help="test")

	parser.add_argument("--yb_audio_length", default=10, type=float, help="audio second")

	parser.add_argument("--yb_dual", default=0, type=int, help="dual module")
	parser.add_argument("--yb_av", default=1, type=int, help="dual module")

	parser.add_argument("--yb_time_cross_audio", default=0, type=int, help="cross audio time feature")

	parser.add_argument("--yb_start_layer", default=-1, type=int, help="strat layer")
	args = parser.parse_args()


	if args.sim_header == "tightTransf":
		args.loose_type = False

	# Check paramenters
	if args.gradient_accumulation_steps < 1:
		raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
			args.gradient_accumulation_steps))
	if not args.do_train and not args.do_eval:
		raise ValueError("At least one of `do_train` or `do_eval` must be True.")

	args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

	return args

def set_seed_logger(args):
	global logger
	# predefining random initial seeds
	random.seed(args.seed)
	os.environ['PYTHONHASHSEED'] = str(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

	world_size = torch.distributed.get_world_size()
	torch.cuda.set_device(args.local_rank)
	args.world_size = world_size
	rank = torch.distributed.get_rank()
	args.rank = rank

	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir, exist_ok=True)

	logger = get_logger(os.path.join(args.output_dir, "log.txt"))

	if args.local_rank == 0:
		logger.info("Effective parameters:")
		for key in sorted(args.__dict__):
			logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

	return args

def init_device(args, local_rank):
	global logger

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)

	n_gpu = torch.cuda.device_count()
	logger.info("device: {} n_gpu: {}".format(device, n_gpu))
	args.n_gpu = n_gpu

	if args.batch_size % args.n_gpu != 0 or args.batch_size_val % args.n_gpu != 0:
		raise ValueError("Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
			args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu))

	return device, n_gpu

def init_model(args, device, n_gpu, local_rank):

	if args.init_model:
		model_state_dict = torch.load(args.init_model, map_location='cpu')
	else:
		model_state_dict = None

	# Prepare model
	cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')

	model = CLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

	# load pre-trained
	# model = load_model(-1, args, n_gpu, device, model_file=None)

	model = load_model(18, args, n_gpu, device, model_file=None)


	######### yb
	# model.clip.visual.class_embedding_seq =  torch.nn.Parameter(model.clip.visual.class_embedding.detach())
	# my_weight = model.clip.visual.positional_embedding.detach()
	# my_weight[0] = my_weight[0] + model.clip.visual.class_embedding.detach()

	# model.clip.visual.positional_embedding = torch.nn.Parameter(my_weight)
	###########

	model.to(device)
	return model

def prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, local_rank, coef_lr=1.):

	if hasattr(model, 'module'):
		model = model.module

	param_optimizer = list(model.named_parameters())
	

	# no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'visual.mlp_audio', 'positional_embedding_tmp','visual.mlp_audio_k','visual.mlp_vis', 
	# "audio_vis_attn", 'vis_audio_attn','audio_vis_ln_1','audio_vis_fc','vis_audio_ln_1','vis_audio_fc', 'audio_a_attn', 'audio_a_ln_1',
	# 'audio_a_fc']

	# yb_train = ['visual.mlp_audio', 'positional_embedding_tmp','visual.mlp_audio_k','visual.mlp_vis', 
	# "audio_vis_attn", 'vis_audio_attn','audio_vis_ln_1','audio_vis_fc','vis_audio_ln_1','vis_audio_fc', 'audio_a_attn', 'audio_a_ln_1',
	# 'audio_a_fc']

	no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight','visual.mlp_audio', 'audio_vis_attn', 'vis_audio_attn']

	yb_train = ['visual.mlp_audio','audio_vis_attn', 'vis_audio_attn']

	decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
	no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

	decay_clip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." in n]
	decay_noclip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." not in n]

	no_decay_clip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." in n]
	no_decay_noclip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." not in n]
	
	

	train_param = [(n, p) for n, p in param_optimizer if any(nd in n for nd in yb_train)]

	# print("########## High LR: ############")
	# for (n,p) in train_param:
	# 	print(n)
	# print("########## Low LR: ############")
	# for n, p in decay_clip_param_tp:
	#     print(n)
	new_no_decay_clip_param_tp = [(n, p) for n, p in no_decay_clip_param_tp if not any(nd in n for nd in yb_train)]

	# print("########## Low LR: ############")
	# for (n,p) in decay_clip_param_tp:
	#     print(n)
	##############3
	my_embed = []

	# [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]
	# for n, p in param_optimizer:
	#     if 'visual.mlp_audio' in n or 'visual.AST' in n or 'positional_embedding_tmp' in n or 'mlp_audio_k' in n:
	#     # if any(tmp in no_decay for tmp in n):
	#         set_trace()
	#         print('Differnt LR:', n)
	#         my_embed.append((n, p))
	
	# for n, p in no_decay_clip_param_tp:
	#     if not (('visual.mlp_audio' in n) or ( 'visual.AST' in n) or ('positional_embedding_tmp' in n)  or ('mlp_audio_k' in n)):   
	#         new_no_decay_clip_param_tp.append((n,p))
	no_decay_clip_param_tp = new_no_decay_clip_param_tp
	#################3
	
	weight_decay = 0.2
	optimizer_grouped_parameters = [
		{'params': [p for n, p in decay_clip_param_tp], 'weight_decay': weight_decay, 'lr': args.lr * coef_lr},
		{'params': [p for n, p in decay_noclip_param_tp], 'weight_decay': weight_decay},
		{'params': [p for n, p in no_decay_clip_param_tp], 'weight_decay': 0.0, 'lr': args.lr * coef_lr},
		{'params': [p for n, p in no_decay_noclip_param_tp], 'weight_decay': 0.0},

		{'params': [p for n, p in train_param], 'weight_decay': 0.0, 'lr': args.lr* coef_lr*100}
	]

	scheduler = None
	optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
						 schedule='warmup_cosine', b1=0.9, b2=0.98, e=1e-6,
						 t_total=num_train_optimization_steps, weight_decay=weight_decay,
						 max_grad_norm=1.0)

	model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
													  output_device=local_rank, find_unused_parameters=True)

	return optimizer, scheduler, model

def save_model(epoch, args, model, type_name=""):
	# Only save the model it-self
	model_to_save = model.module if hasattr(model, 'module') else model
	output_model_file = os.path.join(
		args.output_dir, "pytorch_model.bin.{}{}".format("" if type_name=="" else type_name+".", epoch))
	torch.save(model_to_save.state_dict(), output_model_file)
	logger.info("Model saved to %s", output_model_file)
	return output_model_file

def load_model(epoch, args, n_gpu, device, model_file=None):
	if model_file is None or len(model_file) == 0:
		# model_file = os.path.join(args.output_dir, "pytorch_model.bin.{}".format(epoch))
		model_file = '/playpen/yblin/CLIP4Clip/pretrained_models/audio/VGGSound_20s_dual_32f_42_6.pt'
	if os.path.exists(model_file):
		model_state_dict = torch.load('/playpen/yblin/CLIP4Clip/pretrained_models/audio/VGGSound_20s_dual_32f_42_6.pt', map_location='cpu')
		if args.local_rank == 0:
			logger.info("Model loaded from %s", '/playpen/yblin/CLIP4Clip/pretrained_models/audio/VGGSound_20s_dual_32f_42_6.pt')
		# Prepare model
		cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
		model = CLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

		model.to(device)
	else:
		model = None
	return model

def train_epoch_yb(epoch, args, model, train_dataloader, device, n_gpu, optimizer, scheduler, global_step, local_rank=0):
	global logger
	torch.cuda.empty_cache()
	log_step = args.n_display
	start_time = time.time()
	total_loss = 0

	resnet= resnet_models.AVENet('7414') 
	checkpoint = torch.load('../VGGSound/H.pth.tar', map_location=torch.device('cpu'))
	resnet.load_state_dict(checkpoint['model_state_dict'])
	resnet = resnet.cuda()
	resnet.eval()

	vgg = VGGish(urls=model_urls).cuda()
	vgg.eval()
	
	yc2_dict = {}
	if hasattr(model, 'module'):
		model = model.module.to(device)
	else:
		model = model.to(device)
	model.eval()

	# for step, batch in enumerate(train_dataloader):
	# 	input_ids, input_mask, segment_ids, video, video_mask, audio_seg, audio_mask, path = batch
	# 	break

	# AST = ASTModel(label_dim=512, fstride=10, tstride=10, input_fdim=audio_seg.size(-1),
	# 							input_tdim=audio_seg.size(-2), imagenet_pretrain=True,
	# 							audioset_pretrain=True, model_size='base384').cuda()
	for step, batch in enumerate(train_dataloader):
		print('current step:',int(step))
		# if n_gpu == 1:
			# multi-gpu does scattering it-self
			# batch = tuple(t.to(device=device, non_blocking=True) for t in batch)
			# input_ids, input_mask, segment_ids, video, video_mask, audio_seg, audio_mask, path = batch

		mydata = batch


		##### for attention ---->
		save_path = '/playpen-iop/yblin/act_val_att/'
		# name = path[0].split('/')[-1]
		with torch.no_grad():
			visual_output = model.get_visual_output(mydata['img'].unsqueeze(0).unsqueeze(0).unsqueeze(3).cuda(), 
			mydata['img_mask'].permute(1,2,0).cuda(), mydata['audio'].cuda().permute(1,0,2,3), shaped=False, video_frame=-1, video_shape=None, is_train=False)
		

		total_frame = visual_output[0].shape[0]
		for idx in range(total_frame):
			print('process video frame:', idx, '/',total_frame)

			# fv = F.relu(visual_output[0][idx:idx+1,1:])
			# fa = F.relu(visual_output[1][:,idx:idx+1])

			fv = visual_output[0][idx:idx+1,1:]
			fa = visual_output[1][:,idx:idx+1]

			fv_norm = F.normalize(fv, dim=-1)
			fa_norm = F.normalize(fa, dim=-1)
			att = torch.bmm(fv,fa.permute(0,2,1))
			att = (att - att.min())/(att - att.min()).max()
			att = att**3
			att = (att - att.min())/(att - att.min()).max()
			att = att.view(1,1,7,7)
			att_map = F.interpolate(att,mydata['img_ori'][idx].shape[-2:],mode='bicubic')

			att_map = cv2.applyColorMap(np.uint8((att_map*225).squeeze(0).squeeze(0).cpu()), cv2.COLORMAP_HSV)
			att_map = torch.tensor(att_map).double()/255
			att_map = att_map.permute(2,0,1)
			

			name = mydata['path'][idx].split('/')[-2]
			if not os.path.exists(save_path+name):
				os.mkdir(save_path+name)
			cur_num = float(os.path.splitext(mydata['path'][idx])[0].split('/')[-1])
			save_image(mydata['img_ori'][idx]*0.7 + att_map*0.3, fp=save_path+name+'/'+str("{:04d}".format(int(cur_num)))+ '.jpg')

		##### <-------

		#### Save image ------->
		# root_dir = '/playpen-iop/yblin/didemo_data/video_frame2'
		# print('Frames:', video_mask[0].sum())
		# save_path = root_dir + '/' + path[0].split('/')[-1]
		# if not os.path.exists(save_path):
		# 	os.mkdir(save_path)
		# for idx in range(video_mask[0].sum()):
		# 	torch.save(video[0,0,idx,0].float(), save_path+'/'+str("{:04d}".format(idx+1))+ '.pt')
		# 	# save_image(video[0,0,idx,0], fp=save_path+'/'+str("{:04d}".format(idx+1))+ '.jpg')
		# 	print('Saving:', save_path+'/'+str("{:04d}".format(idx+1))+ '.jpg')

		########## <--------

		
			
		#### VGGSound <-------
		# with torch.no_grad():
		# 	out = resnet(audio_seg.permute(1,0,2,3).cuda())
			
		# 	# if not os.path.exists('/playpen-iop/yblin/didemo_data/raw_audio2/'+ path[0].split('/')[-1] +'.mp4.wav'):
		# 	# 	print('no sound:', path[0].split('/')[-1])
		# 	# 	out[:] = 0
		# 	if not os.path.exists(path[0]):
		# 		os.mkdir(path[0])
		# out = out.view(out.size(0),512,-1)
		# out = out.permute(0,2,1)
		# for tmp_idx in range(out.size(0)):
		# 	# torch.save(out[tmp_idx].unsqueeze(0).unsqueeze(1).cpu(), path[0]+'/'+ str("{:04d}".format(tmp_idx))+ '.pt')
		# 	torch.save(out[tmp_idx].unsqueeze(0).cpu(), path[0]+'/'+ str("{:04d}".format(tmp_idx))+ '.pt')
		# 	# 1 x tokens x dim
		######## VGGsound ----->


		### VGGish <------
		# if not os.path.exists(path[0]):
		# 	os.mkdir(path[0])
		# n_f, n_time,_, w, h =audio_seg[0].shape
		# with torch.no_grad():
		# 	tmp_in = rearrange(audio_seg[0],'f t d w h -> (f t) d w h')
		# 	myout = vgg(tmp_in.cuda())

		# 	myout = rearrange(myout,'(f t) d -> f t d', f=n_f, t=n_time)
		# 	myout = myout.mean(dim=1)


		# for tmp_idx in range(myout.size(0)):
		# 	# torch.save(out[tmp_idx].unsqueeze(0).unsqueeze(1).cpu(), path[0]+'/'+ str("{:04d}".format(tmp_idx))+ '.pt')
		# 	torch.save(myout[tmp_idx].unsqueeze(0).unsqueeze(0).cpu(), path[0]+'/'+ str("{:04d}".format(tmp_idx))+ '.pt')
		# 	# 1 x tokens x dim
		### --------> VGGish

		### AST <-------
		
		# with torch.no_grad():
		# 	out = AST(audio_seg.squeeze(0).cuda())


		# if not os.path.exists(path[0]):
		# 	os.mkdir(path[0])
		# for tmp_idx in range(audio_seg.size(1)):

		# 	with torch.no_grad():
		# 		out = AST(audio_seg[:,tmp_idx].cuda())

		# 	if not os.path.exists('/playpen-iop/yblin/didemo_data/raw_audio2/'+ path[0].split('/')[-1] +'.mp4.wav'):
		# 		print('no sound:', path[0].split('/')[-1])
		# 		out[:] = 0
		# 	torch.save(out[:,:2].cpu(), path[0]+'/'+ str("{:04d}".format(tmp_idx))+ '.pt')
		# 	# torch.save(out[tmp_idx,:2].unsqueeze(0).cpu(), path[0]+'/'+ str("{:04d}".format(tmp_idx))+ '.pt')
		# 	###### 1 x tokens x dim
		### ----------> AST
	
	# if len(train_dataloader) == 1333:
	# 	torch.save(yc2_dict, '/playpen/yblin/UniVL/data/youcookii/clip_train.pt')
	# else:
	# 	torch.save(yc2_dict, '/playpen/yblin/UniVL/data/youcookii/clip_val.pt')
	return 87,87

def train_build_dic(epoch, args, model, train_dataloader, device, n_gpu, optimizer, scheduler, global_step, local_rank=0):
	global logger
	torch.cuda.empty_cache()
	model.train()
	log_step = args.n_display
	start_time = time.time()
	total_loss = 0
	
	text_dict = []
	vis_dict = []
	id_dict = []
	for step, batch in enumerate(train_dataloader):
		if n_gpu == 1:
			# multi-gpu does scattering it-self
			batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

		input_ids, input_mask, segment_ids, video, video_mask, audio_seg, audio_mask, text_pt, vis_pt, idx = batch
		text_dict.append(text_pt)
		vis_dict.append(vis_pt)
		id_dict.append(idx)
	
	text_dict = torch.vstack(text_dict)
	vis_dict = torch.vstack(vis_dict)
	id_dict = torch.vstack(id_dict)
	torch.save(text_dict.cpu(),'/playpen-iop/yblin/v1-2/b32-dict-f96/text_dict.pt')
	torch.save(vis_dict.cpu(),'/playpen-iop/yblin/v1-2/b32-dict-f96/vis_dict.pt')
	torch.save(id_dict.cpu(),'/playpen-iop/yblin/v1-2/b32-dict-f96/id_dict.pt')
	return 87,87
def eval_epoch_yb(args, model, test_dataloader, device, n_gpu):

	if hasattr(model, 'module'):
		model = model.module.to(device)
	else:
		model = model.to(device)

	# #################################################################
	## below variables are used to multi-sentences retrieval
	# multi_sentence_: important tag for eval
	# cut_off_points: used to tag the label when calculate the metric
	# sentence_num: used to cut the sentence representation
	# video_num: used to cut the video representation
	# #################################################################
	multi_sentence_ = False
	cut_off_points_, sentence_num_, video_num_ = [], -1, -1
	if hasattr(test_dataloader.dataset, 'multi_sentence_per_video') \
			and test_dataloader.dataset.multi_sentence_per_video:
		multi_sentence_ = True
		cut_off_points_ = test_dataloader.dataset.cut_off_points
		sentence_num_ = test_dataloader.dataset.sentence_num
		video_num_ = test_dataloader.dataset.video_num
		cut_off_points_ = [itm - 1 for itm in cut_off_points_]

	if multi_sentence_:
		logger.warning("Eval under the multi-sentence per video clip setting.")
		logger.warning("sentence num: {}, video num: {}".format(sentence_num_, video_num_))

	model.eval()
	with torch.no_grad():
		batch_list_t = []
		batch_list_v = []
		batch_sequence_output_list, batch_visual_output_list = [], []
		total_video_num = 0

		# ----------------------------
		# 1. cache the features
		# ----------------------------
		for bid, batch in enumerate(test_dataloader):
			batch = tuple(t.to(device) for t in batch)
			input_ids, input_mask, segment_ids, video, video_mask, audio_seg, audio_mask, path = batch
			
	return 9527

def eval_epoch(args, model, test_dataloader, device, n_gpu):


	### VGGSound ##3
	# resnet= resnet_models.AVENet(args) 
	# checkpoint = torch.load('../VGGSound/H.pth.tar')
	# resnet.load_state_dict(checkpoint['model_state_dict'])
	# resnet = resnet.cuda()
	###
	
	# from modules.ast_models import ASTModel
	# #### AST ###
	# AST = ASTModel(label_dim=512, fstride=10, tstride=10, input_fdim=128,
	#                           input_tdim=1024, imagenet_pretrain=True,
	#                           audioset_pretrain=True, model_size='base384').cuda()
	####

	if hasattr(model, 'module'):
		model = model.module.to(device)
	else:
		model = model.to(device)

	# #################################################################
	## below variables are used to multi-sentences retrieval
	# multi_sentence_: important tag for eval
	# cut_off_points: used to tag the label when calculate the metric
	# sentence_num: used to cut the sentence representation
	# video_num: used to cut the video representation
	# #################################################################
	multi_sentence_ = False
	cut_off_points_, sentence_num_, video_num_ = [], -1, -1
	if hasattr(test_dataloader.dataset, 'multi_sentence_per_video') \
			and test_dataloader.dataset.multi_sentence_per_video:
		multi_sentence_ = True
		cut_off_points_ = test_dataloader.dataset.cut_off_points
		sentence_num_ = test_dataloader.dataset.sentence_num
		video_num_ = test_dataloader.dataset.video_num
		cut_off_points_ = [itm - 1 for itm in cut_off_points_]

	if multi_sentence_:
		logger.warning("Eval under the multi-sentence per video clip setting.")
		logger.warning("sentence num: {}, video num: {}".format(sentence_num_, video_num_))

	model.eval()
	with torch.no_grad():
		batch_list_t = []
		batch_list_v = []
		batch_sequence_output_list, batch_visual_output_list = [], []
		total_video_num = 0

		# ----------------------------
		# 1. cache the features
		# ----------------------------
		for bid, batch in enumerate(test_dataloader):
			batch = tuple(t.to(device) for t in batch)
			input_ids, input_mask, segment_ids, video, video_mask, audio_seg, audio_mask = batch

			if multi_sentence_:
				# multi-sentences retrieval means: one clip has two or more descriptions.
				b, *_t = video.shape
				sequence_output = model.get_sequence_output(input_ids, segment_ids, input_mask)
				batch_sequence_output_list.append(sequence_output)
				batch_list_t.append((input_mask, segment_ids,))

				s_, e_ = total_video_num, total_video_num + b
				filter_inds = [itm - s_ for itm in cut_off_points_ if itm >= s_ and itm < e_]

				if len(filter_inds) > 0:
					video, video_mask = video[filter_inds, ...], video_mask[filter_inds, ...]
					# yb add
					visual_output = model.get_visual_output(video, video_mask, audio_seg)
					batch_visual_output_list.append(visual_output)
					batch_list_v.append((video_mask,))
				total_video_num += b
			else:
				sequence_output, visual_output = model.get_sequence_visual_output(input_ids, segment_ids, input_mask, video, video_mask, audio_seg, is_train=False)




				batch_sequence_output_list.append(sequence_output)
				batch_list_t.append((input_mask, segment_ids,))

				batch_visual_output_list.append(visual_output)
				batch_list_v.append((video_mask,))

			print("{}/{}\r".format(bid, len(test_dataloader)), end="")


			# for idx in range(sequence_output.size(0)):
			#     torch.save(sequence_output[idx].cpu(),'/playpen-iop/yblin/v1-2/train_features_b32_f96/'+ str(vid_idx[idx].item())+'_text.pt')
			#     torch.save(visual_output[idx].mean(0, keepdim=True).cpu(),'/playpen-iop/yblin/v1-2/train_features_b32_f96/'+ str(vid_idx[idx].item())+'_vis.pt')


		# ----------------------------------
		# 2. calculate the similarity
		# ----------------------------------
		if n_gpu > 1:
			device_ids = list(range(n_gpu))
			batch_list_t_splits = []
			batch_list_v_splits = []
			batch_t_output_splits = []
			batch_v_output_splits = []
			bacth_len = len(batch_list_t)
			split_len = (bacth_len + n_gpu - 1) // n_gpu
			for dev_id in device_ids:
				s_, e_ = dev_id * split_len, (dev_id + 1) * split_len
				if dev_id == 0:
					batch_list_t_splits.append(batch_list_t[s_:e_])
					batch_list_v_splits.append(batch_list_v)

					batch_t_output_splits.append(batch_sequence_output_list[s_:e_])
					batch_v_output_splits.append(batch_visual_output_list)
				else:
					devc = torch.device('cuda:{}'.format(str(dev_id)))
					devc_batch_list = [tuple(t.to(devc) for t in b) for b in batch_list_t[s_:e_]]
					batch_list_t_splits.append(devc_batch_list)
					devc_batch_list = [tuple(t.to(devc) for t in b) for b in batch_list_v]
					batch_list_v_splits.append(devc_batch_list)

					devc_batch_list = [b.to(devc) for b in batch_sequence_output_list[s_:e_]]
					batch_t_output_splits.append(devc_batch_list)
					devc_batch_list = [b.to(devc) for b in batch_visual_output_list]
					batch_v_output_splits.append(devc_batch_list)

			parameters_tuple_list = [(batch_list_t_splits[dev_id], batch_list_v_splits[dev_id],
									  batch_t_output_splits[dev_id], batch_v_output_splits[dev_id]) for dev_id in device_ids]
			parallel_outputs = parallel_apply(_run_on_single_gpu, model, parameters_tuple_list, device_ids)
			sim_matrix = []
			for idx in range(len(parallel_outputs)):
				sim_matrix += parallel_outputs[idx]
			sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
		else:
			sim_matrix = _run_on_single_gpu(model, batch_list_t, batch_list_v, batch_sequence_output_list, batch_visual_output_list)

	if multi_sentence_:
		logger.info("before reshape, sim matrix size: {} x {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
		cut_off_points2len_ = [itm + 1 for itm in cut_off_points_]
		max_length = max([e_-s_ for s_, e_ in zip([0]+cut_off_points2len_[:-1], cut_off_points2len_)])
		sim_matrix_new = []
		for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_):
			sim_matrix_new.append(np.concatenate((sim_matrix[s_:e_],
												  np.full((max_length-e_+s_, sim_matrix.shape[1]), -np.inf)), axis=0))
		sim_matrix = np.stack(tuple(sim_matrix_new), axis=0)
		logger.info("after reshape, sim matrix size: {} x {} x {}".
					format(sim_matrix.shape[0], sim_matrix.shape[1], sim_matrix.shape[2]))

		tv_metrics = tensor_text_to_video_metrics(sim_matrix)
		vt_metrics = compute_metrics(tensor_video_to_text_sim(sim_matrix))
	else:
		logger.info("sim matrix size: {}, {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
		tv_metrics = compute_metrics(sim_matrix)
		vt_metrics = compute_metrics(sim_matrix.T)
		logger.info('\t Length-T: {}, Length-V:{}'.format(len(sim_matrix), len(sim_matrix[0])))

	logger.info("Text-to-Video:")
	logger.info('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - R@50: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
				format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['R50'], tv_metrics['MR'], tv_metrics['MeanR']))
	logger.info("Video-to-Text:")
	logger.info('\t>>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}'.
				format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['MR'], vt_metrics['MeanR']))

	if args.wandb and args.local_rank==0 :
		wandb.log({"val/T2V R@1": tv_metrics['R1']})
		wandb.log({"val/V2T R@1": vt_metrics['R1']})
	R1 = tv_metrics['R1']
	return R1

def _run_on_single_gpu(model, batch_list_t, batch_list_v, batch_sequence_output_list, batch_visual_output_list):
	sim_matrix = []
	
	for idx1, b1 in enumerate(batch_list_t):
		input_mask, segment_ids, *_tmp = b1
		sequence_output = batch_sequence_output_list[idx1]
		each_row = []
		for idx2, b2 in enumerate(batch_list_v):
			video_mask, *_tmp = b2
			visual_output = batch_visual_output_list[idx2]
			b1b2_logits, *_tmp = model.get_similarity_logits(sequence_output, visual_output, input_mask, video_mask,
																	 loose_type=model.loose_type)
			b1b2_logits = b1b2_logits.cpu().detach().numpy()
			each_row.append(b1b2_logits)
		each_row = np.concatenate(tuple(each_row), axis=-1)
		sim_matrix.append(each_row)
	return sim_matrix

def main():
	global logger
	args = get_args()

	args = set_seed_logger(args)
	device, n_gpu = init_device(args, args.local_rank)

	tokenizer = ClipTokenizer()

	assert  args.task_type == "retrieval"
	
	model = init_model(args, device, n_gpu, args.local_rank)

	if args.wandb and args.local_rank==0 :
		# wandb.init(project='AV-long',config=args) #tags=["epoch"+str(args.epochs)]
		wandb.init(project='AV-long',config=args,name=args.model_name)


	## ####################################
	# freeze testing
	## ####################################
	assert args.freeze_layer_num <= 12 and args.freeze_layer_num >= -1
	if hasattr(model, "clip") and args.freeze_layer_num > -1:
		for name, param in model.clip.named_parameters():
			# top layers always need to train
			if name.find("ln_final.") == 0 or name.find("text_projection") == 0 or name.find("logit_scale") == 0 \
					or name.find("visual.ln_post.") == 0 or name.find("visual.proj") == 0 \
					or name.find("positional_embedding_tmp")!=-1 or name.find("class_embedding_seq")!=-1 \
					or ('AST.mlp_head' in name) or ('mlp_audio_k' in name) or ('mlp_vis' in name)  or ('mlp_audio' in name) \
					or ('conv_audio' in name) or ('positional_audio' in name) or ('audio_cls' in name):  # or
				# print(" grad:", name)
				continue    # need to train
			elif name.find("visual.transformer.resblocks.") == 0 or name.find("transformer.resblocks.") == 0:
				layer_num = int(name.split(".resblocks.")[1].split(".")[0])
				if layer_num >= args.freeze_layer_num:
					# print(" grad:", name)
					continue    # need to train

			if args.linear_patch == "3d" and name.find("conv2."):
				continue
			else:
				# yb
				# paramenters which < freeze_layer_num will be freezed
				print("not grad:", name)
				param.requires_grad = False
	## ####################################
	# dataloader loading
	## ####################################
	assert args.datatype in DATALOADER_DICT

	assert DATALOADER_DICT[args.datatype]["test"] is not None \
		   or DATALOADER_DICT[args.datatype]["val"] is not None

	test_dataloader, test_length = None, 0
	if DATALOADER_DICT[args.datatype]["test"] is not None:
		test_dataloader, test_length = DATALOADER_DICT[args.datatype]["test"](args, tokenizer)

	if DATALOADER_DICT[args.datatype]["val"] is not None:
		val_dataloader, val_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer, subset="val")
	else:
		val_dataloader, val_length = test_dataloader, test_length

	## report validation results if the ["test"] is None
	if test_dataloader is None:
		test_dataloader, test_length = val_dataloader, val_length

	if args.local_rank == 0:
		logger.info("***** Running test *****")
		logger.info("  Num examples = %d", test_length)
		logger.info("  Batch size = %d", args.batch_size_val)
		logger.info("  Num steps = %d", len(test_dataloader))
		logger.info("***** Running val *****")
		logger.info("  Num examples = %d", val_length)

	## ####################################
	# train and eval
	## ####################################
	if args.do_train:
		train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
		num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
										/ args.gradient_accumulation_steps) * args.epochs

		coef_lr = args.coef_lr
		optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, args.local_rank, coef_lr=coef_lr)

		if args.local_rank == 0:
			logger.info("***** Running training *****")
			logger.info("  Num examples = %d", train_length)
			logger.info("  Batch size = %d", args.batch_size)
			logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)

		best_score = 0.00001
		best_output_model_file = "None"
		global_step = 0
		
		early_stop = 10
		stop_count = 0 
		# text_dict, vis_dict = train_build_dic(0, args, model, train_dataloader, device, n_gpu, optimizer, scheduler, global_step)

		# vis_dict = torch.load('/playpen-iop/yblin/v1-2/b32-dict-f96/vis_dict.pt')
		# id_dict = torch.load('/playpen-iop/yblin/v1-2/b32-dict-f96/id_dict.pt')
		# text_dict = torch.load('/playpen-iop/yblin/v1-2/b32-dict-f96/text_dict.pt')

		# vis_dict = torch.load('/playpen-iop/yblin/v1-2/B16-dict/vis_dict.pt')
		# id_dict = torch.load('/playpen-iop/yblin/v1-2/B16-dict/id_dict.pt')
		# text_dict = torch.load('/playpen-iop/yblin/v1-2/B16-dict/text_dict.pt')
		for epoch in range(args.epochs):
			train_sampler.set_epoch(epoch)


			tr_loss, global_step = train_epoch_yb(epoch, args, model, train_dataloader, device, n_gpu, optimizer,
											   scheduler, global_step, local_rank=args.local_rank)

			# tr_loss, global_step = train_epoch_yb(epoch, args, model, val_dataloader, device, n_gpu, optimizer,
			# 								   scheduler, global_step, local_rank=args.local_rank)
			# R1 = eval_epoch(args, model, test_dataloader, device, n_gpu)
			tr_loss, global_step = train_epoch_yb(epoch, args, model, test_dataloader, device, n_gpu, optimizer,
											   scheduler, global_step, local_rank=args.local_rank)
		
	# elif args.do_eval:
	# 	if args.local_rank == 0:
	# 		eval_epoch(args, model, test_dataloader, device, n_gpu)

if __name__ == "__main__":
	main()