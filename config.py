import os


#########################
######## dataset ########
#########################
input_size = (32, 96)  # h, w
flip_prob = 0.3
affine_prob = 0.2
ro_degree = 10
bright_prob = 0.2
satura_prob = 0.2
contrast_prob = 0.2
hue_prob = 0.2
pad = int(input_size[0] * 0.1)
data_name = "font-color"
train_root = '/home/wangxt/datasets/scp_data0825/train'
val_root = '/home/wangxt/datasets/scp_data0825/val'

if data_name == "font-color":
    classes = ("black", "white")


########################
######## solver ########
########################
device_ids = [0]
batch_size = 128
epoch = 38
optim = "sgd"
lr_gamma = 0.5
lr_step_size = 5
lr = 1e-1
momentum = 0.9
weight_decay = 5e-4
num_workers = 8
use_amp = False
warmup_step = None


########################
######### loss #########
########################
use_triplet_loss = False  # currently nonsupport
loss_name = "ce"  # supports: [ce, amsoftmax]
margin = 0.35
scale = 10.


#########################
######### model #########
#########################
model = "Conformer-tiny-patch16"
pretrained = None
resume = None
# for conformer. Note that Distillation is not currently supported by Conformer
drop_rate = 0.0
drop_path_rate = 0.1
# [0(conv), 1(tran), 2(conv+tran)]
conformer_output_type = 0


#########################
########## KD ###########
#########################
teacher = None
teacker_ckpt = ""
cs_kd = False
alpha = 0.5  # 当 alpha 为0时, 意味着不使用 output 进行蒸馏
temperature = 1
dis_feature = {
    'layer1': (0, 'bn2'),
    # 'layer1': (1, 'bn2'),
    'layer2': (0, 'bn2'),
    # 'layer2': (1, 'bn2'),
    'layer3': (0, 'bn2'),
    # 'layer3': (1, 'bn2'),
}

save_checkpoint = os.path.join('checkpoint', data_name, model)
if not os.path.exists(save_checkpoint):
    os.makedirs(save_checkpoint)
