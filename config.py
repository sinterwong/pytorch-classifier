import os

# dataset
input_size = (128, 128)
flip_prob = 0.3
affine_prob = 0.2
ro_degree = 10
bright_prob = 0.2
satura_prob = 0.2
contrast_prob = 0.2
hue_prob = 0.2
pad = int(input_size[0] * 0.1)
data_name = "hand14c"
train_root = '/home/wangjq/wangxt/datasets/gesture-dataset/gesture_c14_2/train'
val_root = '/home/wangjq/wangxt/datasets/gesture-dataset/gesture_c14_2/val'

if data_name == "hand14c":
    classes = ('000-one', '001-five', '002-fist', '003-ok', '004-heartSingle', '005-yearh', '006-three',
            '007-four', '008-six', '009-Iloveyou', '010-gun', '011-thumbUp', '012-nine', '013-pink')
elif data_name == "hand3c":
    classes = ('0', 'close', 'open')
elif data_name == "hand5c":
    classes = ('0', 'close-back', 'close-front', 'open-back', 'open-front')

# solver
device_ids = [3]
batch_size = 64
epoch = 150
optim = "sgd"
lr_gamma = 0.5  # 衰减比率
lr_step_size = 21  # 多少 epoch 衰减一次
lr = 1e-2
momentum = 0.9
weight_decay = 5e-4
num_workers = 16
use_amp = False
warmup_step = None

# loss
use_triplet_loss = False  # 是否使用 三元组损失
loss_name = "amsoftmax"  # supports: [ce, amsoftmax]
margin = 0.35
scale = 10.

# model
model = "seresnet10"
pretrained = 'weights/resnet18-5c106cde.pth'
resume = None

# knowledge distill
teacher = None
teacker_ckpt = "checkpoint/hand14c/seresnet18/baseline_2/seresnet18_hand14c_128x128_94.286.pth"
cs_kd = True
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
