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
train_root = '/home/wangjq/wangxt/datasets/gesture-dataset/hand_gesture_v1/train'
val_root = '/home/wangjq/wangxt/datasets/gesture-dataset/hand_gesture_v1/val'

if data_name == "hand14c":
    classes = ('000-one', '001-five', '002-fist', '003-ok', '004-heartSingle', '005-yearh', '006-three',
            '007-four', '008-six', '009-Iloveyou', '010-gun', '011-thumbUp', '012-nine', '013-pink')
elif data_name == "hand3c":
    classes = ('0', 'close', 'open')
elif data_name == "hand5c":
    classes = ('0', 'close-back', 'close-front', 'open-back', 'open-front')

# solver
device_ids = [2]
batch_size = 64
epoch = 100
optim = "sgd"
lr_gamma = 0.5  # 衰减比率
lr_step_size = 20  # 多少 epoch 衰减一次
lr = 1e-4
momentum = 0.9
weight_decay = 5e-4
num_workers = 8
use_amp = False

# model info
model = "resnet10"
pretrained = None
save_checkpoint = 'checkpoint'
resume = "checkpoint/resnet10_hand14c_128x128_91.429.pth"

# knowledge distill
teacher = None
teacker_ckpt = "checkpoint/resnet18_hand14c_128x128_93.429.pth"
alpha = 0.9  # 当 alpha 为0时, 意味着不使用 output 进行蒸馏
temperature = 6
dis_feature = {
    'layer1': (0, 'bn2'), 
    'layer2': (0, 'bn2'), 
    'layer3': (0, 'bn2'),
}
