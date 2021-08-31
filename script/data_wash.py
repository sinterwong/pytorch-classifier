import os
from imutils import paths
import shutil
import tqdm


err_data_folder = "/home/wangxt/workspace/pytorch-classifier/data/error"
real_data_folder = "/home/wangxt/datasets/scp_data0825/train"
output_folder = "/home/wangxt/datasets/scp_data0825/error"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 1. 将bad文件夹中同名的数据删除
# 2. 将错误类别移动至正确类别中

err_data_paths = list(paths.list_images(err_data_folder))

# key: 错误的类别和实际的类别, val: 图片路径
err_dict = {}
for i, p in enumerate(err_data_paths):
    dir_structure = p.split("/")
    if dir_structure[-2] == "bad":
        err_dict.setdefault(dir_structure[-2], [])
        err_dict[dir_structure[-2]].append(os.path.basename(p))
    elif dir_structure[-3] != dir_structure[-2]:
        err_dict.setdefault(dir_structure[-3] + "_" + dir_structure[-2], [])
        err_dict[dir_structure[-3] + "_" +
                 dir_structure[-2]].append(os.path.basename(p))

real_data_paths = list(paths.list_images(real_data_folder))

for k in err_dict.keys():
    if k == "bad":
        continue
    src, dst = k.split("_")
    for i, p in enumerate(real_data_paths):
        p_s = p.split("/")
        cls_name = p_s[-2]
        name = os.path.basename(p)
        if i == 0 and name in err_dict["bad"]:
            print("move '{}' to bad".format(name))
            shutil.move(p, os.path.join(output_folder, name))
        elif src == cls_name and name in err_dict[k]:
            # 将图片移动至目标目录
            # p_s[-2] = dst
            # dst_path = "/".join(p_s)
            if not os.path.exists(os.path.join(output_folder, dst)):
                os.makedirs(os.path.join(output_folder, dst))
            print("move '{}' to {}".format(name, dst))
            shutil.move(p, os.path.join(output_folder, dst, name))
