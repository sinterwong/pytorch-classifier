import os
import cv2
import numpy as np
from imutils import paths
import tqdm

data_folder = "/home/wangxt/datasets/scp_data0825/train/black"
output_folder = "/home/wangxt/datasets/scp_data0825/train/white"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

data_paths = list(paths.list_images(data_folder))

for i, p in tqdm.tqdm(enumerate(data_paths), total=len(data_paths)):
    img = cv2.imread(p)
    new_img = np.full_like(img, 255)
    new_img -= img
    cv2.imwrite(os.path.join(output_folder, os.path.basename(p)), new_img)



