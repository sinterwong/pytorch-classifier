import torch
from models.get_network import build_network_by_name
import config as cfg
import cv2
import numpy as np
import shutil
import os
from imutils import paths


class Inference():
    def __init__(self, model, ckpt, input_size, device='cuda'):
        self.input_size = input_size
        self.device = device
        self.net = build_network_by_name(model, None, num_classes=len(cfg.classes), deploy=True)
        self._load_model(ckpt)

    def _load_model(self, ckpt):
        model_info = torch.load(ckpt)
        self.net.load_state_dict(model_info["net"])
        self.net = self.net.to(self.device)
        self.net.eval()

    def _preprocess(self, frame):
        frame = cv2.resize(frame, (self.input_size[1], self.input_size[0])).astype(np.float32)
        frame /= 255.0
        data = torch.from_numpy(np.expand_dims(frame.transpose([2, 0, 1]), axis=0)).to(self.device)
        return data

    def _single_image(self, frame):
        h, w, _ = frame.shape
        data = self._preprocess(frame)
        with torch.no_grad():
            output = self.net(data).to('cpu').numpy()
        output = np.argmax(output.squeeze(0))

        return output

    def single_image(self, im_path):
        image = cv2.imread(im_path)[:, :, ::-1]  # bgr to rgb
        return self._single_image(image)


def main():
    classes = ('one', 'five', 'fist', 'ok', 'heartSingle', 'yearh', 'three',
            'four', 'six', 'Iloveyou', 'gun', 'thumbUp', 'nine', 'pink')
    idx2classes = dict(enumerate(classes))
    model = 'seresnet18'
    model_path = 'checkpoint/hand14c/seresnet18/seresnet18_hand14c_128x128_96.286.pth'
    engine = Inference(model, model_path, [128, 128])

    im_path = 'data/demo.jpg'

    print(idx2classes[engine.single_image(im_path)])


if __name__ == "__main__":
    main()
