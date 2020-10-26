import os
import cv2
import numpy as np
import torch.nn as nn


def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


def print_model(model):
    params_num = 0
    for param in model.parameters():
        params_num += param.numel()

    print(model)
    print("Total number of parameters: %d" % params_num)


def video2images(src_path, dst_path, size=(1280, 720), LR_reverse=True):
    os.makedirs(dst_path, exist_ok=True)

    vidcap = cv2.VideoCapture(src_path)
    i = 1
    while vidcap.isOpened():
        ret, img = vidcap.read()
        if int(vidcap.get(1)) % 30 == 0:
            img = cv2.resize(img, dsize=size, interpolation=cv2.INTER_AREA)
            if LR_reverse:
                if int(vidcap.get(1)) % 60 == 0:
                    img = cv2.flip(img, 1)

            dst_name = "image_B_%05d" % (i) + ".jpg"
            dst = os.path.join(dst_path, dst_name)
            cv2.imwrite(dst, img)
            print("%s is saved." % (dst_name))
            i += 1
        if not ret:
            break


def edge_smoothing(src_path, dst_path, size=None):
    os.makedirs(dst_path, exist_ok=True)
    file_list = os.listdir(src_path)

    kernel_size = 7
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    gauss = cv2.getGaussianKernel(kernel_size, 0)
    gauss_kernel = gauss * gauss.transpose(1, 0)
    # [[0.25],                                [[0.0625, 0.125 , 0.0625],
    #  [0.5 ],    *   [[0.25, 0.5, 0.25]] =    [0.125 , 0.25  , 0.125 ],  => gauss
    #  [0.25]]                                 [0.0625, 0.125 , 0.0625]]

    normal_kernel = np.full((kernel_size, kernel_size), 1 / (kernel_size * kernel_size))

    for file_idx, file_name in enumerate(file_list):
        rgb_img = cv2.imread(os.path.join(src_path, file_name))  # channels = 3  (RGB image)
        gray_img = cv2.imread(
            os.path.join(src_path, file_name), 0
        )  # channels = 1  (Gray image)
        if size is not None:
            rgb_img = cv2.resize(rgb_img, size)
            gray_img = cv2.resize(gray_img, size)

        pad_img = np.pad(
            rgb_img,
            (
                (int(kernel_size / 2), int(kernel_size / 2)),
                (int(kernel_size / 2), int(kernel_size / 2)),
                (0, 0),
            ),
            mode="reflect",
        )  # if the size of img is (H, W), the size of pad_img is (H + kernel_size, W + kernel_size).

        edges = cv2.Canny(
            gray_img, 20, 130
        )  # cv2.Canny(image, low_threshold, high_threshold)
        dilation = cv2.dilate(edges, kernel)

        gauss_img = np.copy(rgb_img)
        idx = np.where(dilation != 0)

        for i in range(np.sum(dilation != 0)):
            gauss_img[idx[0][i], idx[1][i], 0] = np.sum(
                np.multiply(
                    # because of that the difference of the shapes that two images,
                    # it's same with rgb_img[idx[0][i] - int(kernel_size / 2) : idx[0][i] + int(kernel_size / 2) + 1,
                    #                        idx[1][i] - int(kernel_size / 2) : idx[1][i] + int(kernel_size / 2) + 1,
                    #                        0]
                    pad_img[
                        idx[0][i] : idx[0][i] + kernel_size,
                        idx[1][i] : idx[1][i] + kernel_size,
                        0,
                    ],
                    gauss_kernel,
                )
            )
            gauss_img[idx[0][i], idx[1][i], 1] = np.sum(
                np.multiply(
                    pad_img[
                        idx[0][i] : idx[0][i] + kernel_size,
                        idx[1][i] : idx[1][i] + kernel_size,
                        1,
                    ],
                    gauss_kernel,
                )
            )
            gauss_img[idx[0][i], idx[1][i], 2] = np.sum(
                np.multiply(
                    pad_img[
                        idx[0][i] : idx[0][i] + kernel_size,
                        idx[1][i] : idx[1][i] + kernel_size,
                        2,
                    ],
                    gauss_kernel,
                )
            )
        cv2.imwrite(
            os.path.join(dst_path, "%s_edge_smoothed.jpg" % (os.path.splitext(file_name)[0])),
            gauss_img,
        )

        print("[%05d/%05d]" % (file_idx + 1, len(file_list)))


if __name__ == "__main__":
    edge_smoothing(src_path="./edge_test/src", dst_path="./edge_test/dst")
