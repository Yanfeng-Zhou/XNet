import numpy as np
from PIL import Image
import pywt
import argparse
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', default='//10.0.5.233/shared_data/XNet/dataset/CREMI/train_unsup_80/image')
    parser.add_argument('--wavelet_path', default='//10.0.5.233/shared_data/XNet/dataset/CREMI/train_unsup_80/DB2_H')
    parser.add_argument('--wavelet_type', default='db2', help='haar, db2, bior1.5, bior2.4, coif1, dmey')
    parser.add_argument('--if_RGB', default=False)
    args = parser.parse_args()

    if not os.path.exists(args.wavelet_path):
        os.mkdir(args.wavelet_path)

    for i in os.listdir(args.image_path):
        image_path = os.path.join(args.image_path, i)
        wavelet_path = os.path.join(args.wavelet_path, i)

        if args.if_RGB:
            image = Image.open(image_path).convert('L')
        else:
            image = Image.open(image_path)
        image = np.array(image)

        LL, (LH, HL, HH) = pywt.dwt2(image, args.wavelet_type)

        LL = (LL - LL.min()) / (LL.max() - LL.min()) * 255

        LH = (LH - LH.min()) / (LH.max() - LH.min()) * 255
        HL = (HL - HL.min()) / (HL.max() - HL.min()) * 255
        HH = (HH - HH.min()) / (HH.max() - HH.min()) * 255

        merge1 = HH + HL + LH
        merge1 = (merge1-merge1.min()) / (merge1.max()-merge1.min()) * 255

        merge1 = Image.fromarray(merge1.astype(np.uint8))
        merge1.save(wavelet_path)

