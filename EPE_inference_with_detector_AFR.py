import os
import cv2
import numpy as np
from skimage.io import imread
from skimage.transform import estimate_transform, warp

from EPE_detectors.AFR import run_AFR


#imagepath = "./res/inp_images/nrr_00000.jpg"
imagedir = "./res/inp_images"
outdir = "./res/test_res_AFR/"

def infer_image(img, outdir, imagename):
	h, w, _ = img.shape
	bbox, kpt = run_AFR(image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB), return_kpt=True)

	if bbox:
		cv2.rectangle(img, pt1=(bbox[1], bbox[0]), pt2=(bbox[3], bbox[2]), color=(0,255,0), thickness=4)

	if kpt:
		for key in kpt:
			for pt in kpt[key]:
				cv2.circle(img, center=pt, radius = 4, color = (128, 0, 128), thickness=-1)

	os.makedirs(outdir, exist_ok=True)
	cv2.imwrite(os.path.join(outdir, os.path.split(imagename)[1]), img)

imgfiles = sorted(os.listdir(imagedir))
L = len(imgfiles)
for iif, imgfile in enumerate(imgfiles):
	img = cv2.imread(os.path.join(imagedir, imgfile))
	infer_image(img=img, outdir=outdir, imagename=imgfile)
	print("{}/{} ready".format(iif + 1, L))
print("Done!")
