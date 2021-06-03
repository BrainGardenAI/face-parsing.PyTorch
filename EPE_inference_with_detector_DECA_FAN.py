import os
import cv2
import numpy as np
from skimage.io import imread
from skimage.transform import estimate_transform, warp

from EPE_detectors import DECA_FAN


#imagepath = "./res/inp_images/nrr_00000.jpg"
imagedir = "./res/inp_images"
outdir = "./res/test_res_FAN"

def infer_image(image, outdir, imagename):
	h, w, _ = image.shape

	bbox, bbox_type, kpt = DECA_FAN.FAN().run(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), return_kpt=True)

	isFace = False
	if len(bbox) < 4:
		# print('no face detected! run original image')
		left = 0
		top = 0
		right = h - 1
		bottom = w - 1
	# if detector found the face we wanted
	else:
		left = bbox[0]
		top = bbox[1]
		right = bbox[2]
		bottom = bbox[3]
		isFace = True

	if bbox_type == "kpt68":
		old_size = (right - left + bottom - top) / 2 * 1.1
		center = np.rint(np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])).astype(np.int)
	elif bbox_type == "no_detection":
		old_size = 0
		center = np.array([0, 0])
	else:
		print("bbox_type is wrong - {}".format(bbox_type))
		exit()

	size = int(round(old_size * 1.25))
	hsize = int(round(size/2))

	left = center[0] - hsize
	right = center[0] + hsize
	top = center[1] - hsize
	bottom = center[1] + hsize

	# check result
	cv2.rectangle(image, pt1=(left, top), pt2=(right, bottom), color=(0,255,0), thickness=4)
	for pt in kpt:
		cv2.circle(image, center=(int(pt[0]), int(pt[1])), radius = 4, color = (128, 0, 128), thickness=-1)
	os.makedirs(outdir, exist_ok=True)
	cv2.imwrite(os.path.join(outdir, os.path.split(imagename)[1]), image)

imgfiles = sorted(os.listdir(imagedir))
L = len(imgfiles)
for iif, imgfile in enumerate(imgfiles):
	image = cv2.imread(os.path.join(imagedir, imgfile))
	infer_image(image=image, outdir=outdir, imagename=imgfile)
	print("{}/{} ready".format(iif+1, L))
print("Done!")

