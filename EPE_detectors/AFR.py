import os
import cv2
import numpy as np
import face_recognition

def run_AFR(image, return_kpt=False):
	'''
	image: 0-255, uint8, rgb, [h, w, 3]
	return_kpt: bool
	return: detected box list
	'''
	bbox = face_recognition.face_locations(image, model="cnn")
	if len(bbox) > 1:
		print("There is more than one face in the image! (bbox)")
		#exit()
		bbox = bbox[0]
	elif len(bbox) == 0:
		bbox = None
	else:
		bbox = bbox[0]

	if return_kpt:
		kpt = face_recognition.face_landmarks(image)
		if len(kpt) > 1:
			print("There is more than one face in the image! (kpt)")
			#exit()
			kpt = kpt[0]
		elif len(kpt) == 0:
			kpt = None
		else:
			kpt = kpt[0]
		return bbox, kpt

	return bbox

def return_bbox_by_points(image, return_image_result=False):
	'''
	image: 0-255, uint8, rgb, [h, w, 3]
	return_image_result: bool
	return: detected box list
	'''
	h, w, _ = image.shape
	kpt = face_recognition.face_landmarks(image)
	if len(kpt) > 1:
		print("There is more than one face in the image! (kpt)")
		#exit()
		kpt = kpt[0]
	elif len(kpt) == 0:
		kpt = None
	else:
		kpt = kpt[0]

	if kpt:
		kpt_array = np.array([pt for pt_group in list(kpt.values()) for pt in pt_group])
		left, right = np.min(kpt_array[:, 0]), np.max(kpt_array[:, 0])
		top, bottom = np.min(kpt_array[:, 1]), np.max(kpt_array[:, 1])
		old_size = (right - left + bottom - top) / 2 * 1.1
		center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
		size = int(old_size * 1.25)
		bbox = (center[0], center[1], size, True)
	else:
		left = 0
		top = 0
		right = h - 1
		bottom = w - 1
		bbox = (0, 0, 0, False)

	if return_image_result:
		res_image = np.copy(image)
		cv2.rectangle(res_image, pt1=(left, top), pt2=(right, bottom), color=(0, 255, 0), thickness=4)
		if kpt:
			for pt in kpt_array:
				cv2.circle(res_image, center=(int(pt[0]), int(pt[1])), radius=4, color=(128, 0, 128), thickness=-1)
		return bbox, res_image
	else:
		return bbox

def return_bbox(image, return_image_result=False):
	h, w, _ = image.shape
	bbox, kpt = run_AFR(image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB), return_kpt=True)

	if bbox:
		cx = (bbox[1] + bbox[3]) / 2
		cy = (bbox[0] + bbox[2]) / 2
		size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
		isFace = True
	else:
		cx, cy, size = 0, 0, 0
		isFace = False

	if return_image_result:
		res_image = np.copy(image)
		if bbox:
			cv2.rectangle(res_image, pt1=(bbox[1], bbox[0]), pt2=(bbox[3], bbox[2]), color=(0, 255, 0), thickness=4)
		if kpt:
			for key in kpt:
				for pt in kpt[key]:
					cv2.circle(res_image, center=pt, radius=4, color=(128, 0, 128), thickness=-1)
		return (cx, cy, size, isFace), res_image
	else:
		return cx, cy, size, isFace
