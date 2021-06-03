# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms
# in the LICENSE file included with this software distribution.
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import os
import cv2
import numpy as np
import face_alignment


class FAN(object):
    def __init__(self):
        self.model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

    def run(self, image, return_kpt=False):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box list
        '''
        out = self.model.get_landmarks(image)
        if out is None:
            if return_kpt:
                return [0], 'no_detection', None
            else:
                return [0], 'no_detection'
        else:
            kpt = out[0].squeeze()
            left, right = np.min(kpt[:,0]), np.max(kpt[:,0])
            top, bottom = np.min(kpt[:,1]), np.max(kpt[:,1])
            bbox = [left, top, right, bottom]
            if return_kpt:
                return bbox, 'kpt68', kpt
            else:
                return bbox, 'kpt68'

def return_bbox(image, return_image_result=False):
	h, w, _ = image.shape

	bbox, bbox_type, kpt = FAN().run(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), return_kpt=True)

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
		center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
	elif bbox_type == "no_detection":
		old_size = 0
		center = np.array([0, 0])
	else:
		print("bbox_type is wrong - {}".format(bbox_type))
		exit()

	size = int(old_size * 1.25)
	bbox = (center[0], center[1], size, isFace)

	if return_image_result:
		res_image = np.copy(image)
		if isFace:
			cv2.rectangle(res_image, pt1=(left, top), pt2=(right, bottom), color=(0, 255, 0), thickness=4)
			for pt in kpt:
				cv2.circle(res_image, center=(int(pt[0]), int(pt[1])), radius=4, color=(128, 0, 128), thickness=-1)
		return bbox, res_image
	else:
		return bbox

