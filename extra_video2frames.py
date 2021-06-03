import os
import cv2

image_suffixes = (".jpg", ".png", ".jpeg")

# make a list of files that we need
input_video = "/disk/sdb1/avatars/video_data/NormanReedus/real/VXG1K-scGY8-seg3.mp4"
root = "/disk/sdb1/avatars/dataset_EPE_data1"
actor = "NormanReedus"
subdir = "real"

out_dir = os.path.join(root, actor, subdir, os.path.splitext(os.path.split(input_video)[1])[0], "frames")
os.makedirs(out_dir, exist_ok=True)

video = cv2.VideoCapture(input_video)
fps = video.get(cv2.CAP_PROP_FPS)
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

# extract frames and save them
count = 0
success = True
while success:
    imagepath = os.path.join(out_dir, "{}.jpg".format(str(count).zfill(max(5, len(str(total_frames))))))
    success, image = video.read()
    if success:
        cv2.imwrite(imagepath, image)  # save frame as JPEG file
        count += 1
    print("{}/{} video frames saved..".format(count, total_frames))
video.release()

