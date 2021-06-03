import os
import shutil

image_suffixes = (".jpg", ".png", ".jpeg")

# make a list of files that we need
frames_root = "/disk/sdb1/avatars/dataset_processed"
names_root = "/disk/sdb1/avatars/dataset_EPE_data1"
actor_dirs = sorted([dir for dir in os.listdir(names_root) if os.path.isdir(os.path.join(names_root, dir))])
subdir = "virtual"

La = len(actor_dirs)

for ia, actor_dir in enumerate(actor_dirs):
    full_actor_subdir = os.path.join(names_root, actor_dir, subdir)
    video_dirs = sorted([dir for dir in os.listdir(full_actor_subdir) if os.path.isdir(os.path.join(full_actor_subdir, dir))])
    Lv = len(video_dirs)
    for iv, video_dir in enumerate(video_dirs):
        full_video_dir = os.path.join(full_actor_subdir, video_dir)
        frames = sorted([file for file in os.listdir(os.path.join(full_video_dir, "original_renders", "normal_renders")) if os.path.splitext(file)[1] in image_suffixes])
        Lf = len(frames)
        for i, frame in enumerate(frames):
            if actor_dir == "MelinaJuergens" and subdir == "virtual":
                frame = os.path.splitext(frame)[0] + ".png"
            from_path = os.path.join(frames_root, actor_dir, subdir, video_dir, "" if video_dir == "hb2" else "frames", frame)
            to_path = os.path.join(full_video_dir, "frames", frame)
            assert os.path.isfile(from_path), "The file doesn't exist! {}".format(from_path)
            os.makedirs(os.path.split(to_path)[0], exist_ok=True)
            shutil.copyfile(src=from_path, dst=to_path)
            print("{}/{} --- {}/{} || {}/{} done! [{}]".format(ia+1, La, iv+1, Lv, i+1, Lf, to_path))



