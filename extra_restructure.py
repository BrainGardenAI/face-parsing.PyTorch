import os
import shutil

image_suffixes = (".jpg", ".png", ".jpeg")

# make a list of files that we need
root = "/disk/sdb1/avatars/dataset_EPE_data1"

actor_dirs = sorted([dir for dir in os.listdir(root) if os.path.isdir(os.path.join(root, dir))])
La = len(actor_dirs)

os.makedirs(os.path.join(root, "virtual"))
os.makedirs(os.path.join(root, "real"))

# move files from render_dir_to_remove
for ia, actor_dir in enumerate(actor_dirs):
    for subdir in ("virtual", "real"):
        full_actor_subdir = os.path.join(root, actor_dir, subdir)
        video_dirs = sorted([dir for dir in os.listdir(full_actor_subdir) if os.path.isdir(os.path.join(full_actor_subdir, dir))])
        Lv = len(video_dirs)
        for iv, video_dir in enumerate(video_dirs):
            to_videodir = os.path.join(root, subdir, f"{actor_dir}__{video_dir}")
            from_videodir = os.path.join(full_actor_subdir, video_dir)
            print(f"Sub {subdir} | Act {ia+1}/{La} ({actor_dir}), Vid {iv+1}/{Lv} ({video_dir}) || copying renders.. ",
                  end='')
            shutil.copytree(src=from_videodir, dst=to_videodir, dirs_exist_ok=True)
            print("done!")

