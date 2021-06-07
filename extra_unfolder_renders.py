import os
import shutil

image_suffixes = (".jpg", ".png", ".jpeg")

# make a list of files that we need
root = "/disk/sdb1/avatars/dataset_EPE_data1"
subdir = "virtual"

actor_dirs = sorted([dir for dir in os.listdir(root) if os.path.isdir(os.path.join(root, dir))])
La = len(actor_dirs)

# move files from render_dir_to_remove
for ia, actor_dir in enumerate(actor_dirs):
    full_actor_subdir = os.path.join(root, actor_dir, subdir)
    video_dirs = sorted([dir for dir in os.listdir(full_actor_subdir) if os.path.isdir(os.path.join(full_actor_subdir, dir))])
    Lv = len(video_dirs)
    for iv, video_dir in enumerate(video_dirs):
        full_videodir = os.path.join(full_actor_subdir, video_dir)
        from_dir = os.path.join(full_videodir, "original_renders")
        print("Act {}/{} ({}), Vid {}/{} () || copying renders.. ".format(ia+1, La, actor_dir, iv+1, Lv, video_dir),
              end='')
        shutil.copytree(src=from_dir, dst=full_videodir, dirs_exist_ok=True)
        print("done!")

