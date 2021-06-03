# face-parsing.PyTorch

<p align="center">
	<a href="https://github.com/zllrunning/face-parsing.PyTorch">
    <img class="page-image" src="https://github.com/zllrunning/face-parsing.PyTorch/blob/master/6.jpg" >
	</a>
</p>

Our fork scripts:
- `EPE_seg_infer.py` - takes input directory with images, predicts segmentation masks for them and saves masks to the given output directory
- `EPE_inference_with_detector_*` - scripts for testing detectors, takes input directory with images and saves detection results to given output_directory
- `extra_video2frames.py` - extract frames from given video and save them to "frames" directory in our dataset structure
- `extra_copyframes_afterwards.py` - copy frames from one dataset to another, both datasets must have our dataset structure


### Contents
- [Training](#training)
- [Demo](#Demo)
- [References](#references)

## Training

1. Prepare training data:
    -- download [CelebAMask-HQ dataset](https://github.com/switchablenorms/CelebAMask-HQ)

	--  change file path in the `prepropess_data.py`  and run
```Shell
python prepropess_data.py
```

2. Train the model using CelebAMask-HQ dataset:
Just run the train script: 
```
    $ CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
```

If you do not wish to train the model, you can download [our pre-trained model](https://drive.google.com/open?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812) and save it in `res/cp`.


## Demo
1. Evaluate the trained model using:
```Shell
# evaluate using GPU
python test.py
```

## Face makeup using parsing maps
[**face-makeup.PyTorch**](https://github.com/zllrunning/face-makeup.PyTorch)
<table>

<tr>
<th>&nbsp;</th>
<th>Hair</th>
<th>Lip</th>
</tr>

<!-- Line 1: Original Input -->
<tr>
<td><em>Original Input</em></td>
<td><img src="makeup/116_ori.png" height="256" width="256" alt="Original Input"></td>
<td><img src="makeup/116_lip_ori.png" height="256" width="256" alt="Original Input"></td>
</tr>

<!-- Line 3: Color -->
<tr>
<td>Color</td>
<td><img src="makeup/116_1.png" height="256" width="256" alt="Color"></td>
<td><img src="makeup/116_3.png" height="256" width="256" alt="Color"></td>
</tr>

</table>


## References
- [BiSeNet](https://github.com/CoinCheung/BiSeNet)