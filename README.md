# NerveNet

Our project segments out the nerve structures in X-ray medical images.

Input:
![alt text](https://github.com/karangrewal/NerveNet/blob/master/raw_img/input.tif "Input")

Input:
![alt text](https://github.com/karangrewal/NerveNet/blob/master/raw_img/output.png "Output")

### Convolutional Neural Networks

<Enter Description>

### Setup

Clone all files into your local repository. Your training and testing images should be in two different directories which share a common path.

```
<path-to-your-folder>
	train/
	test/
```

The `train/` and `test/` folders should contain all the training and testing images respectively.

Ensure `config.cfg` is in the same folder as `data.py` and set the `path` variable to be the directory which contains the `train/` and `test/` folders.

```
path=<path-to-your-folder>
```

### Training

To begin training, execute `segment.sh`. You must specify the network architecture, batch size, number of epochs to train for and learning rate that you desire.

Example:

```bash
$ ./segment.sh alexnet 20 50 0.001
```

Once the training is complete, the network will make predictions on the test data. The segmented images will be output in a directory called `output/` in the same directory that contains `train.py`.

### Results

Our best training CNN achieved a Dice Coefficient of 0.57 as part of [this Kaggle competition](https://www.kaggle.com/c/ultrasound-nerve-segmentation).
We wrote a summary of our experiment in `results/NerveNet.pdf`.