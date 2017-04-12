# About

![Example image](images/example-optical-illusion.jpg?raw=true "Example image")
![Example image](images/example-medal-of-honor-cat.jpg?raw=true "Example image")
![Example image](images/example-horror-movie.jpg?raw=true "Example image")
![Example image](images/example-toy-cat.jpg?raw=true "Example image")

This project contains code to train and run a neural network to detect cat faces in videos.
The network uses a pretrained ResNet-18 with á trous trick as its core and adds three additional convolutional layers on top of that.
It predicts heatmaps of face locations and derives bounding boxes from those outputs.
The model does not use an RPN (region proposal network).
Runtime is around 30-60ms per frame on medium hardware (though only ~5ms of that is down to the CNN, so there is a lot of room for improvement).
Implementation is done in PyTorch.

# Videos

Example video of detected bounding boxes:

[![Example video](images/video-bbs.jpg?raw=true)](https://www.youtube.com/watch?v=2FCsQaqW5B8)

Example video of the training progress:

[![Example video training progress](images/video-bbs.jpg?raw=true)](https://www.youtube.com/watch?v=Nply4o_Zgg8)

# Dependencies

* python 2.7 (only tested in that version)
* scipy
* numpy
* scikit-image
* matplotlib
* imgaug (`sudo pip install imgaug`)
* [PyTorch](http://pytorch.org/)
* NVIDIA GPU (might not work without CUDA+CuDNN, not tested), about 8GB (4GB might require to decrease batch sizes)
* Optimized for Ubuntu, may or may not work in other systems.

# Usage

* Download the [10k cats dataset](https://web.archive.org/web/20150520175645/http://137.189.35.203/WebUI/CatDatabase/catData.html) and extract it, e.g. into directory `/foo/bar/10k-cats`. That directory should contain the subdirectories `CAT_00`, `CAT_01`, etc.
* Clone the repository via `git clone https://github.com/aleju/cat-bbs.git`
* Switch into the repository's directory via `cd cat-bbs`
* Create a pickle file of 10k cats via `python create_dataset.py --dataset_dir="/foo/bar/10k-cats"`
* Train a network via `python train.py`
  * This runs for 30k batches, but you can usually stop before that. After 5k batches it is already pretty good.
* Analyze a video via `python predict_video.py --video="/path/to/video.mp4" --conf=0.7 size=400"`
  * `conf` is the confidence threshold of bounding boxes (higher values lead to less bounding boxes shown).
  * `size` is the size of the images to feed through the network (higher value lead to smaller cat faces being spotted).
  * Frames are written to `<repository-directory>/outputs/videos/<video-filename>/%05d.jpg`.
    * You can convert the frames to a video via `cd <repository-directory>/outputs/videos` and then `avconv -i "<video-filename>/%05d.jpg" -b:v 1000k "<video-filename>.mp4"` (you might have to replace `avconv` with `ffmpeg`, depending on what is installed on your system - parameters are the same for both).
