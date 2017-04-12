"""Script to train a model to spot cat faces in images."""
from __future__ import print_function, division
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import imgaug as ia
from imgaug import augmenters as iaa
from scipy import misc
import time
import cPickle as pickle
import numpy as np
import cv2
from create_dataset import Example
from plotting import History, LossPlotter
from common import draw_heatmap
from model import Model, Model2
import multiprocessing
import threading
import math
import random
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

torch.backends.cudnn.benchmark = True

GPU = 0 # id of the gpu to use
BATCH_SIZE_TRAIN = 32 # training batch size
BATCH_SIZE_VAL = 32 # validation batch size
BATCHES_PER_VAL = 40 # average validation loss over N batches per validation
GRID_SIZE = 28 # output heatmap size
VAL_EVERY = 250 # validate every N batches
SAVE_EVERY = 250 # save model every N batches
PLOT_EVERY = 250 # plot loss curve every N batches
TRAIN_WINDOW_NAME = "trainwin" # window/file name of training batch debug output
VAL_WINDOW_NAME = "valwin" # window/file name of validation batch debug output
NB_BATCHES = 30000 # train for N batches
SHOW_DEBUG_WINDOWS = False # whether to show example outputs in windows (True)
                           # or write to files instead (False)

def main():
    # load datsets, create it via create_dataset.py
    with open("cats-dataset.pkl", "r") as f:
        examples = pickle.load(f)
    examples_val = examples[0:1024]
    examples_train = examples[1024:]

    # history of loss values gathered during the experiment
    history = History()
    history.add_group("loss", ["train", "val"], increasing=False)

    # object to generate loss plots
    loss_plotter = LossPlotter(
        history.get_group_names(),
        history.get_groups_increasing(),
        save_to_fp="plot.jpg"
    )
    loss_plotter.start_batch_idx = 100

    # load model, loss and stochastic optimizer
    model = Model2()
    if GPU >= 0:
        model.cuda(GPU)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())

    # initialize image augmentation cascade
    rarely = lambda aug: iaa.Sometimes(0.1, aug)
    sometimes = lambda aug: iaa.Sometimes(0.25, aug)
    often = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential([
            iaa.Fliplr(0.5), # horizontally flip 50% of all images
            iaa.Flipud(0.5), # vertically flip 50% of all images
            rarely(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
            often(iaa.Crop(percent=(0, 0.1))), # crop images by 0-10% of their height/width
            sometimes(iaa.GaussianBlur((0, 3.0))), # blur images with a sigma between 0 and 3.0
            sometimes(iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))), # sharpen images
            sometimes(iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0))), # emboss images
            # search either for all edges or for directed edges
            rarely(iaa.Sometimes(0.5,
                iaa.EdgeDetect(alpha=(0, 0.7)),
                iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
            )),
            often(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.2), per_channel=0.5)), # add gaussian noise to images
            often(iaa.Dropout((0.0, 0.1), per_channel=0.5)), # randomly remove up to 10% of the pixels
            rarely(iaa.Invert(0.25, per_channel=True)), # invert color channels
            often(iaa.Add((-10, 10), per_channel=0.5)), # change brightness of images (by -10 to 10 of original value)
            often(iaa.Multiply((0.5, 1.5), per_channel=0.25)), # change brightness of images (50-150% of original value)
            often(iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5)), # improve or worsen the contrast
            sometimes(iaa.Grayscale(alpha=(0.0, 1.0))),
            often(iaa.Affine(
                scale={"x": (0.6, 1.4), "y": (0.6, 1.4)}, # scale images to 60-140% of their size, individually per axis
                translate_percent={"x": (-0.3, 0.3), "y": (-0.3, 0.3)}, # translate by -30 to +30% percent (per axis)
                rotate=(-45, 45), # rotate by -45 to +45 degrees
                shear=(-16, 16), # shear by -16 to +16 degrees
                order=[0, 1], # use any of scikit-image's interpolation methods
                cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                mode=["constant", "edge"] # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)) # apply elastic transformations with random strengths
        ],
        random_order=True # do all of the above in random order
    )

    # method to generate batches
    def load_training_batch():
        examples_batch = random.sample(examples_train, BATCH_SIZE_TRAIN)
        images = [ex.image for ex in examples_batch]
        bb_coords = [ia.KeypointsOnImage(ex.get_bb_coords_keypoints(), shape=image.shape) for ex, image in zip(examples_batch, images)]
        return Batch(identifiers=None, images=images, keypoints=bb_coords)

    img_loader = ImageLoader(load_training_batch, nb_workers=1)
    bg_augmenter = BackgroundAugmenter(seq, img_loader.queue, nb_workers=8)

    # training loop
    for batch_idx in range(NB_BATCHES):
        # load training batch
        time_cbatch = time.time()
        batch = bg_augmenter.get_batch()
        inputs, outputs_gt = images_coords_to_batch(batch.images_aug, batch.keypoints_aug)
        time_cbatch = time.time() - time_cbatch

        # train on batch
        time_fwbw = time.time()
        loss = run_batch(inputs, outputs_gt, model, criterion, optimizer, train=True)
        time_fwbw = time.time() - time_fwbw
        print("[T] %06d | loss=%.4f | %.4fs cbatch | %.4fs fwbw" % (batch_idx, loss, time_cbatch, time_fwbw))
        history.add_value("loss", "train", batch_idx, loss)

        # every N batches, show the true and generated outputs for the training
        # batch
        if batch_idx % 50 == 0:
            update_window(TRAIN_WINDOW_NAME, inputs[0:8], outputs_gt[0:8], model)

        # Code to generate a video of the training progress
        #time_vid_start = time.time()
        #grid = generate_video_image(batch_idx, examples_val[0:66], model)
        #misc.imsave("training-video/%05d.jpg" % (batch_idx,), grid)

        # every N batches, validate
        if (batch_idx+1) % VAL_EVERY == 0:
            # the validation computes an average over N randomly picked batches
            time_cbatch_total = 0
            time_fwbw_total = 0
            loss_total = 0
            for i in range(BATCHES_PER_VAL):
                # load batch
                time_cbatch = time.time()
                examples_batch = random.sample(examples_val, BATCH_SIZE_VAL)
                inputs, outputs_gt = examples_to_batch(examples_batch, iaa.Noop())
                time_cbatch = time.time() - time_cbatch
                time_cbatch_total += time_cbatch

                # validate on batch (forward + loss)
                time_fwbw = time.time()
                loss = run_batch(inputs, outputs_gt, model, criterion, optimizer, train=False)
                time_fwbw = time.time() - time_fwbw
                time_fwbw_total += time_fwbw
                loss_total += loss
            loss_total_avg = loss_total / BATCHES_PER_VAL

            # check if average loss of val batches was best value so far
            if len(history.line_groups["loss"].lines["val"].ys) == 0:
                is_new_best = True
            else:
                minval = np.min(history.line_groups["loss"].lines["val"].ys)
                is_new_best = (loss_total_avg < minval)

            print("[V] %06d | loss=%.4f | %.4fs cbatch | %.4fs fwbw" % (batch_idx, loss_total_avg, time_cbatch_total, time_fwbw_total))
            history.add_value("loss", "val", batch_idx, loss_total_avg)

            # show true and generated outputs for the first 8 validation
            # examples
            inputs, outputs_gt = examples_to_batch(examples_val[0:8], iaa.Noop())
            update_window(VAL_WINDOW_NAME, inputs, outputs_gt, model)

            # save a checkpoint if the model was the best one so far
            if is_new_best:
                torch.save({
                    "batch_idx": batch_idx,
                    "history": history.to_string(),
                    "state_dict": model.state_dict()
                }, "model.best.tar")

        # every N batches, save a checkpoint
        if (batch_idx+1) % SAVE_EVERY == 0:
            torch.save({
                "batch_idx": batch_idx,
                "history": history.to_string(),
                "state_dict": model.state_dict()
            }, "model.tar")

        # every N batches, plot the current loss curve
        if (batch_idx+1) % PLOT_EVERY == 0:
            loss_plotter.plot(history)

def generate_video_image(batch_idx, examples, model):
    """Generate frames for a video of the training progress.
    Each frame contains N examples shown in a grid. Each example shows
    the input image and the main heatmap predicted by the model."""
    start_time = time.time()
    #print("A", time.time() - start_time)
    model.eval()

    # fw through network
    inputs, outputs_gt = examples_to_batch(examples, iaa.Noop())
    inputs_torch = torch.from_numpy(inputs)
    inputs_torch = Variable(inputs_torch, volatile=True)
    if GPU >= 0:
        inputs_torch = inputs_torch.cuda(GPU)
    outputs_pred_torch = model(inputs_torch)
    #print("B", time.time() - start_time)

    outputs_pred = outputs_pred_torch.cpu().data.numpy()
    inputs = (inputs * 255).astype(np.uint8).transpose(0, 2, 3, 1)
    #print("C", time.time() - start_time)
    heatmaps = []
    for i in range(inputs.shape[0]):
        hm_drawn = draw_heatmap(inputs[i], np.squeeze(outputs_pred[i][0]), alpha=0.5)
        heatmaps.append(hm_drawn)
    #print("D", time.time() - start_time)
    grid = ia.draw_grid(heatmaps, cols=11, rows=6).astype(np.uint8)
    #grid_rs = misc.imresize(grid, (720-32, 1280-32))
    # pad by 42 for the text and to get the image to 720p aspect ratio
    grid_pad = np.pad(grid, ((0, 42), (0, 0), (0, 0)), mode="constant")
    grid_pad_text = ia.draw_text(
        grid_pad,
        x=grid_pad.shape[1]-220,
        y=grid_pad.shape[0]-35,
        text="Batch %05d" % (batch_idx,),
        color=[255, 255, 255]
    )
    #print("E", time.time() - start_time)
    return grid_pad_text

def update_window(win, inputs, outputs_gt, model):
    """Show true and generated outputs/heatmaps for example images."""
    model.eval()

    # prepare inputs and forward through network
    inputs, outputs_gt = torch.from_numpy(inputs), torch.from_numpy(outputs_gt)
    inputs, outputs_gt = Variable(inputs), Variable(outputs_gt)
    if GPU >= 0:
        inputs = inputs.cuda(GPU)
        outputs_gt = outputs_gt.cuda(GPU)
    outputs_pred = model(inputs)

    # draw rows of resulting image
    rows = []
    for i in range(inputs.size()[0]):
        # image, ground truth outputs, predicted outputs
        img_np = (inputs[i].cpu().data.numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
        hm_gt_np = outputs_gt[i].cpu().data.numpy()
        hm_pred_np = outputs_pred[i].cpu().data.numpy()

        # per image
        #   first row: ground truth outputs,
        #   second row: predicted outputs
        # each row starts with the input image, followed by heatmap images
        row_truth = [img_np] + [draw_heatmap(img_np, np.squeeze(hm_gt_np[hm_idx]), alpha=0.5) for hm_idx in range(hm_gt_np.shape[0])]
        row_pred = [img_np] + [draw_heatmap(img_np, np.squeeze(hm_pred_np[hm_idx]), alpha=0.5) for hm_idx in range(hm_pred_np.shape[0])]

        rows.append(np.hstack(row_truth))
        rows.append(np.hstack(row_pred))
    grid = np.vstack(rows)

    if SHOW_DEBUG_WINDOWS:
        # show grid in opencv window
        if cv2.getWindowProperty(win, 0) == -1:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win, 1200, 600)
            time.sleep(1)
        cv2.imshow(win, grid.astype(np.uint8)[:, :, ::-1])
        cv2.waitKey(10)
    else:
        # save grid to file
        misc.imsave("window_%s.jpg" % (win,), grid.astype(np.uint8))

def examples_to_batch(examples, seq=None):
    """Convert examples from the dataset to inputs and ground truth outputs
    for the model.
    """
    if seq is None:
        seq = iaa.Noop()
    seq_det = seq.to_deterministic()

    inputs = [ex.image for ex in examples]
    inputs_aug = seq_det.augment_images(inputs)

    bb_coords = [ex.get_bb_coords_keypoints(seq_det) for ex in examples]
    return images_coords_to_batch(inputs_aug, bb_coords)

def images_coords_to_batch(images, bb_coords):
    """Convert input images and bounding box coordinates to expected
    inputs and outputs for the model."""
    bb_grids = [bb_coords_to_grid(bb_coords_one, img.shape, GRID_SIZE) for img, bb_coords_one in zip(images, bb_coords)]
    outputs_gt = bb_grids

    inputs = (np.array(images)/255.0).astype(np.float32).transpose(0, 3, 1, 2)
    outputs_gt = np.array(outputs_gt).astype(np.float32).transpose(0, 3, 1, 2)

    return inputs, outputs_gt

def bb_coords_to_grid(bb_coords_one, img_shape, grid_size):
    """Convert bounding box coordinates (corners) to ground truth heatmaps."""
    if isinstance(bb_coords_one, ia.KeypointsOnImage):
        bb_coords_one = bb_coords_one.keypoints

    # bb edges after augmentation
    x1b = min([kp.x for kp in bb_coords_one])
    x2b = max([kp.x for kp in bb_coords_one])
    y1b = min([kp.y for kp in bb_coords_one])
    y2b = max([kp.y for kp in bb_coords_one])

    # clip
    x1c = np.clip(x1b, 0, img_shape[1]-1)
    y1c = np.clip(y1b, 0, img_shape[0]-1)
    x2c = np.clip(x2b, 0, img_shape[1]-1)
    y2c = np.clip(y2b, 0, img_shape[0]-1)

    # project
    x1d = int((x1c / img_shape[1]) * grid_size)
    y1d = int((y1c / img_shape[0]) * grid_size)
    x2d = int((x2c / img_shape[1]) * grid_size)
    y2d = int((y2c / img_shape[0]) * grid_size)

    assert 0 <= x1d < grid_size
    assert 0 <= y1d < grid_size
    assert 0 <= x2d < grid_size
    assert 0 <= y2d < grid_size

    # output ground truth:
    # - 1 heatmap that is 1 everywhere where there is a bounding box
    # - 9 position sensitive heatmaps,
    #   e.g. the first one is 1 everywhere where there is the _top left corner_
    #        of a bounding box,
    #        the second one is 1 for the top center cell,
    #        the third one is 1 for the top right corner,
    #        ...
    grids = np.zeros((grid_size, grid_size, 1+9), dtype=np.float32)
    # first heatmap
    grids[y1d:y2d+1, x1d:x2d+1, 0] = 1
    # position sensitive heatmaps
    nb_cells_x = 3
    nb_cells_y = 3
    cell_width = (x2d - x1d) / nb_cells_x
    cell_height = (y2d - y1d) / nb_cells_y
    cell_counter = 0
    for j in range(nb_cells_y):
        cell_y1 = y1d + cell_height * j
        cell_y2 = cell_y1 + cell_height
        cell_y1_int = np.clip(int(math.floor(cell_y1)), 0, img_shape[0]-1)
        cell_y2_int = np.clip(int(math.floor(cell_y2)), 0, img_shape[0]-1)
        for i in range(nb_cells_x):
            cell_x1 = x1d + cell_width * i
            cell_x2 = cell_x1 + cell_width
            cell_x1_int = np.clip(int(math.floor(cell_x1)), 0, img_shape[1]-1)
            cell_x2_int = np.clip(int(math.floor(cell_x2)), 0, img_shape[1]-1)
            grids[cell_y1_int:cell_y2_int+1, cell_x1_int:cell_x2_int+1, 1+cell_counter] = 1
            cell_counter += 1
    return grids

def run_batch(inputs, outputs_gt, model, criterion, optimizer, train):
    """Train or validate on a batch (inputs + outputs)."""
    if train:
        model.train()
    else:
        model.eval()
    val = not train
    inputs, outputs_gt = torch.from_numpy(inputs), torch.from_numpy(outputs_gt)
    inputs, outputs_gt = Variable(inputs, volatile=val), Variable(outputs_gt)
    if GPU >= 0:
        inputs = inputs.cuda(GPU)
        outputs_gt = outputs_gt.cuda(GPU)
    if train:
        optimizer.zero_grad()
    outputs_pred = model(inputs)
    loss = criterion(outputs_pred, outputs_gt)
    if train:
        loss.backward()
        optimizer.step()
    return loss.data[0]

class Batch(object):
    """Class encapsulating a batch before and after augmentation."""
    def __init__(self, identifiers, images, keypoints):
        self.identifiers = identifiers
        self.images = images
        self.images_aug = None
        # keypoints here are the corners of the bounding box
        self.keypoints = keypoints
        self.keypoints_aug = None

class ImageLoader(object):
    """Class to load batches in the background."""

    def __init__(self, load_batch_func, nb_workers=1, queue_size=50, threaded=True):
        self.queue = multiprocessing.Queue(queue_size)
        self.workers = []
        for i in range(nb_workers):
            if threaded:
                worker = threading.Thread(target=self._load_batches, args=(load_batch_func, self.queue))
            else:
                worker = multiprocessing.Process(target=self._load_batches, args=(load_batch_func, self.queue))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)

    def _load_batches(self, load_batch_func, queue):
        while True:
            queue.put(pickle.dumps(load_batch_func(), protocol=-1))

class BackgroundAugmenter(object):
    """Class to augment batches in the background (while training on
    the GPU)."""
    def __init__(self, augseq, queue_source, nb_workers, queue_size=50, threaded=False):
        assert 0 < queue_size <= 10000
        self.augseq = augseq
        self.queue_source = queue_source
        self.queue_result = multiprocessing.Queue(queue_size)
        self.workers = []
        for i in range(nb_workers):
            augseq.reseed()
            if threaded:
                worker = threading.Thread(target=self._augment_images_worker, args=(self.augseq, self.queue_source, self.queue_result))
            else:
                worker = multiprocessing.Process(target=self._augment_images_worker, args=(self.augseq, self.queue_source, self.queue_result))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)

    def get_batch(self):
        """Returns a batch from the queue of augmented batches."""
        batch_str = self.queue_result.get()
        batch = pickle.loads(batch_str)
        return batch

    def _augment_images_worker(self, augseq, queue_source, queue_result):
        """Worker function that endlessly queries the source queue (input
        batches), augments batches in it and sends the result to the output
        queue."""
        while True:
            # wait for a new batch in the source queue and load it
            batch_str = queue_source.get()
            batch = pickle.loads(batch_str)

            # augment the batch
            if batch.images is not None and batch.keypoints is not None:
                augseq_det = augseq.to_deterministic()
                batch.images_aug = augseq_det.augment_images(batch.images)
                batch.keypoints_aug = augseq_det.augment_keypoints(batch.keypoints)
            elif batch.images is not None:
                batch.images_aug = augseq.augment_images(batch.images)
            elif batch.keypoints is not None:
                batch.keypoints_aug = augseq.augment_keypoints(batch.keypoints)

            # send augmented batch to output queue
            queue_result.put(pickle.dumps(batch, protocol=-1))

if __name__ == "__main__":
    main()
