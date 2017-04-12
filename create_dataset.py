"""File to transform the 10K cats dataset to a set of example objects that
are saved as a pickle file and later used by the training."""
from __future__ import print_function, division
from scipy import misc, ndimage
import numpy as np
import os
import argparse
import cStringIO as StringIO
import cPickle as pickle
import math
import imgaug as ia
from common import to_aspect_ratio_add

def main():
    """Generate datasets."""

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--dataset_dir", help="")
    parser.add_argument("--out_dir", default="", help="")
    parser.add_argument("--img_size", default=224, type=int, help="")
    args = parser.parse_args()

    assert args.dataset_dir is not None, "Expected 10k cats dataset directory via --dataset_dir"

    # load images and their facial keypoints
    fps = find_image_filepaths(args.dataset_dir)
    examples = []
    for fp in fps:
        img = ndimage.imread(fp, mode="RGB")
        kps = load_keypoints(fp, image_height=img.shape[0], image_width=img.shape[1])
        img_square, (added_top, added_right, added_bottom, added_left) = to_aspect_ratio_add(img, 1.0, return_paddings=True)
        kps = [(x+added_left, y+added_top) for (x, y) in kps]
        #rs_factor = args.img_size / img_square.shape[0]
        img_square_rs = misc.imresize(img_square, (args.img_size, args.img_size))
        kps_rs = [(int((x/img_square.shape[1])*args.img_size), int((y/img_square.shape[0])*args.img_size)) for (x, y) in kps]
        examples.append(Example(fp, img_square_rs, kps_rs))

    # save datasets
    with open(os.path.join(args.out_dir, "cats-dataset.pkl"), "w") as f:
        pickle.dump(examples, f)

def find_image_filepaths(dataset_dir):
    """Load image filepaths from the 10k cats dataset."""
    result = []
    for root, dirs, files in os.walk(dataset_dir):
        if "/CAT_" in root:
            for name in files:
                fp = os.path.join(root, name)
                if name.endswith(".jpg") and os.path.isfile("%s.cat" % (fp,)):
                    result.append(fp)
    return result

def load_keypoints(image_filepath, image_height, image_width):
    """Load facial keypoints of one image."""
    fp_keypoints = "%s.cat" % (image_filepath,)
    if not os.path.isfile(fp_keypoints):
        raise Exception("Could not find keypoint coordinates for image '%s'." \
                        % (image_filepath,))
    else:
        coords_raw = open(fp_keypoints, "r").readlines()[0].strip().split(" ")
        coords_raw = [abs(int(coord)) for coord in coords_raw]
        keypoints = []
        #keypoints_arr = np.zeros((9*2,), dtype=np.int32)
        for i in range(1, len(coords_raw), 2): # first element is the number of coords
            x = np.clip(coords_raw[i], 0, image_width-1)
            y = np.clip(coords_raw[i+1], 0, image_height-1)
            keypoints.append((x, y))

        return keypoints

def compress_to_jpg(img):
    """Compress an image (numpy array) to jpg."""
    return compress_img(img, method="JPEG")

def compress_img(img, method):
    """Compress an image (numpy array) using the provided image compression
    method."""
    img_compressed_buffer = StringIO.StringIO()
    im = misc.toimage(img)
    im.save(img_compressed_buffer, format=method)
    img_compressed = img_compressed_buffer.getvalue()
    img_compressed_buffer.close()
    return img_compressed

def decompress_img(img_compressed):
    """Decompress a compressed image to a numpy array."""
    img_compressed_buffer = StringIO.StringIO()
    img_compressed_buffer.write(img_compressed)
    img = ndimage.imread(img_compressed_buffer, mode="RGB")
    img_compressed_buffer.close()
    return img

class Example(object):
    """Class that encapsulates an example from the 10k cats dataset.
    It contains an example image (padded and resized) for the training
    and the facial keypoints."""

    def __init__(self, fp, image, keypoints):
        self.fp = fp
        # save images jpg-compressed to save memory
        self._image = compress_to_jpg(image)
        self.keypoints = keypoints

    @property
    def image(self):
        """Return the decompressed image of the example (i.e. numpy array)."""
        return decompress_img(self._image)

    def get_bb_coords_keypoints(self, augseq=None):
        """Get the coordinates of the bounding box appropiate for the
        cat's face."""
        # augment the coordinates if an augmentation sequence is provided
        if augseq is not None:
            keypoints = ia.KeypointsOnImage([], shape=self.image.shape)
            for (x, y) in self.keypoints:
                keypoints.keypoints.append(ia.Keypoint(x=x, y=y))
            keypoints_aug = augseq.augment_keypoints([keypoints])[0]
            keypoints_aug = [(kp.x, kp.y) for kp in keypoints_aug.keypoints]
        else:
            keypoints_aug = self.keypoints

        # the facial keypoints include eyes, nose and ears.
        # drawing a bounding box around them misses the area below the nose.
        # so we extend the BB based on the vectors pointing from the eye to the
        # nose (and extending these a bit to generate new pseudo keypoints).
        left_eye = keypoints_aug[0]
        right_eye = keypoints_aug[1]
        mouth = keypoints_aug[2]

        left_eye_vec = (left_eye[0] - mouth[0], left_eye[1] - mouth[1])
        right_eye_vec = (right_eye[0] - mouth[0], right_eye[1] - mouth[1])

        l = 0.5
        le_point = (int(mouth[0] - l*left_eye_vec[0]), int(mouth[1] - l*left_eye_vec[1]))
        re_point = (int(mouth[0] - l*right_eye_vec[0]), int(mouth[1] - l*right_eye_vec[1]))

        keypoints_aug = keypoints_aug + [le_point, re_point]

        # draw bounding box around facial keypoints and pseudo
        # keypoints (i.e. mouth area)
        bb_x1 = min([x for (x, y) in keypoints_aug])
        bb_x2 = max([x for (x, y) in keypoints_aug])
        bb_y1 = min([y for (x, y) in keypoints_aug])
        bb_y2 = max([y for (x, y) in keypoints_aug])
        #return {"x1": bb_x1, "x2": bb_x2, "y1": bb_y1, "y2": bb_y2}
        return [
            ia.Keypoint(x=bb_x1, y=bb_y1),
            ia.Keypoint(x=bb_x2, y=bb_y1),
            ia.Keypoint(x=bb_x2, y=bb_y2),
            ia.Keypoint(x=bb_x1, y=bb_y2)
        ]

    #def get_bb_coords(self, augseq=None):
    #    kps = self.get_bb_coords_keypoints(augseq=augseq)
    #    return (kps[0].x, kps[0].y, kps[2].x, kps[2].y)

if __name__ == "__main__":
    main()
