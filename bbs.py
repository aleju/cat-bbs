from __future__ import print_function, division
import os
import numpy as np
import random
from scipy import ndimage, misc
import pickle
import json
from skimage import draw
import copy

class Rectangle(object):
    def __init__(self, y1, x1, y2, x2):
        if y1 < y2:
            self.y1 = y1
            self.y2 = y2
        else:
            self.y1 = y2
            self.y2 = y1
        if x1 < x2:
            self.x1 = x1
            self.x2 = x2
        else:
            self.x1 = x2
            self.x2 = x1
        #self.width = self.x2 - self.x1
        #self.height = self.y2 - self.y1

    def copy(self, x1=None, x2=None, y1=None, y2=None):
        rect = copy.deepcopy(self)
        if x1 is not None:
            rect.x1 = x1
        if x2 is not None:
            rect.x2 = x2
        if y1 is not None:
            rect.y1 = y1
        if y2 is not None:
            rect.y2 = y2
        return rect

    @staticmethod
    def from_dlib_rect(rect):
        return Rectangle(x1=rect.left(), x2=rect.right(), y1=rect.top(), y2=rect.bottom())

    def to_dlib_rect(self):
        import dlib
        return dlib.rectangle(left=self.x1, right=self.x2, top=self.y1, bottom=self.y2)

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def center_x(self):
        return self.x1 + self.width/2

    @property
    def center_y(self):
        return self.y1 + self.height/2

    @property
    def area(self):
        return self.height * self.width

    def is_identical_with(self, other_rect):
        return self == other_rect

    def overlaps_with(self, other_rect):
        assert isinstance(other_rect, Rectangle)
        return self.area_intersection(other_rect) > 0

    def area_intersection(self, other_rect):
        assert isinstance(other_rect, Rectangle)
        return max(0, min(self.x2, other_rect.x2) - max(self.x1, other_rect.x1)) * max(0, min(self.y2, other_rect.y2) - max(self.y1, other_rect.y1))

    def area_union(self, other_rect):
        assert isinstance(other_rect, Rectangle)
        return self.area + other_rect.area - self.area_intersection(other_rect)

    def intersection_over_union(self, other_rect):
        assert isinstance(other_rect, Rectangle)
        area_intersection = self.area_intersection(other_rect)
        area_union = self.area_union(other_rect)
        return area_intersection / area_union

    def iou(self, other_rect):
        return self.intersection_over_union(other_rect)

    def is_fully_within_image(self, img):
        height, width = img.shape[0], img.shape[1]
        return self.x1 >= 0 and self.x2 < width and self.y1 >= 0 and self.y2 < height

    def fix_by_image_dimensions(self, height, width=None):
        if isinstance(height, (tuple, list)):
            assert width is None
            height, width = height[0], height[1]
        elif isinstance(height, (np.ndarray, np.generic)):
            assert width is None
            height, width = height.shape[0], height.shape[1]
        else:
            assert width is not None
            assert isinstance(height, int)
            assert isinstance(width, int)

        self.x1 = int(np.clip(self.x1, 0, width-1))
        self.x2 = int(np.clip(self.x2, 0, width-1))
        self.y1 = int(np.clip(self.y1, 0, height-1))
        self.y2 = int(np.clip(self.y2, 0, height-1))

        if self.x1 > self.x2:
            self.x1, self.x2 = self.x2, self.x1
        if self.y1 > self.y2:
            self.y1, self.y2 = self.y2, self.y1

        if self.x1 == self.x2:
            if self.x1 > 0:
                self.x1 = self.x1 - 1
            else:
                self.x2 = self.x2 + 1

        if self.y1 == self.y2:
            if self.y1 > 0:
                self.y1 = self.y1 - 1
            else:
                self.y2 = self.y2 + 1

        #self.width = self.x2 - self.x1
        #self.height = self.y2 - self.y1

    def resize(self, from_img, to_img):
        if isinstance(from_img, tuple):
            from_height, from_width = from_img[0], from_img[1]
        else:
            from_height, from_width = from_img.shape[0], from_img.shape[1]
        if isinstance(to_img, tuple):
            to_height, to_width = to_img[0], to_img[1]
        else:
            to_height, to_width = to_img.shape[0], to_img.shape[1]

        x1 = int((self.x1 / from_width) * to_width)
        x2 = int((self.x2 / from_width) * to_width)
        y1 = int((self.y1 / from_height) * to_height)
        y2 = int((self.y2 / from_height) * to_height)

        return Rectangle(x1=x1, y1=y1, x2=x2, y2=y2)

    def merge(self, other_rect):
        return Rectangle(x1=min(self.x1, other_rect.x1), x2=max(self.x2, other_rect.x2), y1=min(self.y1, other_rect.y1), y2=max(self.y2, other_rect.y2))

    def to_squared(self, border=0.0, keep_within_shape=None):
        size = max(self.height, self.width)
        if keep_within_shape is None:
            return self.to_aspect_ratio_add(size, size, -1, -1, border=border, prevent_outside_img=False)
        else:
            return self.to_aspect_ratio_add(size, size, keep_within_shape[0], keep_within_shape[1], border=border, prevent_outside_img=True)

    def to_aspect_ratio_add(self, height, width, img_height, img_width, border=0.0, prevent_outside_img=True):
        #img_height, img_width = img.shape[0], img.shape[1]
        by1, by2, bx1, bx2 = self.y1, self.y2, self.x1, self.x2
        bheight, bwidth = self.height, self.width
        bratio = bwidth / bheight
        aspect_ratio = width / height
        #print(height, width, img_height, img_width)
        #print(by1, by2, bx1, bx2)
        if bratio < aspect_ratio:
            # bb ist zu hoch / nicht breit genug
            # => breite erhoehen
            width_required = bheight * aspect_ratio
            width_diff = int(width_required - bwidth) # ohne int() werden bb koordinaten floats
            bx1 = bx1 - (width_diff // 2)
            bx2 = bx2 + (width_diff // 2)
            if width_diff % 2 != 0:
                bx1 = bx1 - 1
            #print("bratio < aspect_ratio", width_required, width_diff)
            #print(bx1, bx2, by1, by2)
        elif bratio > aspect_ratio:
            # bb ist zu breit / nicht hoch genug
            # => hoehe erhoehen
            height_required = bwidth / aspect_ratio
            height_diff = int(height_required - bheight) # ohne int() werden bb koordinaten floats
            by1 = by1 - (height_diff // 2)
            by2 = by2 + (height_diff // 2)
            if height_diff % 2 != 0:
                by1 = by1 - 1
            #print("bratio > aspect_ratio", height_required, height_diff)

        #print(by1, by2, bx1, bx2)

        if border > 0:
            bheight_new = by2 - by1
            bwidth_new = bx2 - bx1
            bx1 = bx1 - int(bwidth_new * border)
            bx2 = bx2 + int(bwidth_new * border)
            by1 = by1 - int(bheight_new * border)
            by2 = by2 + int(bheight_new * border)

        if prevent_outside_img:
            # shift if outside of image
            if bx1 < 0:
                bx2 += abs(bx1)
                bx1 = 0
            elif bx2 >= img_width:
                bx1 = bx1 - (bx2 - (img_width - 1))
                bx2 = img_width - 1

            if by1 < 0:
                by2 += abs(by1)
                by1 = 0
            elif by2 >= img_height:
                by1 = by1 - (by2 - (img_height - 1))
                by2 = img_height - 1

            # still too big?
            if bx1 < 0:
                bx1 = 0
            if bx2 >= img_width:
                bx2 = img_width - 1
            if by1 < 0:
                by1 = 0
            if by2 >= img_height:
                by2 = img_height - 1

        return Rectangle(y1=by1, y2=by2, x1=bx1, x2=bx2)

    def shift_from_left(self, by):
        return Rectangle(x1=self.x1+by, x2=self.x2+by, y1=self.y1, y2=self.y2)
        """
        if by != 0:
            self.x1 += by
            self.x2 += by
        """

    def shift_from_top(self, by):
        return Rectangle(x1=self.x1, x2=self.x2, y1=self.y1+by, y2=self.y2+by)
        """
        if by != 0:
            self.y1 += by
            self.y2 += by
        """

        """
        pad_top = abs(by1) if by1 < 0 else 0
        pad_right = bx2 - img_width if bx2 >= img_width else 0
        pad_bottom = by2 - img_height if by2 >= img_height else 0
        pad_left = abs(bx1) if bx1 < 0 else 0

        bx1 = max(bx1, 0)
        by1 = max(by1, 0)
        bx2 = max(bx2, img_width)
        by2 = max(by2, img_height)

        img_body = img[by1:by2, bx1:bx2, ...]
        if any([val > 0 for val in [pad_top, pad_right, pad_bottom, pad_left]]):
            img_body = np.pad(img_body, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="")
        img_body = misc.imresize(img_body, (height, width))

        return img_body
        """

    def shift(self, top=None, right=None, bottom=None, left=None):
        top = top if top is not None else 0
        right = right if right is not None else 0
        bottom = bottom if bottom is not None else 0
        left = left if left is not None else 0
        return Rectangle(x1=self.x1+left-right, x2=self.x2+left-right, y1=self.y1+top-bottom, y2=self.y2+top-bottom)

    def draw_on_image(self, img, color=[0, 255, 0], alpha=1.0, thickness=1, copy=copy):
        assert img.dtype in [np.uint8, np.float32, np.int32, np.int64]

        result = np.copy(img) if copy else img
        for i in range(thickness):
            y = [self.y1-i, self.y1-i, self.y2+i, self.y2+i]
            x = [self.x1-i, self.x2+i, self.x2+i, self.x1-i]
            rr, cc = draw.polygon_perimeter(y, x, shape=img.shape)
            if alpha >= 0.99:
                result[rr, cc, 0] = color[0]
                result[rr, cc, 1] = color[1]
                result[rr, cc, 2] = color[2]
            else:
                if result.dtype == np.float32:
                    result[rr, cc, 0] = (1 - alpha) * result[rr, cc, 0] + alpha * color[0]
                    result[rr, cc, 1] = (1 - alpha) * result[rr, cc, 1] + alpha * color[1]
                    result[rr, cc, 2] = (1 - alpha) * result[rr, cc, 2] + alpha * color[2]
                    result = np.clip(result, 0, 255)
                else:
                    result = result.astype(np.float32)
                    result[rr, cc, 0] = (1 - alpha) * result[rr, cc, 0] + alpha * color[0]
                    result[rr, cc, 1] = (1 - alpha) * result[rr, cc, 1] + alpha * color[1]
                    result[rr, cc, 2] = (1 - alpha) * result[rr, cc, 2] + alpha * color[2]
                    result = np.clip(result, 0, 255).astype(np.uint8)

        return result

    def draw_on_image_filled_binary(self, img, copy=True):
        if copy:
            img = np.copy(img)
        h, w = img.shape[0], img.shape[1]
        x1 = np.clip(self.x1, 0, w-1)
        x2 = np.clip(self.x2, 0, w-1)
        y1 = np.clip(self.y1, 0, h-1)
        y2 = np.clip(self.y2, 0, h-1)
        if x1 < x2 and y1 < y2:
            img[self.y1:self.y2, self.x1:self.x2] = 1
        return img

    def extract_from_image(self, img):
        pad_top = 0
        pad_right = 0
        pad_bottom = 0
        pad_left = 0

        height, width = img.shape[0], img.shape[1]
        x1, x2, y1, y2 = self.x1, self.x2, self.y1, self.y2

        if x1 < 0:
            pad_left = abs(x1)
            x2 = x2 + abs(x1)
            x1 = 0
        if y1 < 0:
            pad_top = abs(y1)
            y2 = y2 + abs(y1)
            y1 = 0
        if x2 >= width:
            pad_right = x2 - (width - 1)
        if y2 >= height:
            pad_bottom = y2 - (height - 1)

        if any([val > 0 for val in [pad_top, pad_right, pad_bottom, pad_left]]):
            if len(img.shape) == 2:
                img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant")
            else:
                img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="constant")

        return img[y1:y2, x1:x2]

    # val = int, float, tuple of int (top, right, btm, left), tuple of float (top, right, btm, left)
    def add_border(self, val, img_shape=None):
        if val == 0:
            return self.copy()
        else:
            if isinstance(val, int):
                rect = Rectangle(x1=self.x1-val, x2=self.x2+val, y1=self.y1-val, y2=self.y2+val)
            elif isinstance(val, float):
                rect = Rectangle(x1=int(self.x1 - self.width*val), x2=int(self.x2 + self.width*val), y1=int(self.y1 - self.height*val), y2=int(self.y2 + self.height*val))
            elif isinstance(val, tuple):
                assert len(val) == 4, str(len(val))

                if all([isinstance(subval, int) for subval in val]):
                    rect = Rectangle(x1=self.x1-val[3], x2=self.x2+val[1], y1=self.y1-val[0], y2=self.y2+val[2])
                elif all([isinstance(subval, float) or subval == 0 for subval in val]): # "or subval==0" da sonst zB (0.1, 0, 0.1, 0) einen fehler erzeugt (0 ist int)
                    rect = Rectangle(x1=int(self.x1 - self.width*val[3]), x2=int(self.x2 + self.width*val[1]), y1=int(self.y1 - self.height*val[0]), y2=int(self.y2 + self.height*val[2]))
                else:
                    raise Exception("Tuple of all ints or tuple of all floats expected, got %s" % (str([type(v) for v in val]),))
            else:
                raise Exception("int or float or tuple of ints/floats expected, got %s" % (type(val),))

            if img_shape is not None:
                rect.fix_by_image_dimensions(height=img_shape[0], width=img_shape[1])

            return rect

    def contains_point(self, x, y):
        return (self.x1 <= x < self.x2) and (self.y1 <= y < self.y2)

    def copy(self):
        return Rectangle(x1=self.x1, x2=self.x2, y1=self.y1, y2=self.y2)
        #return copy.copy(self)

    def __eq__(self, other):
        assert isinstance(other, Rectangle)
        return (self.x1, self.x2, self.y1, self.y2) == (other.x1, other.x2, other.y1, other.y2)

    def __ne__(self, other):
        assert isinstance(other, Rectangle)
        return self.x1 != other.x1 or self.x2 != other.x2 or self.y1 != other.y1 or self.y2 != other.y2

    def __hash__(self):
        return hash((self.x1, 1000+self.x2, 2000+self.y1, 3000+self.y2))

    def __str__(self):
        #return "Rectangle(y1=%d, x1=%d, y2=%d, x2=%d, height=%d, width=%d)" % (self.y1, self.x1, self.y2, self.x2, self.height, self.width)
        return "Rectangle(x1=%d, x2=%d, y1=%d, y2=%d, height=%d, width=%d)" % (self.x1, self.x2, self.y1, self.y2, self.height, self.width)

    def __repr__(self):
        return self.__str__()

class RectangleOnImage(object):
    def __init__(self, y1, x1, y2, x2, shape, class_label=None, score=None):
        self.rect = Rectangle(y1=y1, x1=x1, y2=y2, x2=x2)
        self.shape = shape if isinstance(shape, tuple) else shape.shape
        self.class_label = class_label
        self.score = score

    @staticmethod
    def from_rect(rect, shape, class_label=None, score=None):
        return RectangleOnImage(y1=rect.y1, x1=rect.x1, y2=rect.y2, x2=rect.x2, shape=shape, class_label=class_label, score=score)

    def copy_with_new_rect(self, rect):
        #return RectangleOnImage.from_rect(rect, self.shape, score=self.score)
        #rectimg = copy.copy(self)
        #rectimg.rect = rect
        #return rectimg
        return self.copy(rect=rect)

    def copy(self, rect=None, x1=None, x2=None, y1=None, y2=None, shape=None, class_label=None, score=None):
        """
        rectimg = copy.deepcopy(self)
        if rect is not None:
            rectimg.rect = rect
        if x1 is not None:
            rectimg.x1 = x1
        if x2 is not None:
            rectimg.x2 = x2
        if y1 is not None:
            rectimg.y1 = y1
        if y2 is not None:
            rectimg.y2 = y2
        if shape is not None:
            assert isinstance(shape, tuple)
            rectimg.shape = shape
        if class_label is not None:
            rectimg.class_label = class_label
        if score is not None:
            rectimg.score = score
        #return self.copy_with_new_rect(self.rect.copy())
        """

        # neue variante hier weil deepcopy wohl sehr langsam ist(?)
        if rect is None:
            rectimg = RectangleOnImage(
                x1=self.rect.x1 if x1 is None else x1,
                x2=self.rect.x2 if x2 is None else x2,
                y1=self.rect.y1 if y1 is None else y1,
                y2=self.rect.y2 if y2 is None else y2,
                shape=tuple(self.shape) if shape is None else shape,
                class_label=self.class_label if class_label is None else class_label,
                score=self.score if score is None else score
            )
        else:
            assert x1 is None
            assert x2 is None
            assert y1 is None
            assert y2 is None
            rectimg = RectangleOnImage(
                x1=rect.x1,
                x2=rect.x2,
                y1=rect.y1,
                y2=rect.y2,
                shape=tuple(self.shape) if shape is None else shape,
                class_label=self.class_label if class_label is None else class_label,
                score=self.score if score is None else score
            )

        return rectimg

    def on(self, img):
        shape_new = img if isinstance(img, tuple) else img.shape
        if shape_new == self.shape:
            return self
        else:
            rect_rs = self.rect.resize(self.shape, shape_new)
            return self.copy(rect=rect_rs, shape=shape_new)

    @staticmethod
    def from_dlib_rect(rect, img):
        return RectangleOnImage.from_rect(Rectangle.from_dlib_rect(rect), img)

    def to_dlib_rect(self):
        return self.rect.to_dlib_rect()

    @property
    def x1(self):
        return self.rect.x1

    @property
    def x2(self):
        return self.rect.x2

    @property
    def y1(self):
        return self.rect.y1

    @property
    def y2(self):
        return self.rect.y2

    @x1.setter
    def x1(self, value):
        self.rect.x1 = value

    @x2.setter
    def x2(self, value):
        self.rect.x2 = value

    @y1.setter
    def y1(self, value):
        self.rect.y1 = value

    @y2.setter
    def y2(self, value):
        self.rect.y2 = value

    @property
    def height(self):
        return self.rect.height

    @property
    def width(self):
        return self.rect.width

    @property
    def center_x(self):
        return self.rect.center_x

    @property
    def center_y(self):
        return self.rect.center_y

    @property
    def x1_normalized(self):
        return self.rect.x1 / self.shape[1]

    @property
    def x2_normalized(self):
        return self.rect.x2 / self.shape[1]

    @property
    def y1_normalized(self):
        return self.rect.y1 / self.shape[0]

    @property
    def y2_normalized(self):
        return self.rect.y2 / self.shape[0]

    @property
    def height_normalized(self):
        return self.rect.height / self.shape[1]

    @property
    def width_normalized(self):
        return self.rect.width / self.shape[0]

    @property
    def center_x_normalized(self):
        return self.rect.center_x / self.shape[1]

    @property
    def center_y_normalized(self):
        return self.rect.center_y / self.shape[0]

    @property
    def area(self):
        return self.rect.area

    def is_identical_with(self, other_rect):
        return self == other_rect

    def overlaps_with(self, other_rect):
        assert isinstance(other_rect, RectangleOnImage)
        return self.rect.overlaps_with(other_rect.rect)

    def area_intersection(self, other_rect):
        assert isinstance(other_rect, RectangleOnImage)
        return self.rect.area_intersection(other_rect.rect)

    def area_union(self, other_rect):
        assert isinstance(other_rect, RectangleOnImage)
        return self.rect.area_union(other_rect.area_union)

    def intersection_over_union(self, other_rect):
        assert isinstance(other_rect, RectangleOnImage)
        return self.rect.intersection_over_union(other_rect.rect)

    def iou(self, other_rect):
        return self.intersection_over_union(other_rect)

    def is_fully_within_image(self, img):
        return self.rect.is_fully_within_image(img)

    def fix_by_image_dimensions(self, height=None, width=None):
        if height is None:
            assert width is None
            assert self.shape is not None
            self.rect.fix_by_image_dimensions(height=self.shape[0], width=self.shape[1])
        else:
            self.rect.fix_by_image_dimensions(height=height, width=width)

    def resize(self, from_img, to_img):
        return self.on(to_img)

    def merge(self, other_rect):
        rects_merged = self.rect.merge(other_rect if isinstance(other_rect, Rectangle) else other_rect.rect)
        rectimg = self.copy_with_new_rect(rects_merged)
        if isinstance(other_rect, Rectangle):
            rectimg.score = self.score
            rectimg.class_label = self.class_label
        else:
            scores = [val for val in [self.score, other_rect.score] if val is not None]
            if len(scores) > 0:
                rectimg.score = max(scores)
            class_labels = [val for val in [self.class_label, other_rect.class_label] if val is not None]
            if len(class_labels) > 0:
                rectimg.class_label = class_labels[0]
        return rectimg

    def to_squared(self, border=0.0, keep_within_shape=None):
        s = keep_within_shape
        if keep_within_shape is not None and keep_within_shape == True:
            s = self.shape
        rect_rs = self.rect.to_squared(border=border, keep_within_shape=keep_within_shape)
        return self.copy_with_new_rect(rect_rs)

    def to_aspect_ratio_add(self, height, width, img_height=None, img_width=None, border=0.0, prevent_outside_img=True):
        img_height = self.shape[0] if img_height is None else img_height
        img_width = self.shape[1] if img_width is None else img_width
        rect_rs = self.rect.to_aspect_ratio_add(height, width, img_height=img_height, img_width=img_width, border=border, prevent_outside_img=prevent_outside_img)
        return self.copy_with_new_rect(rect_rs)

    def shift_from_left(self, by):
        return self.copy_with_new_rect(self.rect.shift_from_left(by))

    def shift_from_top(self, by):
        return self.copy_with_new_rect(self.rect.shift_from_top(by))

    def shift(self, top=None, right=None, bottom=None, left=None):
        return self.copy_with_new_rect(self.rect.shift(top=top, right=right, bottom=bottom, left=left))

    def draw_on_image(self, img, color=[0, 255, 0], alpha=1.0, thickness=1, copy=True, project=True):
        if project:
            return self.on(img).rect.draw_on_image(img, color=color, alpha=alpha, thickness=thickness, copy=copy)
        else:
            return self.rect.draw_on_image(img, color=color, alpha=alpha, thickness=thickness, copy=copy)

    def draw_on_image_filled_binary(self, img, copy=True, project=True):
        if project:
            return self.on(img).rect.draw_on_image_filled_binary(img, copy=copy)
        else:
            return self.rect.draw_on_image_filled_binary(img, copy=copy)

    def extract_from_image(self, img, project=True):
        if project:
            return self.on(img).rect.extract_from_image(img)
        else:
            return self.rect.extract_from_image(img)

    # val = int, float, tuple of int (top, right, btm, left), tuple of float (top, right, btm, left)
    def add_border(self, val, img_shape=None):
        img_shape = self.shape if img_shape is not None and img_shape == True else img_shape
        rect_rs = self.rect.add_border(val, img_shape=img_shape)
        return self.copy_with_new_rect(rect_rs)

    def contains_point(self, x, y):
        return self.rect.contains_point(x=x, y=y)

    def __eq__(self, other):
        assert isinstance(other, RectangleOnImage)
        return self.rect.__eq__(other.rect) and self.shape == other.shape

    def __ne__(self, other):
        assert isinstance(other, RectangleOnImage)
        return self.rect.__ne__(other.rect) or self.shape != other.shape

    def __hash__(self):
        return self.rect.__hash__()

    def __str__(self):
        return "RectangleOnImage(x1=%d, x2=%d, y1=%d, y2=%d, height=%d, width=%d, shape=%s)" % (self.x1, self.x2, self.y1, self.y2, self.height, self.width, str(self.shape))

    def __repr__(self):
        return self.__str__()

class RectangleList(list):
    def draw_on_image(self, img, color=[0, 255, 0], alpha=1.0, copy=True, from_img=None):
        if copy:
            img = np.copy(img)

        orig_dtype = img.dtype
        if alpha != 1.0 and img.dtype != np.float32:
            img = img.astype(np.float32, copy=False)

        for rect in self:
            if from_img is not None:
                rect.resize(from_img, img).draw_on_image(img, color=color, alpha=alpha, copy=False)
            else:
                rect.draw_on_image(img, color=color, alpha=alpha, copy=False)

        if orig_dtype != img.dtype:
            img = img.astype(orig_dtype, copy=False)

        return img

    def extract_from_image(self, img, resize_from_img=None):
        if from_img is not None:
            return [rect.resize(resize_from_img, img).extract_from_image(img) if rect is not None else None for rect in self]
        else:
            return [rect.extract_from_image(img) if rect is not None else None for rect in self]

class RectangleOnImageList(list):
    def on(self, img):
        return RectangleOnImageList([rectimg.on(img) if rectimg is not None else None for rectimg in self])

    def draw_on_image(self, img, color=[0, 255, 0], alpha=1.0, copy=True):
        if copy:
            img = np.copy(img)

        orig_dtype = img.dtype
        if alpha != 1.0 and img.dtype != np.float32:
            img = img.astype(np.float32, copy=False)

        for rectimg in self:
            if rectimg is not None:
                rectimg.draw_on_image(img, color=color, alpha=alpha, copy=False)

        if orig_dtype != img.dtype:
            img = img.astype(orig_dtype, copy=False)

        return img

    def extract_from_image(self, img):
        return [rectimg.extract_from_image(img) if rectimg is not None else None for rectimg in self]
