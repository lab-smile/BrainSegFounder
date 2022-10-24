import copy
from configparser import SectionProxy

import torch
import random
import numpy as np
from math import comb, floor
from typing import Collection, Tuple
from monai.transforms import RandomizableTransform, Compose


class FlipTransform(RandomizableTransform):
    def __init__(self, prob: float) -> None:
        super().__init__(prob=prob)
        self._axes = None

    def randomize(self, data=None) -> None:
        super().randomize(None)
        num_axes = self.R.randint(1, 3)
        axes = self.R.choice([0, 1, 2], size=num_axes, replace=False)
        self._axes = tuple(axes.tolist())

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        self.randomize()
        if self._do_transform:
            img = torch.flip(img, self._axes)
        return img


class ShuffleTransform(RandomizableTransform):
    def __init__(self, prob, n_blocks):
        super().__init__(prob=prob)
        self.slices = None
        self.block_sizes = None
        self.n_blocks = n_blocks
        self.subset = None

    def randomize(self, data: Tuple[list, torch.Tensor]) -> None:
        sizes, orig_img = data

        super().randomize(data=None)
        self.block_sizes = [self.R.randint(1, num_pixels // 10) for num_pixels in sizes]
        start_pts = [self.R.randint(0, num_pixels - block_size)
                     for num_pixels, block_size in zip(sizes, self.block_sizes)]

        self.slices = [slice(start, start+size) for start, size in zip(start_pts, self.block_sizes)]

        # We need the original image here: blocks overlapping would otherwise cause pixels to fly everywhere
        self.subset = orig_img[self.slices].numpy()  # We have to go to numpy as well - otherwise won't shuffle

        if self.subset is not None:
            self.R.shuffle(self.subset)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if self._do_transform:
            size = list(img.size())
            orig_img = copy.deepcopy(img)
            for _ in range(self.n_blocks):
                # Get new block sizes and start points, shuffle a new subset
                self.randomize(data=(size, orig_img))

                # If the subset was just shuffled, then replace into original image
                if self.subset is not None:
                    img[self.slices] = torch.from_numpy(self.subset)
            
        return img


class NonlinearTransformation(RandomizableTransform):
    def __init__(self, prob):
        super().__init__(prob=prob)
        self.flip_y = None
        self.bezier_pts = None
        pass

    def randomize(self, data=None) -> None:
        super().randomize(data=None)
        self.bezier_pts = [[0, 0], [self.R.random_sample(), self.R.random_sample()],
                           [self.R.random_sample(), self.R.random_sample()], [1, 1]]
        self.flip_y = self.R.random_sample() > 0.5

    def __call__(self, img: torch.Tensor):
        if self._do_transform:
            self.randomize()
            bezier_x, bezier_y = bezier_curve(self.bezier_pts, n_times=100000)  # 100,000 is the number MG uses

            flipped_x = np.sort(bezier_x)
            flipped_y = np.sort(bezier_y) if random.random() < 0.5 else bezier_y  # Half of the time, don't flip y
            return torch.from_numpy(np.interp(img.numpy(), flipped_x, flipped_y))
        else:
            return img


class PaintTransform(RandomizableTransform):
    def __init__(self, prob: float, inpaint_rate: float, shape: list):
        super().__init__(prob=prob)
        self.paint_type = None
        self.inpaint_rate = inpaint_rate
        self.shape = shape
        self.block_sizes = None
        self.start_pts = None
        self.slices = None
        self.painted_array = None
        self.num_paintings = None

    def randomize(self, sizes: list):
        super().randomize(data=None)
        self.paint_type = 'inpaint' if self.R.rand(1, 1) < self.inpaint_rate else 'outpaint'
        min_fraction = 1 / 6 if self.paint_type == 'inpaint' else 3 / 7
        max_fraction = 1 / 3 if self.paint_type == 'inpaint' else 4 / 7
        min_sizes = [floor(dimension * min_fraction) for dimension in self.shape]
        max_sizes = [floor(dimension * max_fraction) for dimension in self.shape]

        self.block_sizes = [self.R.randint(min_size, max_size) for min_size, max_size in zip(min_sizes, max_sizes)]
        self.start_pts = [self.R.randint(3, size - block_size - 3) for size, block_size in zip(sizes, self.block_sizes)]

        self.slices = [slice(start_pt, block_size) for start_pt, block_size in zip(self.start_pts, self.block_sizes)]
        self.painted_array = torch.Tensor(self.R.random_sample(sizes))
        if self.num_paintings is None:
            self.num_paintings = self.R.binomial(5, 0.95)

    def __call__(self, img: torch.Tensor):
        if img is None:
            raise ValueError("No image")
        if self._do_transform:
            self.randomize(sizes=img.shape)  # Set num_paintings
            if self.paint_type == 'inpaint':
                for _ in range(self.num_paintings):
                    self.randomize(sizes=img.shape)
                    img[self.slices] = self.painted_array[self.slices]
            else:
                for _ in range(self.num_paintings):
                    self.randomize(sizes=img.shape)
                    left_slices = [slice(0, start_pt) for start_pt in self.start_pts]
                    right_slices = [slice(start_pt + block_size, shape)
                                    for start_pt, block_size, shape in zip(self.start_pts, self.block_sizes, img.shape)]
                    img[left_slices] = self.painted_array[left_slices]
                    img[right_slices] = self.painted_array[right_slices]
        return img


def init_transform(settings: SectionProxy) -> Compose:
    shuffling_rate = settings.getfloat('shuffling_rate')
    painting_rate = settings.getfloat('painting_rate')
    inpainting_rate = settings.getfloat('inpainting_rate')
    nlt_rate = settings.getfloat('non_linear_transformation_rate')
    window_size = settings.getint('window_size')
    shuffle = ShuffleTransform(shuffling_rate, n_blocks=10000)
    paint = PaintTransform(painting_rate, inpainting_rate, [window_size] * 3)
    nlt = NonlinearTransformation(nlt_rate)
    slice_transform = Compose(transforms=[shuffle, paint, nlt])
    return slice_transform


class TransformImage:
    def __init__(self, settings: SectionProxy, slice_transform: Compose):
        self.settings = settings
        self.slice_transform = slice_transform

    def __call__(self, img):
        flip_rate = self.settings.getfloat('flip_rate')
        num_slices = self.settings.getint('num_slices')
        window_size = self.settings.getint('window_size')

        # Flip if necessary
        img = FlipTransform(flip_rate)(img)
        tf_img = copy.deepcopy(img)

        for _ in range(num_slices):
            sliced_img, slices = slice_img(tf_img, window_size)
            #print(f'Sliced image: {sliced_img}')
            sliced_img = self.slice_transform(sliced_img)
            tf_img[slices] = sliced_img

        return img, tf_img


def slice_img(img: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, list[slice]]:
    # Random start point
    for dim in img.shape:
        if dim <= window_size:
            raise ValueError("Image smaller than window size. Could not slice.")
    start_pts = [random.randint(0, dim - window_size) for dim in img.shape]
    slices = [slice(start_pt, start_pt + window_size) for start_pt in start_pts]
    img_slice = img[slices]
    return img_slice, slices


def bezier_curve(points: Collection[list], n_times=1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the Bézier curve defined by the input points, sampled at n_times times (modified from MG)
    :param points: control points for curve
    :param n_times:
    :return: tuple of (x_vals, y_vals) where each is the x- and y-timepoints of the curve
    """

    n_pts = len(points)
    x_pts = np.array([p[0] for p in points])
    y_pts = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, n_times)

    polynomial_array = np.array([bernstein_poly(i, n_pts - 1, t) for i in range(0, n_pts)])
    x_vals = np.dot(x_pts, polynomial_array)
    y_vals = np.dot(y_pts, polynomial_array)

    return x_vals, y_vals


def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t (source: MG)
    """
    return comb(n, i) * (t ** (n - i)) * (1 - t) ** i
