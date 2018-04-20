"""Data loader and all supporting code"""

import csv
import numpy as np
import os
import math
import parsing


class DataLoader(object):
    """Represents loader for loading batches of (DICOM image, contour mask) pairs"""

    def __init__(self, path, batch_size, seed=None):
        """Configures loader

        :param path: path to folder containing link.csv and DICOMs, contourfiles subfolders
        :param batch_size: number of samples in a single batch
        :param seed: seed used to shuffle dataset
        """

        self._batch_size = batch_size
        self._seed = seed

        self._files = []
        with open(os.path.join(path, 'link.csv')) as f:
            reader = csv.DictReader(f)
            for ids in reader:
                for pair in _find_pairs(path, ids):
                    self._files.append(pair)

    def __iter__(self):
        """Converts loader to generator for iterating

        :return: generator which yields batches of (DICOM image, contour mask) pairs
        """

        num_samples = len(self._files)
        indices_shuffled = np.random.RandomState(seed=self._seed).permutation(num_samples)
        num_batches = math.ceil(num_samples / self._batch_size)

        for batch_index in range(num_batches):
            images = []
            i_contours = []

            start = batch_index * self._batch_size
            stop = (batch_index + 1) * self._batch_size
            sample_indices = indices_shuffled[start:stop]

            for sample_index in sample_indices:
                sample = self._files[sample_index]
                image = parsing.parse_dicom_file(sample['image_path'])
                i_contour = parsing.parse_contour_file(sample['i_contour_path'])
                i_contour = parsing.poly_to_mask(i_contour, width=image.shape[1], height=image.shape[0])

                images.append(image)
                i_contours.append(i_contour)

            images = np.array(images)
            i_contours = np.array(i_contours)

            yield images, i_contours


def _find_pairs(path, ids):
    """Find all pairs of images and i-contours given single row from link.csv

    :param path: path to folder containing DICOMs and contourfiles subfolders
    :param ids: dict holding paths to linked DICOM and contourfiles subfolders
    :return: list of dicts with paths to DICOM and contourfile pair
    """

    images_folder = os.path.join(path, 'dicoms', ids['patient_id'])
    i_contours_folder = os.path.join(path, 'contourfiles', ids['original_id'], 'i-contours')
    images = os.listdir(images_folder)

    pairs = []

    for image in images:
        index = os.path.splitext(image)[0]
        i_contour = 'IM-0001-{}-icontour-manual.txt'.format(index.zfill(4))

        image_path = os.path.join(images_folder, image)
        i_contour_path = os.path.join(i_contours_folder, i_contour)

        if os.path.exists(i_contour_path):
            pair = {'image_path': image_path, 'i_contour_path': i_contour_path}
            pairs.append(pair)

    return pairs
