"""Data loader and all supporting code"""

import csv
import numpy as np
import os
import math
import parsing


class DataLoader(object):
    """Represents loader for loading batches of (DICOM image, contour mask) pairs"""

    def __init__(self, path, batch_size, include_i_contours=True, include_o_contours=True, seed=None):
        """Configures loader

        :param path: path to folder containing link.csv and DICOMs, contourfiles subfolders
        :param batch_size: number of samples in a single batch
        :param include_i_contours: whether or not include corresponding i-contour, defaults to `True`
        :param include_o_contours: whether or not include corresponding o-contour, defaults to `True`
        :param seed: seed used to shuffle dataset, defaults to `None`
        """

        self._batch_size = batch_size
        self._include_i_contours = include_i_contours
        self._include_o_contours = include_o_contours
        self._seed = seed
        self._files = []

        with open(os.path.join(path, 'link.csv')) as f:
            reader = csv.DictReader(f)
            for ids in reader:
                for pair in _find_matching_images(path, ids, include_i_contours=include_i_contours,
                                                  include_o_contours=include_o_contours):
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

            i_contours = [] if self._include_i_contours else None
            o_contours = [] if self._include_o_contours else None

            start = batch_index * self._batch_size
            stop = (batch_index + 1) * self._batch_size
            sample_indices = indices_shuffled[start:stop]

            for sample_index in sample_indices:
                sample = self._files[sample_index]
                image = parsing.parse_dicom_file(sample['image_path'])
                images.append(image)

                if self._include_i_contours:
                    i_contour = parsing.parse_contour_file(sample['i_contour_path'])
                    i_contour = parsing.poly_to_mask(i_contour, width=image.shape[1], height=image.shape[0])
                    i_contours.append(i_contour)
                if self._include_o_contours:
                    o_contour = parsing.parse_contour_file(sample['o_contour_path'])
                    o_contour = parsing.poly_to_mask(o_contour, width=image.shape[1], height=image.shape[0])
                    o_contours.append(o_contour)

            images = np.array(images)

            if self._include_i_contours:
                i_contours = np.array(i_contours)
            if self._include_o_contours:
                o_contours = np.array(o_contours)

            yield images, i_contours, o_contours


def _find_matching_images(path, ids, include_i_contours, include_o_contours):
    """Find all pairs of images and i-contours given single row from link.csv

    :param path: path to folder containing DICOMs and contourfiles subfolders
    :param ids: dict holding paths to linked DICOM and contourfiles subfolders
    :param include_i_contours: whether or not include corresponding i-contour
    :param include_o_contours: whether or not include corresponding o-contour
    :return: list of dicts with paths to DICOM and contourfile pair
    """

    images_folder = os.path.join(path, 'dicoms', ids['patient_id'])

    if include_i_contours:
        i_contours_folder = os.path.join(path, 'contourfiles', ids['original_id'], 'i-contours')
    if include_o_contours:
        o_contours_folder = os.path.join(path, 'contourfiles', ids['original_id'], 'o-contours')

    images = os.listdir(images_folder)
    samples = []

    for image in images:
        index = os.path.splitext(image)[0]
        image_path = os.path.join(images_folder, image)
        sample = {'image_path': image_path}

        if include_i_contours:
            i_contour = 'IM-0001-{}-icontour-manual.txt'.format(index.zfill(4))
            i_contour_path = os.path.join(i_contours_folder, i_contour)
            sample['i_contour_path'] = i_contour_path
            if not os.path.exists(i_contour_path):
                continue

        if include_o_contours:
            o_contour = 'IM-0001-{}-ocontour-manual.txt'.format(index.zfill(4))
            o_contour_path = os.path.join(o_contours_folder, o_contour)
            sample['o_contour_path'] = o_contour_path

            if not os.path.exists(o_contour_path):
                continue

        samples.append(sample)

    return samples
