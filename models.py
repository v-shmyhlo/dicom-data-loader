"""Models and all supporting code"""

import cv2
import numpy as np
import utils


class OtsuThresholding(object):
    """Represents model for predicting i-contours given images and o-countours using Otsu's method"""

    def __init__(self, kernel_size=None):
        """Configures loader

        :param kernel_size: size of kernel for morphological closing, when `None` closing is not performed,
            defaults to `None`
        """

        self._kernel_size = kernel_size

    def predict(self, input):
        """Predicts i-contour

        :param input: tuple with images and o-contours
        :return: predicted i-contours
        """

        images, o_contours = input
        assert images.ndim == o_contours.ndim == 3
        assert images.shape == o_contours.shape
        assert o_contours.dtype == np.bool

        i_contours = []

        for image, o_contour in zip(images, o_contours):
            image = (image / image.max() * 255).astype(np.uint8)

            _, i_contour_vector = cv2.threshold(image[o_contour], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            i_contour_vector = i_contour_vector.squeeze()
            i_contour = np.copy(o_contour)
            i_contour[o_contour] = i_contour_vector

            if self._kernel_size:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self._kernel_size, self._kernel_size))
                i_contour = cv2.morphologyEx(i_contour.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(np.bool)
                i_contour[np.logical_not(o_contour)] = False  # ignore everything outside o-contour

            i_contours.append(i_contour)

        i_contours = np.array(i_contours)

        return i_contours

    def score(self, input, i_contours):
        """Computes mean IOU on the given data and labels

        :param input: tuple with images and o-contours
        :param i_contours: i-contours labels
        :return: mean IOU
        """

        i_contours_pred = self.predict(input)

        return utils.iou(i_contours, i_contours_pred)
