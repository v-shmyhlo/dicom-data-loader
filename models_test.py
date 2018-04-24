import pytest
import numpy as np
import models


# toy example to validate some basic assumptions
@pytest.fixture
def input():
    image = [
        [.2, .1, .2],
        [.8, .3, .7],
        [.9, .8, .9]
    ]
    o_contours = [
        [0, 1, 1],
        [0, 1, 1],
        [0, 1, 1]
    ]

    image = np.expand_dims(image, 0)
    o_contours = np.expand_dims(o_contours, 0).astype(np.bool)

    return image, o_contours


def test_otsu_thresholding_predict(input):
    model = models.OtsuThresholding()

    i_contours = [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 1]
    ]

    i_contours = np.expand_dims(i_contours, 0).astype(np.bool)
    i_contours_pred = model.predict(input)

    assert np.array_equal(i_contours_pred, i_contours)


def test_otsu_thresholding_score(input):
    model = models.OtsuThresholding()

    i_contours = [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ]

    i_contours = np.expand_dims(i_contours, 0).astype(np.bool)
    score = model.score(input, i_contours)

    assert score == 2 / 3
