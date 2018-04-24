import numpy as np
from data_loader import DataLoader


def test_iterates_over_data_loader():
    data_loader = DataLoader(path='./test/fixtures/final_data', batch_size=8)

    batches = list(data_loader)  # load all data in memory
    assert len(batches) == 6

    for i, (image, i_contour, o_contour) in enumerate(batches):
        batch_size = 6 if i == len(batches) - 1 else 8  # not enough samples to fill last batch
        assert image.shape == i_contour.shape == o_contour.shape == (batch_size, 256, 256)
        assert image.dtype == np.int16
        assert i_contour.dtype == o_contour.dtype == np.bool


def test_data_loader_skips_i_contours():
    data_loader = DataLoader(path='./test/fixtures/final_data', include_i_contours=False, batch_size=8)

    batches = list(data_loader)  # load all data in memory
    assert len(batches) == 6

    for image, i_contour, o_contour in batches:
        assert image is not None
        assert i_contour is None
        assert o_contour is not None


def test_data_loader_skips_o_contours():
    data_loader = DataLoader(path='./test/fixtures/final_data', include_o_contours=False, batch_size=8)

    batches = list(data_loader)  # load all data in memory
    assert len(batches) == 12

    for image, i_contour, o_contour in batches:
        assert image is not None
        assert i_contour is not None
        assert o_contour is None


def test_data_loader_skips_contours():
    data_loader = DataLoader(path='./test/fixtures/final_data', include_i_contours=False, include_o_contours=False,
                             batch_size=8)

    batches = list(data_loader)  # load all data in memory
    assert len(batches) == 143

    for image, i_contour, o_contour in batches:
        assert image is not None
        assert i_contour is None
        assert o_contour is None


def test_data_loader_shuffles_dataset():
    data_loader = DataLoader(path='./test/fixtures/final_data', batch_size=8)

    for a, b in zip(data_loader, data_loader):
        assert not np.array_equal(a, b)


def test_data_loader_fixes_shuffle_seed():
    data_loader = DataLoader(path='./test/fixtures/final_data', batch_size=8, seed=42)

    for a, b in zip(data_loader, data_loader):
        assert np.array_equal(a, b)
