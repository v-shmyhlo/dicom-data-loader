import numpy as np
from data_loader import DataLoader


def test_iterates_over_data_loader():
    data_loader = DataLoader(path='./test/fixtures/final_data', batch_size=8)

    batches = list(data_loader)  # load all data in memory
    assert len(batches) == 12

    for x, y in batches:
        assert x.shape == y.shape == (8, 256, 256)
        assert x.dtype == np.int16
        assert y.dtype == np.bool


def test_data_loader_shuffles_dataset():
    data_loader = DataLoader(path='./test/fixtures/final_data', batch_size=8)

    assert not np.array_equal(list(data_loader), list(data_loader))


def test_data_loader_fixes_shuffle_seed():
    data_loader = DataLoader(path='./test/fixtures/final_data', batch_size=8, seed=42)

    assert np.array_equal(list(data_loader), list(data_loader))
