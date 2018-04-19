import numpy as np
from parsing import poly_to_mask, parse_contour_file, parse_dicom_file


def test_poly_to_mask():
    # L-shaped polygon
    poly = [(0, 0), (1, 0), (2, 0), (2, 1), (3, 1), (3, 2), (3, 3), (2, 3), (1, 3), (0, 3), (0, 2), (0, 1)]

    actual = poly_to_mask(poly, 4, 4)

    expected = np.array([
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0]
    ]).astype(np.bool)

    assert np.array_equal(actual, expected)


def test_parse_dicom_file():
    filename = './test/fixtures/final_data/dicoms/SCD0000101/1.dcm'
    image = parse_dicom_file(filename)
    assert image.shape == (256, 256)
    assert image.dtype == np.int16


def test_parse_contour_file():
    filename = 'test/fixtures/final_data/contourfiles/SC-HF-I-1/i-contours/IM-0001-0048-icontour-manual.txt'
    contour = parse_contour_file(filename)

    contour = np.array(contour)  # convert to array just to simplify assertion
    assert contour.shape == (150, 2)
    assert contour.dtype == np.float64
