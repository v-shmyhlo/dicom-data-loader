import utils


def test_iou():
    a = [False, True, True, True]
    b = [True, True, True, False]

    actual = utils.iou(a, b)

    assert actual == .5


def test_iou_no_intersection():
    a = [False, False, True, True]
    b = [True, True, False, False]

    actual = utils.iou(a, b)

    assert actual == 0.


def test_iou_no_positive():
    a = [False, False, False, False]
    b = [False, False, False, False]

    actual = utils.iou(a, b)

    assert actual == 0.
