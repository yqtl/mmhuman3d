import pytest
import torch

from mmhuman3d.core.evaluation.mean_ap import mAP


def test_mAP():
    target = torch.Tensor([[1, 1, 0, -1], [1, 1, 0, -1], [0, -1, 1, -1],
                           [0, 1, 0, -1]])
    pred = torch.Tensor([[0.9, 0.8, 0.3, 0.2], [0.1, 0.2, 0.2, 0.1],
                         [0.7, 0.5, 0.9, 0.3], [0.8, 0.1, 0.1, 0.2]])

    # target and pred should both be np.ndarray or torch.Tensor
    with pytest.raises(TypeError):
        target_list = target.tolist()
        _ = mAP(pred, target_list)

    # target and pred should be in the same shape
    with pytest.raises(AssertionError):
        target_shorter = target[:-1]
        _ = mAP(pred, target_shorter)

    assert mAP(pred, target) == pytest.approx(68.75, rel=1e-2)

    target_no_difficult = torch.Tensor([[1, 1, 0, 0], [0, 1, 0, 0],
                                        [0, 0, 1, 0], [1, 0, 0, 0]])
    assert mAP(pred, target_no_difficult) == pytest.approx(70.83, rel=1e-2)