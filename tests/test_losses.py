import unittest

import torch

import torch_basic_models


class MyTestCase(unittest.TestCase):
    def test_losses(self):
        predict = torch.Tensor([[0, 4]])
        output = torch.Tensor([1]).long()

        cross_entropy = torch_basic_models.CrossEntropyLoss.factory()
        cross_entropy_loss = cross_entropy(predict, output)

        label_smoothing = torch_basic_models.LabelSmoothingLoss.factory(config={
            'smooth_ratio': 0.0
        })
        label_smoothing_loss = label_smoothing(predict, output)
        self.assertTrue(cross_entropy_loss == label_smoothing_loss)

        label_smoothing = torch_basic_models.LabelSmoothingLoss.factory(config={
            'smooth_ratio': 0.1
        })
        label_smoothing_loss = label_smoothing(predict, output)
        self.assertTrue(cross_entropy_loss < label_smoothing_loss)

    def test_l2_loss(self):
        predict = torch.Tensor([[0.1, 0.2]])
        output = torch.Tensor([[0.1, 0.2]])
        l2_metric = torch_basic_models.L2Loss.factory()
        self.assertAlmostEqual(l2_metric(predict, output).item(), 0.0)

        predict = torch.Tensor([[1.0, 1.0]])
        output = torch.Tensor([[0.0, 0.0]])
        l2_metric = torch_basic_models.L2Loss.factory()
        self.assertAlmostEqual(l2_metric(predict, output).item(), 1.0)

        predict = torch.Tensor([[1.0, 1.0, 1.0]])
        output = torch.Tensor([[0.0, 0.0, 0.0]])
        l2_metric = torch_basic_models.L2Loss.factory({
            'normalize': True
        })
        self.assertAlmostEqual(l2_metric(predict, output).item(), 1.0)

        predict = torch.Tensor([[1.0, 0.0, 0.0]])
        output = torch.Tensor([[-1.0, 0.0, 0.0]])
        normalized_l2_metric = torch_basic_models.NormalizedL2Loss.factory()
        self.assertAlmostEqual(normalized_l2_metric(predict, output).item(), 2.0)


if __name__ == '__main__':
    unittest.main()
