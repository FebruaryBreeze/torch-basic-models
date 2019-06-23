import unittest

import torch

import torch_basic_models


class MyTestCase(unittest.TestCase):
    def test_arc_face(self):
        batch_size = 10
        feature_dim = 256
        num_classes = 1000

        arc_face = torch_basic_models.ArcFace.factory(config={
            'feature_dim': feature_dim,
            'num_classes': num_classes
        })

        mock_feature = torch.randn(batch_size, feature_dim)
        label = torch.randint(0, num_classes, (batch_size,), dtype=torch.long)
        output = arc_face(mock_feature, label)
        self.assertEqual(output.shape, (batch_size, num_classes))


if __name__ == '__main__':
    unittest.main()
