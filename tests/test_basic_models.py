import unittest

import box
import jsonschema
import torch
import torch.nn as nn

import torch_basic_models


class MyTestCase(unittest.TestCase):
    def test_resnet(self):
        model = torch_basic_models.ResNet.factory()
        self.assertIsNotNone(model)
        del model

        config = {
            'feature_dim': 256
        }
        model = torch_basic_models.ResNet.factory(config=config)

        data_in = torch.randn(1, 3, 224, 224)
        data_out = model(data_in)
        self.assertEqual(data_out.size(1), 256)

    def test_mobile_net_v2(self):
        model = torch_basic_models.MobileNetV2.factory()
        self.assertIsNotNone(model)
        del model

        config = {
            'feature_dim': 256,
            'dropout_ratio': 0.2
        }
        model = torch_basic_models.MobileNetV2.factory(config=config)

        data_in = torch.randn(1, 3, 224, 224)
        data_out = model(data_in)
        self.assertEqual(data_out.size(1), 256)

    def test_mobile_net_v3(self):
        model = torch_basic_models.MobileNetV3.factory()
        self.assertIsNotNone(model)

        data_in = torch.randn(1, 3, 224, 224)
        data_out = model(data_in)
        self.assertEqual(data_out.size(1), 1000)

    def test_json_schema_validate(self):
        def load_invalid_config():
            config = {
                'invalid_item': 0
            }
            torch_basic_models.MobileNetV2.factory(config=config)

        self.assertRaises(jsonschema.exceptions.ValidationError, load_invalid_config)

    def test_factory(self):
        model = box.factory(config={
            'type': 'ResNet'
        }, tag='model')
        self.assertIsInstance(model, torch_basic_models.ResNet)

    def test_layers(self):
        self.assertIsNotNone(torch_basic_models.layers.Squeeze())
        self.assertIsNotNone(torch_basic_models.layers.UnSqueeze())
        self.assertIsNotNone(torch_basic_models.layers.GlobalPooling())
        self.assertIsNotNone(torch_basic_models.layers.InplaceReLU())
        self.assertIsNotNone(torch_basic_models.layers.InplaceReLU6())

        tensor = torch.randn(3, 4, 5, 6)
        view = torch_basic_models.layers.View
        self.assertEqual(tuple(view(1)(tensor).shape), (360,))
        self.assertEqual(tuple(view(2)(tensor).shape), (3, 120))
        self.assertEqual(tuple(view(3)(tensor).shape), (3, 4, 30))
        self.assertEqual(tuple(view(4)(tensor).shape), (3, 4, 5, 6))
        self.assertEqual(tuple(view(5)(tensor).shape), (3, 4, 5, 6, 1))
        self.assertEqual(tuple(view(6)(tensor).shape), (3, 4, 5, 6, 1, 1))

        un_squeeze = torch_basic_models.layers.UnSqueeze()
        self.assertEqual(tuple(un_squeeze(torch.randn(3, 4)).shape), (3, 4, 1, 1))

        swish = torch_basic_models.layers.Swish
        se_block = torch_basic_models.layers.SELayer(in_channels=4, reduction=2, no_linear=swish)
        self.assertEqual(se_block(tensor).shape, tensor.shape)

    def test_default_batch_norm_2d(self):
        class CustomBN(nn.BatchNorm2d):
            pass

        torch_basic_models.set_default_batch_norm_2d(CustomBN)
        torch_basic_models.reset_default_batch_norm_2d()
        self.assertTrue(torch_basic_models.load_default_batch_norm_2d() is nn.BatchNorm2d)


if __name__ == '__main__':
    unittest.main()
