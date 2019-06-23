# torch-basic-models [![Build Status](https://travis-ci.com/FebruaryBreeze/torch-basic-models.svg?branch=master)](https://travis-ci.com/FebruaryBreeze/torch-basic-models) [![codecov](https://codecov.io/gh/FebruaryBreeze/torch-basic-models/branch/master/graph/badge.svg)](https://codecov.io/gh/FebruaryBreeze/torch-basic-models) [![PyPI version](https://badge.fury.io/py/torch-basic-models.svg)](https://pypi.org/project/torch-basic-models/)

Basic Models for PyTorch, with Unified Interface

## Installation

Need Python 3.6+.

```bash
pip install torch-basic-models
```

## Usage

```python
import torch_basic_models

ResNet = torch_basic_models.ResNet

# or

import box

ResNet = box.load(name='ResNet', tag='model')
```
