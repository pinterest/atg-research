"""
Custom ops for pinterest
By importing this module, you can use the custom ops / classes from
torch.ops.pinterst_ops.[FUNCTIONS] / torch.classes.pinterst_ops.[CLASSES].
"""

import logging
import os

import torch

base_path = os.path.dirname(__file__)
LOG = logging.getLogger(__name__)
LOG.setLevel(level=logging.INFO)

# loading operators
ops_lib_path = os.path.join(base_path, "liboperators.so")
torch.ops.load_library(ops_lib_path)
