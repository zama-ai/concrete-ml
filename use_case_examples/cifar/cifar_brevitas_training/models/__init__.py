# MIT License
#
# Copyright (c) 2019 Xilinx
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# Original file can be found at https://github.com/Xilinx/brevitas/blob/8c3d9de0113528cf6693c6474a13d802a66682c6/src/brevitas_examples/bnn_pynq/models/__init__.py

import os
from configparser import ConfigParser

import torch
from torch import hub

__all__ = ["cnv_2w2a"]

from .model import cnv

model_impl = {
    "CNV": cnv,
}


def get_model_cfg(name):
    cfg = ConfigParser()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, name.lower() + ".ini")
    assert os.path.exists(config_path)
    cfg.read(config_path)
    return cfg


def model_with_cfg(name, pre_trained):
    cfg = get_model_cfg(name)
    arch = cfg.get("MODEL", "ARCH")
    model = model_impl[arch](cfg)
    if pre_trained:
        checkpoint = cfg.get("MODEL", "PRETRAINED_URL")
        state_dict = hub.load_state_dict_from_url(checkpoint, progress=True, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
    return model, cfg


def cnv_2w2a(pre_trained=False):
    assert (
        pre_trained == False
    ), "No online pre-trained network are available. Use --resume instead with a valid checkpoint."
    model, _ = model_with_cfg("cnv_2w2a", pre_trained)
    return model
