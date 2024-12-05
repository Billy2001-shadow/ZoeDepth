# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

import torch
# import numpy as np
# from torchvision.transforms import ToTensor
# from PIL import Image
# from zoedepth.utils.misc import get_image_from_url, colorize

# from zoedepth.models.builder import build_model
# from zoedepth.utils.config import get_config
# from pprint import pprint



# Trigger reload of MiDaS
# torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True) 


# model = torch.hub.load(".", "ZoeD_K", source="local", pretrained=True)
# model = torch.hub.load(".", "ZoeD_NK", source="local", pretrained=True)
# model = torch.hub.load(".", "ZoeD_N", source="local", pretrained=True)
# model_zoe_n = torch.hub.load("/home/chenwu/ZoeDepth/pretrained/hub/isl-org-MiDaS-4545977", "ZoeD_N", source="local", pretrained=True)

# 设置本地模型目录
local_model_dir = "/home/chenwu/ZoeDepth/pretrained/hub/MiDaS"

# 加载 MiDaS 模型
midas_model = torch.hub.load(local_model_dir, "DPT_BEiT_L_384", source="local", pretrained=True)

# 加载 ZoeDepth 模型
model_zoe_n = torch.hub.load(".", "ZoeD_N", source="local", pretrained=True)