from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import torch
import torch.nn as nn



checkpoint = torch.load("/home/josmy/Code/HIT_ucf/data/models/pretrained_models/checkpoint.pth", map_location='cpu')

args.checkpoint = checkpoint

print(checkpoint['module'].keys())