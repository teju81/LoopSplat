import contextlib
import io
import sys

import torch
from einops import *

sys.path.append("thirdparty/Hierarchical-Localization")

with contextlib.redirect_stderr(io.StringIO()):
    from hloc.extractors.netvlad import NetVLAD
    from hloc.extractors.superpoint import SuperPoint


class GlobalDesc:

    def __init__(self):
        conf = {
                'output': 'global-feats-netvlad',
                'model': {'name': 'netvlad'},
                'preprocessing': {'resize_max': 1024},
            }
        self.netvlad = NetVLAD(conf).to('cuda').eval()

    @torch.no_grad()
    def __call__(self, images):
        assert parse_shape(images, '_ rgb _ _') == dict(rgb=3)
        assert (images.dtype == torch.float) and (images.max() <= 1.0001), images.max()
        return self.netvlad({'image': images})['global_descriptor'] # B 4096



class LocalDesc:

    def __init__(self):
        conf = {
                "output": "feats-superpoint-n4096-rmax1600",
                "model": {
                    "name": "superpoint",
                    "nms_radius": 3,
                    "max_keypoints": 4096,
                },
                "preprocessing": {
                    "grayscale": True,
                    "resize_max": 1600,
                    "resize_force": True,
                },
            }
        self.superpoint = SuperPoint(conf).to('cuda').eval()

    @torch.no_grad()
    def __call__(self, images):
        assert parse_shape(images, '_ rgb _ _') == dict(rgb=1)
        assert (images.dtype == torch.float) and (images.max() <= 1.0001), images.max()
        superpoint_output_dict = self.superpoint({'image': images})
        keypoints = superpoint_output_dict['keypoints']
        descriptors = superpoint_output_dict['descriptors']
        scores = superpoint_output_dict['scores']
        return keypoints, descriptors, scores