import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from configs import g_conf
from moco_v3.moco.builder import MoCo
import torchvision

class Build_MoCo(nn.Module):
  def __init__(self):
    super(Build_MoCo, self).__init__()
    
    self.moco = MoCo(torchvision.models.__dict__['resnet50'])
  
    model = torchvision.models.resnet50()
    model.layer4.fc = nn.Identity(2048, 2048)
    self.moco.base_encoder.fc = nn.Identity()
    
    checkpoint = torch.load('moco_v3/checkpoints_256_7_100_horiz_enhanced_color/checkpoint_0175.pth.tar')
    checkpoint_dict = checkpoint['state_dict']
  
    for k in list (checkpoint_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('moco_v3.module.encoder_q') and not k.startswith('moco_v3.module.encoder_q.fc'):
            # remove prefix
            k_no_prefix = k[len("module."):]
            checkpoint_dict[k_no_prefix] = checkpoint_dict[k]   # leave encoder_q in the param name
            # copy state from the query encoder into a new parameter for the key encoder
            checkpoint_dict[k_no_prefix.replace('moco_v3.encoder_q', 'moco_v3.encoder_k')] = checkpoint_dict[k]
        del checkpoint_dict[k]

    self.moco.load_state_dict(checkpoint_dict, strict=False)
    
    