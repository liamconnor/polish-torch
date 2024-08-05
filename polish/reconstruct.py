import sys

import numpy as np
from runtorch import WDSR, super_resolve
import torch

mm = sys.argv[1]

model_name = 'lens800x2singleframe.pth'
#model_name = 'final_lens800x2singleframe.pth'
datadir = './lens800x2singleframe'

device='cuda'
model = WDSR(scale_factor=2).to(device)

model.load_state_dict(torch.load(model_name))

filename = f'{datadir}/POLISH_valid_LR_bicubic/X2/0%sx2.npy' % mm

print(f'Reconstructing {filename}')

sr_image = super_resolve(model, filename, 'cuda')

np.save('data.npy', np.array(sr_image))
