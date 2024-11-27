import sys

import numpy as np
from runtorch import WDSR, super_resolve, WDSRpsf
import torch

import matplotlib.pylab as plt

model_name = sys.argv[1]
datadir = sys.argv[2]
mm = sys.argv[3]

try:
    psf = sys.argv[4]
except:
    psf = None

#model_name = 'lens800x2singleframe.pth'
#model_name = 'final_lens800x2singleframe.pth'
#datadir = './lens800x2singleframe'

device='cuda'

if psf is None:
    model = WDSR(scale_factor=2).to(device)
else:
    model = WDSRpsf(scale_factor=2).to(device)
    psfarr = np.load('./data/exampleLWA1024x2/psf/psf_ideal.npy')
    npsf = len(psfarr)
    psfarr = psfarr[npsf//2-256:npsf//2+256, npsf//2-256:npsf//2+256]
    psfarr = psfarr[None,None] * np.ones([1,1,1,1])
    psfarr = torch.from_numpy(psfarr).to(device).float()
    
model.load_state_dict(torch.load(model_name))

filename = f'{datadir}/POLISH_valid_LR_bicubic/X2/0%sx2.npy' % mm

print(f'Reconstructing {filename}')

if psf is None:
    sr_image = super_resolve(model, filename, 'cuda')
else:
    sr_image = super_resolve(model, filename, 'cuda', psf=psfarr)

np.save('data.npy', np.array(sr_image))

sr_image = np.array(sr_image)

print(sr_image.shape)

fig = plt.figure()
plt.imshow(sr_image, vmax=sr_image.max()*0.01, vmin=0)
plt.show()
