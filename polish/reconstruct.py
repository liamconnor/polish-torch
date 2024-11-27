import argparse
import numpy as np
from runtorch import WDSR, super_resolve, WDSRpsf
import torch
import matplotlib.pyplot as plt

def parse_arguments():
    parser = argparse.ArgumentParser(description='Super-Resolution Image Reconstruction')
    parser.add_argument('model_name', type=str, help='Path to the trained model file')
    parser.add_argument('datadir', type=str, help='Directory containing the data')
    parser.add_argument('mm', type=str, help='Identifier for the input image file')
    parser.add_argument('--psf', type=str, default=None, help='Path to the PSF file (optional)')
    return parser.parse_args()

def plot_images(lr_image, sr_image, hr_image=None, gamma=0.5):
    if hr_image is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharex=True, sharey=True)

    # Define extents assuming the images represent the same physical area
    extent_lr = [0, lr_image.shape[1], 0, lr_image.shape[0]]
    extent_sr = [0, lr_image.shape[1], 0, lr_image.shape[0]]  # Use LR dimensions for both

    lr_image = lr_image**gamma
    axes[0].imshow(lr_image, cmap='gray', extent=extent_lr, vmax=lr_image.max()*0.1)
    axes[0].set_title('Low-Resolution Image')
    axes[0].axis('off')

    sr_image = sr_image**gamma
    axes[1].imshow(sr_image, cmap='gray', extent=extent_sr, vmax=sr_image.max()*0.1)
    axes[1].set_title('Super-Resolved Image')
    axes[1].axis('off')

    if hr_image is not None:
        hr_image = hr_image**gamma
        axes[2].imshow(hr_image, cmap='gray', extent=extent_sr, vmax=hr_image.max()*0.1)
        axes[2].set_title('True Image')
        axes[2].axis('off')
        
    plt.tight_layout()
    plt.show()
    
def main():
    args = parse_arguments()
    model_name = args.model_name
    datadir = args.datadir
    mm = args.mm
    psf_path = args.psf

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    print(f"Using {device}")
    
    # Load the appropriate model
    if psf_path is None:
        model = WDSR(scale_factor=2).to(device)
    else:
        model = WDSRpsf(scale_factor=2).to(device)
        # Load PSF array from the provided path
        psfarr = np.load(psf_path)
        npsf = psfarr.shape[0]
        # Center crop the PSF array
        psfarr = psfarr[npsf//2-256:npsf//2+256, npsf//2-256:npsf//2+256]
        psfarr = psfarr[None, None]
        psfarr = torch.from_numpy(psfarr).float().to(device)

    # Load the trained model weights
    model.load_state_dict(torch.load(model_name))
    model.eval()

    # Construct the filename for the input low-resolution image
    filename = f'{datadir}/POLISH_valid_LR_bicubic/X2/0{mm}x2.npy'
    filename_hr = f'{datadir}/POLISH_valid_HR/0{mm}.npy'    
    print(f'Reconstructing {filename}')

    log = False
    # Perform super-resolution
    if psf_path is None:
        sr_image = super_resolve(model, filename, device, log=log)
    else:
        sr_image = super_resolve(model, filename, device, psf=psfarr)
        
    sr_image = np.array(sr_image)

    if log==True:
        sr_image[np.isnan(sr_image)] = 1e-32
        sr_image = np.exp(sr_image)

    print(f'Super-resolved image shape: {sr_image.shape}')

    # Load the low-resolution image
    lr_image = np.load(filename)
    hr_image = np.load(filename_hr)
    
    # Plot the low-resolution and super-resolved images side by side
    plot_images(lr_image, sr_image, hr_image)

if __name__ == '__main__':
    main()
