import argparse
import numpy as np
from runtorch import WDSR, super_resolve, WDSRpsf
import torch
import matplotlib.pyplot as plt

from CLEAN import hogbom_clean

def parse_arguments():
    parser = argparse.ArgumentParser(description='Super-Resolution Image Reconstruction')
    parser.add_argument('model_name', type=str, help='Path to the trained model file')
    parser.add_argument('datadir', type=str, help='Directory containing the data')
    parser.add_argument('mm', type=str, help='Identifier for the input image file')
    parser.add_argument('--psf', type=str, default=None, help='Path to the PSF file (optional)')
    parser.add_argument('--scale', type=int, default=2, help='Superresolution upscaling factor')
    parser.add_argument('--clean', action='store_true', default=None, help='Image plane CLEAN the dirty image')
    return parser.parse_args()

def plot_images(lr_image, sr_image, hr_image=None, cleaned=None, gamma=0.5, title=''):

    if hr_image is None and cleaned is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    elif hr_image is not None and cleaned is None:
        fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharex=True, sharey=True)
    else:
        fig, axes = plt.subplots(1, 4, figsize=(12, 4.5), sharex=True, sharey=True)

    # Define extents assuming the images represent the same physical area
    extent_lr = [0, lr_image.shape[1], 0, lr_image.shape[0]]
    extent_sr = [0, lr_image.shape[1], 0, lr_image.shape[0]]  # Use LR dimensions for both

    lr_image = np.abs(lr_image)**gamma
    axes[0].imshow(np.abs(lr_image), cmap='Greys', extent=extent_lr, vmax=lr_image.max()*0.1)
    axes[0].set_title('Low-Resolution Image')
    axes[0].axis('off')

    sr_image = sr_image**gamma
    axes[1].imshow(sr_image, cmap='Greys', extent=extent_sr, vmax=sr_image.max()*0.1, vmin=0)
    axes[1].set_title('Super-Resolved Image')
    axes[1].axis('off')

    if hr_image is not None:
        hr_image = hr_image**gamma
        axes[2].imshow(hr_image, cmap='Greys', extent=extent_sr, vmax=hr_image.max()*0.1, vmin=0)
        axes[2].set_title('True Image')
        axes[2].axis('off')

    if cleaned is not None:
        cleaned = np.abs(cleaned)**gamma
        axes[3].imshow(cleaned, cmap='Greys', extent=extent_sr)
        axes[3].set_title('CLEANED')
        axes[3].axis('off')

    plt.title(f'{title}')
    plt.tight_layout()
    plt.show()

def plot_images_fractional(lr_image, sr_image, hr_image=None, diff_image=None, gamma=0.5, title=''):
    if hr_image is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharex=True, sharey=True)

    # Define extents assuming the images represent the same physical area
    extent_lr = [0, lr_image.shape[1], 0, lr_image.shape[0]]
    extent_sr = [0, lr_image.shape[1], 0, lr_image.shape[0]]  # Use LR dimensions for both

    axes[0].imshow(sr_image, cmap='gray', extent=extent_sr, vmax=sr_image.max()*0.01)
    axes[0].set_title('Super-Resolved Image')
    axes[0].axis('off')

    axes[1].imshow(hr_image, cmap='gray', extent=extent_sr, vmax=hr_image.max()*0.01)
    axes[1].set_title('True Image')
    axes[1].axis('off')

    axes[2].imshow(diff_image, cmap='RdBu', extent=extent_sr, vmax=0.1, vmin=-0.1)
    axes[2].set_title('Fractional difference')
    axes[2].axis('off')

    plt.title(f'{title}')
    plt.tight_layout()
    plt.show()


def compare_images(image1, image2, alpha=0.5, cmap1='Reds', cmap2='Blues', title1='Image 1', title2='Image 2'):
    """
    Compare two images by plotting them overlaid with different colormaps and a slider for opacity control.
    
    Parameters:
    -----------
    image1, image2 : numpy.ndarray
        Input images to compare
    alpha : float, optional
        Initial opacity for the overlay (default: 0.5)
    cmap1, cmap2 : str, optional
        Colormaps for the two images (default: 'Reds' and 'Blues')
    title1, title2 : str, optional
        Titles for the images in the legend
    """
    
    # Create figure and subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    image1 = np.abs(image1)**0.5
    image2 = np.abs(image2)**0.5
    
    # Plot first image
    im1 = ax1.imshow(image1, cmap=cmap1)
    ax1.set_title(title1)
    ax1.axis('off')
    
    # Plot second image
    im2 = ax2.imshow(image2, cmap=cmap2)
    ax2.set_title(title2)
    ax2.axis('off')
    
    # Plot overlay
    im3_1 = ax3.imshow(image1, cmap=cmap1)
    im3_2 = ax3.imshow(image2, cmap=cmap2, alpha=alpha)
    ax3.set_title('Overlay Comparison')
    ax3.axis('off')
    
    # Add colorbar
    plt.colorbar(im1, ax=ax1)
    plt.colorbar(im2, ax=ax2)
    
    # Add slider for alpha control
    from matplotlib.widgets import Slider
    ax_slider = plt.axes([0.15, 0.02, 0.7, 0.03])
    slider = Slider(ax_slider, 'Opacity', 0, 1, valinit=alpha)
    
    def update(val):
        im3_2.set_alpha(slider.val)
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=plt.get_cmap(cmap1)(0.7), label=title1),
        Patch(facecolor=plt.get_cmap(cmap2)(0.7), label=title2)
    ]
    ax3.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    return fig, (ax1, ax2, ax3)


def main():
    args = parse_arguments()
    model_name = args.model_name
    datadir = args.datadir
    mm = args.mm
    psf_path = args.psf
    scale = args.scale

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

    if datadir.endswith('npy'):
        filename = datadir
        filename_hr = mm
    elif datadir.endswith('.fits'):
        filename = datadir
    else:
        # Construct the filename for the input low-resolution image
        filename = f'{datadir}/POLISH_valid_LR_bicubic/X2/{mm.zfill(4)}x2.npy'
        filename_hr = f'{datadir}/POLISH_valid_HR/{mm.zfill(4)}.npy'
        
    print(f'Reconstructing {filename}')

    log = False
    
    # Perform super-resolution
    if psf_path is None:
        sr_image, lr_image = super_resolve(model, filename, device, log=log)
    else:
        sr_image, lr_image = super_resolve(model, filename, device, psf=psfarr)

    sr_image = np.array(sr_image)

    np.save('SRimage.npy', sr_image)
    
    print(f'Super-resolved image shape: {sr_image.shape}')

    try:
        hr_image = np.load(filename_hr)[:,:,0]
    except:
        hr_image = None

    # CLEAN image with Hogbom for direct comparison
    if args.clean:
        psf = np.load(f'{datadir}/psf/psf_ideal.npy')[1:args.scale, 1::args.scale]        
        cleaned, residual = hogbom_clean(lr_image, psf, gain=0.1, threshold=0.00005, max_iterations=1000)
        cleaned = (cleaned - cleaned.min()) / (cleaned.max() - cleaned.min())
    else:
        cleaned = None
        
    plot_images(lr_image, sr_image, hr_image, cleaned=cleaned, title=filename)

if __name__ == '__main__':
    main()
