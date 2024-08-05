# polish-torch

## Installation

This project supports both Poetry and pip for package management.

### Using Poetry (recommended)

1. Install Poetry if you haven't already: 
2. poetry install (this will create the environment)
3. poetry shell (this will activate that environemnt)
4. To add new packages: poetry add package-name

## Example usage:

### Forward model true sky / dirty image pairs:

A DSA-2000 example:
python polish/make_img_pairs.py -k ./psf/dsa-2000-fullband-psf.fits -o ./data/DSA2000_1024_x2/ --nside 1024 -r 2 -s 1024

An LWA example:
python polish/make_img_pairs.py -k ./psf/lwa.briggs0-psf.fits -o ./data/exampleLWA1024x2/ --nside 2048 -r 2 -s 2048 -p --pix 60.0 --src_density 2

### Train a POLISH model on simulated data:
python polish/runtorch.py ./data/DSA2000_1024_x2/ 2

If you'd like to fine tune an existing model (for example, a slightly different pointing and therefore a different but similar PSF):

python polish/runtorch.py ./data/DSA2000_1024_x2/ 2 path_to_model.pth

