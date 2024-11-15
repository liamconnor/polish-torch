import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import torchvision.transforms.functional as TF
import torch.nn.functional as F

import matplotlib.pylab as plt

# WDSR model definition
class WDSR(nn.Module):
    def __init__(self, num_residual_blocks=32, num_features=32, scale_factor=2):
        super(WDSR, self).__init__()
        self.scale_factor = scale_factor
        
        # Initial convolution
        self.conv_first = nn.Conv2d(1, num_features, kernel_size=3, padding=1)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            WDSRBlock(num_features) for _ in range(num_residual_blocks)
        ])
        
        # Upsampling
        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, num_features * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor)
        )
        
        # Final convolution
        self.conv_last = nn.Conv2d(num_features, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv_first(x)
        residual = x
        for block in self.residual_blocks:
            x = block(x)
        x += residual
        x = self.upsample(x)
        x = self.conv_last(x)
        return x

class WDSRBlock(nn.Module):
    def __init__(self, num_features):  # Use double underscores for __init__
        super(WDSRBlock, self).__init__()  # Use double underscores for __init__
        self.conv1 = nn.Conv2d(num_features, num_features * 4, stride=1, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_features * 4, num_features, stride=1, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x += residual
        return x

class WDSRpsf(nn.Module):
    def __init__(self, num_residual_blocks=32, num_features=32, scale_factor=2):
        super(WDSRpsf, self).__init__()
        self.scale_factor = scale_factor
        
        # Initial convolution (now accepts 2 channels: image and PSF)
        self.conv_first = nn.Conv2d(2, num_features, kernel_size=3, padding=1)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            WDSRBlockpsf(num_features) for _ in range(num_residual_blocks)
        ])
        
        # Upsampling
        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, num_features * (scale_factor ** 2), 
            kernel_size=3, padding=1, stride=1),
            nn.PixelShuffle(scale_factor)
        )
        
        # Final convolution
        self.conv_last = nn.Conv2d(num_features, 1, kernel_size=3, stride=1, padding=1)
        
        # Remove the psf_conv layer as we'll apply the PSF differently
        
    def forward(self, x, psf):
        # Resize PSF to match input image dimensions
        psf_resized = F.interpolate(psf, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # Combine input image and resized PSF
        x = torch.cat([x, psf_resized], dim=1)
        
        x = self.conv_first(x)
        residual = x
        for block in self.residual_blocks:
            x = block(x)
        x += residual
        x = self.upsample(x)
        x = self.conv_last(x)
        
        # Apply PSF convolution
        # Ensure PSF is the right size for convolution
        if psf.shape[2:] != (3, 3):
            psf = F.interpolate(psf, size=(3, 3), mode='bilinear', align_corners=False)
        
        # Apply PSF convolution manually for each item in the batch
        batch_size = x.shape[0]
        output = []
        for i in range(batch_size):
            current_psf = psf[i]
            output.append(F.conv2d(x[i].unsqueeze(0), current_psf.unsqueeze(0), padding=1, stride=1))
        
        x = torch.cat(output, dim=0)
        
        return x

class WDSRBlockpsf(nn.Module):
    def __init__(self, num_features):  # Use double underscores for __init__
        super(WDSRBlockpsf, self).__init__()  # Use double underscores for __init__
        self.conv1 = nn.Conv2d(num_features, num_features * 4, stride=1, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_features * 4, num_features, stride=1, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x += residual
        return x

class SuperResolutionDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, start_num, end_num,
                 crop_size=256, transform=None, scale_factor=2):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.transform = transform
        self.image_files = [f"{i:04d}.npy" for i in range(start_num, end_num + 1)]
        self.crop_size = crop_size
        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        #hr_image = Image.open(os.path.join(self.hr_dir, img_name))
        #lr_image = Image.open(os.path.join(self.lr_dir, img_name.replace('.png', 'x%d.png' % self.scale_factor)))

        hr_image = np.load(os.path.join(self.hr_dir, img_name))
        lr_image = np.load(os.path.join(self.lr_dir, img_name.replace('.npy', 'x%d.npy' % self.scale_factor)))

        hr_image = torch.from_numpy(hr_image).squeeze()
        lr_image = torch.from_numpy(lr_image).squeeze()
        
        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(hr_image, output_size=(self.crop_size, self.crop_size))
        hr_image = TF.crop(hr_image, i, j, h, w)
        lr_image = TF.crop(lr_image, i // self.scale_factor, j // self.scale_factor,
                           h // self.scale_factor, w // self.scale_factor)  # Adjust for LR size
        
        # Convert to numpy array and normalize
        hr_image = np.array(hr_image).astype(np.float32) #/ float(hr_image.max())  # Normalize 16-bit to [0, 1]
        lr_image = np.array(lr_image).astype(np.float32) #/ float(lr_image.max())
        
        # Convert to tensor
        hr_image = torch.from_numpy(hr_image).unsqueeze(0)
        lr_image = torch.from_numpy(lr_image).unsqueeze(0)
        
        if self.transform:
            hr_image = self.transform(hr_image)
            lr_image = self.transform(lr_image)

        return lr_image, hr_image

# Training function
def train(model, train_loader, criterion, optimizer, device, psf=None):
    model.train()
    running_loss = 0.0
    for lr_imgs, hr_imgs in train_loader:
        lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
        
        optimizer.zero_grad()
        if psf is None:
            outputs = model(lr_imgs)
        else:
            outputs = model(lr_imgs, psf)
        #print(lr_imgs)
        #print(outputs)
        loss = criterion(outputs, hr_imgs)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(train_loader)

# Validation function
def validate(model, val_loader, criterion, device, psf=None):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for lr_imgs, hr_imgs in val_loader:
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

            if psf is None:
                outputs = model(lr_imgs)
            else:
                outputs = model(lr_imgs, psf)

            loss = criterion(outputs, hr_imgs)
            running_loss += loss.item()
    
    return running_loss / len(val_loader)

# Assuming you have a PSNR calculation function. If not, I'll provide one.
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# Main execution
def main(datadir, scale=2, model_name=None, psf=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters
    num_epochs = 500
    batch_size = 4
    learning_rate = 0.0001
    
    # Create model
    if psf:
        model = WDSRpsf(scale_factor=scale).to(device)
        psfarr = np.load('./data/exampleLWA1024x2/psf/psf_ideal.npy')
        npsf = len(psfarr)
        psfarr = psfarr[npsf//2-256:npsf//2+256, npsf//2-256:npsf//2+256]
        psfarr = psfarr[None,None] * np.ones([batch_size,1,1,1])
        psfarr = torch.from_numpy(psfarr).to(device).float()
    else:
        model = WDSR(scale_factor=scale).to(device)
        psfarr = None



    
    if model_name != None:
        model.load_state_dict(torch.load(model_name))
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Load datasets
    train_dataset = SuperResolutionDataset('./%s/POLISH_train_HR/' % datadir, './%s/POLISH_train_LR_bicubic/X%d/' % (datadir,scale), 0, 799, scale_factor=scale)
    val_dataset = SuperResolutionDataset('./%s/POLISH_valid_HR/' % datadir, './%s/POLISH_valid_LR_bicubic/X%d/' % (datadir, scale), 800, 899, scale_factor=scale)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Training loop
    best_val_loss = float('inf')
    best_val_psnr = 0.0
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device, psf=psfarr)
        val_loss, val_psnr = validate_with_psnr(model, val_loader, criterion, device, psf=psfarr)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val PSNR: {val_psnr:.2f}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'runs/%s.pth' % datadir.strip('/').split('/')[-1])

        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            torch.save(model.state_dict(), 'runs/%s_PSNR.pth' % datadir.strip('/').split('/')[-1])
            
    # Save the final model
    torch.save(model.state_dict(), 'runs/final_%s.pth' % datadir.strip('/').split('/')[-1])

# Modified validation function to include PSNR calculation
def validate_with_psnr(model, val_loader, criterion, device, psf=None):
    model.eval()
    total_loss = 0
    total_psnr = 0
    with torch.no_grad():
        for lr_imgs, hr_imgs in val_loader:
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
            if psf is None:
                sr_imgs = model(lr_imgs)
            else:
                sr_imgs = model(lr_imgs, psf)
            loss = criterion(sr_imgs, hr_imgs)
            total_loss += loss.item()
            
            # Calculate PSNR
            for sr, hr in zip(sr_imgs, hr_imgs):
                sr_np = sr.cpu().numpy().transpose(1, 2, 0)  # CHW to HWC
                hr_np = hr.cpu().numpy().transpose(1, 2, 0)  # CHW to HWC
                total_psnr += calculate_psnr(sr_np, hr_np)
    
    avg_loss = total_loss / len(val_loader)
    avg_psnr = total_psnr / (len(val_loader) * val_loader.batch_size)
    return avg_loss, avg_psnr

if __name__=='__main__':
    try:
        model_name = sys.argv[3]
    except:
        model_name = None
    main(sys.argv[1], int(sys.argv[2]), model_name=model_name)

# Inference function (for using the trained model)
def super_resolve(model, lr_image_path, device, psf=None):
    model.eval()

    if lr_image_path.endswith('.png'):
        lr_image = Image.open(lr_image_path)
        lr_image = np.array(lr_image).astype(np.float32) / 65535.0
        lr_image = torch.from_numpy(lr_image).unsqueeze(0).unsqueeze(0).to(device)
    elif lr_image_path.endswith('.npy'):
        lr_image = np.load(lr_image_path)[:,:,0].astype(np.float32)
        lr_image = torch.from_numpy(lr_image).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        if psf is None:
            sr_image = model(lr_image)
        else:
            print(lr_image.shape, psf.shape)
            sr_image = model(lr_image, psf)
    
    sr_image = sr_image.squeeze().cpu().numpy()
    sr_image = (sr_image * 65535.0).clip(0, 65535).astype(np.uint16)
    return Image.fromarray(sr_image)

# Example usage of inference
# model = WDSR().to(device)
# model.load_state_dict(torch.load('wdsr_model.pth'))
# sr_image = super_resolve(model, 'path/to/test/lr_image.png', device)
# sr_image.save('super_resolved_image.png')
