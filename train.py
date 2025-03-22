import os
import glob
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
import lpips
import numpy as np

# Here is the loss, same as what used in the Raindrop Clarity Paper
loss_fn_vgg = lpips.LPIPS(net='vgg')

def compute_psnr_torch(pred, target):
    mse = F.mse_loss(pred, target)
    psnr = 10 * torch.log(1.0 / (mse + 1e-8)) / torch.log(torch.tensor(10.0, device=pred.device))
    return psnr

def gaussian_window(window_size, sigma, device):
    gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/(2*sigma**2)) for x in range(window_size)]).to(device)
    return gauss / gauss.sum()

def create_window(window_size, channel, device):
    _1D_window = gaussian_window(window_size, 1.5, device).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def compute_ssim_torch(img1, img2, window_size=11, size_average=True):
    device = img1.device
    channel = img1.size(1)
    window = create_window(window_size, channel, device)
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = (0.01) ** 2
    C2 = (0.03) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def compute_lpips(pred, target):
    device_lpips = next(loss_fn_vgg.parameters()).device
    if pred.dim() == 4:
        batch_size = pred.shape[0]
        lpips_sum = 0.0
        for i in range(batch_size):
            sample_pred = (pred[i].unsqueeze(0) * 2 - 1).to(device_lpips)
            sample_target = (target[i].unsqueeze(0) * 2 - 1).to(device_lpips)
            lpips_val = loss_fn_vgg(sample_pred, sample_target)
            lpips_sum += lpips_val.mean()
        return lpips_sum / batch_size
    else:
        sample_pred = (pred.unsqueeze(0) * 2 - 1).to(device_lpips)
        sample_target = (target.unsqueeze(0) * 2 - 1).to(device_lpips)
        lpips_val = loss_fn_vgg(sample_pred, sample_target)
        return lpips_val.mean()

def to_y_channel_torch(img):
    if img.dim() == 3:
        R = img[0]
        G = img[1]
        B = img[2]
        Y = 16/255.0 + (65.738 * R + 129.057 * G + 25.064 * B) / 256.0
    elif img.dim() == 4:
        R = img[:, 0, :, :]
        G = img[:, 1, :, :]
        B = img[:, 2, :, :]
        Y = 16/255.0 + (65.738 * R + 129.057 * G + 25.064 * B) / 256.0
    return Y

def fusion_loss_torch(pred, target):
    pred_y = to_y_channel_torch(pred)
    target_y = to_y_channel_torch(target)
    if pred_y.dim() == 3:
        pred_y = pred_y.unsqueeze(1)
        target_y = target_y.unsqueeze(1)
    psnr = compute_psnr_torch(pred_y, target_y)
    ssim = compute_ssim_torch(pred_y, target_y)
    lpips_val = compute_lpips(pred, target)
    loss = - psnr - 10 * ssim + 5 * lpips_val
    return loss

# This is a very basic (even silly) convolution block, but actually this is working better than all other implementations, like attention/transformer/unet, etc.
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

# This is the FusionNet, input 6 channels (2 images), output 3 channels (RGB)
class FusionNet(nn.Module):
    def __init__(self, in_channels=6, base_channels=64, out_channels=3):
        super(FusionNet, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.fusion_block1 = ConvBlock(base_channels, base_channels)
        self.fusion_block2 = ConvBlock(base_channels, base_channels)
        self.final = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
    def forward(self, x):
        x = self.initial(x)
        x = self.fusion_block1(x)
        x = self.fusion_block2(x)
        out = self.final(x)
        return out

# This is the dataset class for training the FusionNet
class FusionDataset(Dataset):
    def __init__(self, domain_configs, transform=None):
        self.domain_configs = domain_configs
        self.transform = transform
        self.index_mapping = []
        for d_idx, config in enumerate(domain_configs):
            gt_dir = config["gt"]
            m1_dir = config["model1"]
            m2_dir = config["model2"]
            sample_folders = sorted(os.listdir(gt_dir))
            for sample in sample_folders:
                gt_sample_dir = os.path.join(gt_dir, sample)
                m1_sample_dir = os.path.join(m1_dir, sample)
                m2_sample_dir = os.path.join(m2_dir, sample)
                if not (os.path.isdir(gt_sample_dir) and os.path.isdir(m1_sample_dir) and os.path.isdir(m2_sample_dir)):
                    continue
                gt_imgs = sorted(glob.glob(os.path.join(gt_sample_dir, '*')))
                m1_imgs = sorted(glob.glob(os.path.join(m1_sample_dir, '*')))
                m2_imgs = sorted(glob.glob(os.path.join(m2_sample_dir, '*')))
                if len(gt_imgs) == 0 or len(m1_imgs) == 0 or len(m2_imgs) == 0:
                    continue
                min_count = min(len(gt_imgs), len(m1_imgs), len(m2_imgs))
                for i in range(min_count):
                    self.index_mapping.append((d_idx, sample, i))
    def __len__(self):
        return len(self.index_mapping)
    def __getitem__(self, idx):
        d_idx, sample, img_idx = self.index_mapping[idx]
        config = self.domain_configs[d_idx]
        gt_sample_dir = os.path.join(config["gt"], sample)
        m1_sample_dir = os.path.join(config["model1"], sample)
        m2_sample_dir = os.path.join(config["model2"], sample)
        gt_imgs = sorted(glob.glob(os.path.join(gt_sample_dir, '*')))
        m1_imgs = sorted(glob.glob(os.path.join(m1_sample_dir, '*')))
        m2_imgs = sorted(glob.glob(os.path.join(m2_sample_dir, '*')))
        gt_img_path = gt_imgs[img_idx]
        m1_img_path = m1_imgs[img_idx]
        m2_img_path = m2_imgs[img_idx]
        gt_img = Image.open(gt_img_path).convert('RGB')
        m1_img = Image.open(m1_img_path).convert('RGB')
        m2_img = Image.open(m2_img_path).convert('RGB')
        if self.transform:
            gt_img = self.transform(gt_img)
            m1_img = self.transform(m1_img)
            m2_img = self.transform(m2_img)
        fusion_input = torch.cat([m1_img, m2_img], dim=0)
        return fusion_input, gt_img

def train_model(model, dataloader, optimizer, num_epochs=10, device='cuda'):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i, (fusion_input, gt) in enumerate(dataloader):
            fusion_input = fusion_input.to(device)
            gt = gt.to(device)
            optimizer.zero_grad()
            output = model(fusion_input)
            loss = fusion_loss_torch(output, gt)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        checkpoint_path = f"fusionnet_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

# So we are using day and night data to train the same model.
def main_train():
    domain_configs = [
        {
            "gt": r"E:/CVPR2025_Raindrop/DayRainDrop_Train/Clear",
            "model1": r"E:/CVPR2025_Raindrop/Train/Reassembled/DiT_Day",
            "model2": r"E:/CVPR2025_Raindrop/Train/Reassembled/RDiffusion_Day"
        },
        {
            "gt": r"E:/CVPR2025_Raindrop/NightRainDrop_Train/Clear",
            "model1": r"E:/CVPR2025_Raindrop/Train/Reassembled/DiT_Night",
            "model2": r"E:/CVPR2025_Raindrop/Train/Reassembled/RDiffusion_Night"
        }
    ]
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = FusionDataset(domain_configs, transform=transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn_vgg.to(device)
    model = FusionNet(in_channels=6, base_channels=64, out_channels=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    print("Training...")
    train_model(model, dataloader, optimizer, num_epochs=10, device=device)
    print("Training complete.")

if __name__ == '__main__':
    main_train()
