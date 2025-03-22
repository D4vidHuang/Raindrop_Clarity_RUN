import os
import glob
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn as nn

# The same fusion network as in the training process
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

def test_fusion(domain='day', model_checkpoint="fusionnet_epoch_10.pth", output_dir="fused_results"):

    if domain.lower() == 'day':
        folder1 = r"E:/CVPR2025_Raindrop/Test_Reassembled/DiT_Day"
        folder2 = r"E:/CVPR2025_Raindrop/Test_Reassembled/RDiffusion_Day"
    elif domain.lower() == 'night':
        folder1 = r"E:/CVPR2025_Raindrop/Test_Reassembled/DiT_Night"
        folder2 = r"E:/CVPR2025_Raindrop/Test_Reassembled/RDiffusion_Night"
    else:
        raise ValueError("Domain mush be 'day' or 'night'")

    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.ToTensor()

    files1 = sorted(glob.glob(os.path.join(folder1, "*.png")))
    files2 = sorted(glob.glob(os.path.join(folder2, "*.png")))

    num_samples = min(len(files1), len(files2))
    print(f"Found {num_samples} pairs of images.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FusionNet(in_channels=6, base_channels=64, out_channels=3).to(device)
    model.load_state_dict(torch.load(model_checkpoint, map_location=device))
    model.eval()

    with torch.no_grad():
        for i in range(num_samples):
            img1 = Image.open(files1[i]).convert('RGB')
            img2 = Image.open(files2[i]).convert('RGB')
            tensor1 = transform(img1)
            tensor2 = transform(img2)
            fusion_input = torch.cat([tensor1, tensor2], dim=0).unsqueeze(0).to(device)
            output = model(fusion_input)
            output = output.squeeze(0).cpu()
            output_path = os.path.join(output_dir, f"{domain}_{i:05d}.png")
            vutils.save_image(output, output_path)
            print(f"Saved results to: {output_path}")

# Here the domain is only for the path choice, we are using the actual same model so not influencing the model output.
if __name__ == '__main__':
    test_fusion(domain='day', model_checkpoint="fusionnet_epoch_10.pth", output_dir="fused_results_day")
    test_fusion(domain='night', model_checkpoint="fusionnet_epoch_10.pth", output_dir="fused_results_night")
