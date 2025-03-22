import os
from PIL import Image

# Define the location of patches and output them into a 
patch_positions = {
    1: (0, 0),
    2: (232, 0),
    3: (464, 0),
    4: (0, 224),
    5: (232, 224),
    6: (464, 224)
}

def reassemble_patches(patch_files, output_file):
    full_img = Image.new('RGB', (720, 480))

    for i, patch_path in enumerate(patch_files, start=1):
        patch = Image.open(patch_path)
        pos = patch_positions[i]
        full_img.paste(patch, pos)

    full_img.save(output_file)
    print(f"Restored Image: {output_file}")

def reassemble_from_folders(input_root, output_root):
    os.makedirs(output_root, exist_ok=True)

    for folder in os.listdir(input_root):
        input_folder = os.path.join(input_root, folder)
        output_folder = os.path.join(output_root, folder)

        if not os.path.isdir(input_folder):
            continue 

        os.makedirs(output_folder, exist_ok=True)

        groups = {}
        for file in os.listdir(input_folder):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                parts = file.split('_')
                if len(parts) < 2:
                    continue
                prefix = parts[0]
                groups.setdefault(prefix, []).append(file)

        for prefix, files in groups.items():
            if len(files) != 6:
                print(f"No enough patches for {prefix}")
                continue

            files_sorted = sorted(files, key=lambda x: int(os.path.splitext(x)[0].split('_')[1]))
            patch_files = [os.path.join(input_folder, f) for f in files_sorted]
            output_file = os.path.join(output_folder, f"{prefix}.png")
            reassemble_patches(patch_files, output_file)

def main():
    input_root = r"E:/CVPR2025_Raindrop/Train/DiT_Night/output"  
    output_root = r"E:/CVPR2025_Raindrop/Train/Reassembled/DiT_Night"

    reassemble_from_folders(input_root, output_root)
    print("Done!")

if __name__ == "__main__":
    main()
