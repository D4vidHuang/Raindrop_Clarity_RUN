import os
from PIL import Image

def crop_image_to_patches(image, patch_size=(256, 256)):
    patches = []
    x_starts = [0, 232, 464]
    y_starts = [0, 224]
    for y in y_starts:
        for x in x_starts:
            patch = image.crop((x, y, x + patch_size[0], y + patch_size[1]))
            patches.append(patch)
    return patches

def process_folder(input_dir, output_dir, process_first_n=3):
    for subfolder in os.listdir(input_dir):
        subfolder_path = os.path.join(input_dir, subfolder)
        if os.path.isdir(subfolder_path):
            out_subfolder = os.path.join(output_dir, subfolder)
            os.makedirs(out_subfolder, exist_ok=True)
            files = sorted(os.listdir(subfolder_path))
            valid_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
            files_to_process = valid_files
            for file in files_to_process:
                img_path = os.path.join(subfolder_path, file)
                image = Image.open(img_path)

                if image.size != (720,480):
                    print(f"Warning: {img_path} has a size of {image.size}. We expect 720x480.")

                patches = crop_image_to_patches(image)
                file_base = os.path.splitext(file)[0]

                for idx, patch in enumerate(patches, start=1):
                    filename = f"{file_base}_{idx:02d}.png"
                    patch_save_path = os.path.join(out_subfolder, filename)
                    patch.save(patch_save_path)
                    print(f"已保存: {patch_save_path}")

def main():

    base_dir = r"E:/CVPR2025_Raindrop/NightRainDrop_Train"
    drop_input = os.path.join(base_dir, "Drop")
    clear_input = os.path.join(base_dir, "Clear")
    drop_output = os.path.join(base_dir, "Sliced")
    clear_output = os.path.join(base_dir, "gt_Sliced")

    os.makedirs(drop_output, exist_ok=True)
    os.makedirs(clear_output, exist_ok=True)
    process_folder(drop_input, drop_output)
    process_folder(clear_input, clear_output)

    print("Done!")

if __name__ == "__main__":
    main()
