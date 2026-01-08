import os
from pathlib import Path

def run_cellpose_and_save_mask(ch2_path, save_dir):
    """
    Performs Cellpose segmentation and saves only the resulting mask array (.npy)
    and the mask overlay image (.png).
    """
    print(f"/n--- Segmenting file: {os.path.basename(ch2_path)} ---")
    
    # 1. Load Images (as 2D grayscale)
    ch1_img = imread(ch2_path, as_gray=True)
    
    # 2. Run Cellpose (on the ch2/GFP channel)
    masks, flows, styles = model.eval(
        [ch1_img], 
        diameter=30, 
        flow_threshold=2, 
        cellprob_threshold=1
    )
    current_mask = masks[0]
    
    # Ensure the save directory exists
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    file_stem = Path(ch2_path).stem
    
    # 3. Save the RAW MASK ARRAY (.npy file)
    # This is the most essential output for saving masks
    mask_npy_path = save_path / f"{file_stem}_mask.npy"
    np.save(mask_npy_path, current_mask)
    print(f"Mask array saved to: {mask_npy_path.name}")
    
    # 4. Save the MASK OVERLAY IMAGE (.png) for visual check
    mask_RGB = plot.mask_overlay(ch1_img, current_mask)
    overlay_path = save_path / f"{file_stem}_overlay.png"
    plt.imsave(overlay_path, mask_RGB)
    print(f"Mask overlay saved to: {overlay_path.name}")

    return current_mask, ch1_img # Return mask and image for further analysis


    

def batch_process_cellpose(input_dir, save_dir, extension=".tif"):
    """
    Finds all images in input_dir and runs the segmentation function on each.
    """
    # 1. Get a list of all files with the specified extension
    input_path = Path(input_dir)
    image_files = list(input_path.glob(f"*{extension}"))
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print(f"Found {len(image_files)} files to process.")

    # 2. Loop through each file
    for img_path in image_files:
        # Convert Path object to string for your existing function
        run_cellpose_and_save_mask(str(img_path), save_dir)
        
    print("\n--- All files in folder processed! ---")

if __name__ == "__main__":
    # Example usage:
    # batch_process_cellpose('path/to/images', 'path/to/masks')
    pass