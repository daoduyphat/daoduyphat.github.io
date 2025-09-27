from PIL import Image
import numpy as np
import SimpleITK as sitk
import os

#cho nay chua anh mha
image_folder = r"E:\fetal_head_segmentation\datagoc\train\image_mha"

#luu lai slice
output_folder = r"E:\fetal_head_segmentation\tachslice\images"
os.makedirs(output_folder, exist_ok=True)

#lay danh sach file mha
image_files = [f for f in os.listdir(image_folder) if f.endswith(".mha")]

for fname in image_files:
    image_path = os.path.join(image_folder, fname)

    #load anh
    img = sitk.ReadImage(image_path)
    img_arr = sitk.GetArrayFromImage(img)

    if img_arr.ndim == 2:
        #neu chi co 1 slice
        img_slice = img_arr
    else:
        #lay slice giua
        mid_index = img_arr.shape[0] // 2
        img_slice = img_arr[mid_index]

    #chuan hoa ve [0,255]
    img_slice_norm = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min()) * 255
    img_slice_norm = img_slice_norm.astype(np.uint8)

    #luu anh
    out_path = os.path.join(output_folder, fname.replace('.mha', '.png'))
    Image.fromarray(img_slice_norm).save(out_path)

print("luu het slice giua cua image roi")
