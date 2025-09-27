from PIL import Image
import numpy as np
import SimpleITK as sitk
import os

#link den cho co file mha
label_folder = r"E:\fetal_head_segmentation\datagoc\train\label_mha"

#tach xong luu o day
output_label_folder = r"E:\fetal_head_segmentation\tachslice\labels"
os.makedirs(output_label_folder, exist_ok=True)

#lay danh sach cac file mha
label_files = [f for f in os.listdir(label_folder) if f.endswith(".mha")]

for lbl_file in label_files:
    label_path = os.path.join(label_folder, lbl_file)

    #load label
    label = sitk.ReadImage(label_path)
    label_array = sitk.GetArrayFromImage(label)

    if label_array.ndim == 2:
        #neu anh co 1 lat
        label_slice = label_array
    else:
        #tim slice co mask
        nonzero_indices = np.where(label_array.sum(axis=(1, 2)) > 0)[0]
        if len(nonzero_indices) == 0:
            print(f"{lbl_file}: khong tim thay slice co mask > 0,next")
            continue

        #cai nay de lay slice o giua o nhung cai ma da loc co mask
        mid_index = nonzero_indices[len(nonzero_indices) // 2]
        label_slice = label_array[mid_index]

    #scale (binary 0/1 -> 0/255) de nhin thay mang mau den va mau trang
    label_slice_to_save = (label_slice.astype(np.uint8) * 255)

    #luu outputs lai
    out_path = os.path.join(output_label_folder, lbl_file.replace('.mha', '.png'))
    Image.fromarray(label_slice_to_save).save(out_path)

print("da luu slice giua cua labels")
