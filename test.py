import os
import pydicom
import numpy as np
import napari

def load_dicom_series(folder_path):
    # DICOM 파일 리스트 정렬
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.dcm')]
    files.sort()  # slice 순서대로 정렬 (중요)

    # 각 DICOM 파일 읽기 및 이미지 추출
    slices = [pydicom.dcmread(f) for f in files]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))  # Z-axis 기준 정렬

    image_stack = np.stack([s.pixel_array for s in slices])
    image_stack = image_stack.astype(np.float32)
    
    # intensity normalization (선택사항)
    image_stack -= np.min(image_stack)
    image_stack /= np.max(image_stack)

    return image_stack

def visualize_with_napari(volume):
    viewer = napari.Viewer()
    viewer.add_image(volume, name="MRI Volume", colormap='gray', rendering='mip', scale=(1, 1, 1))  # MIP rendering
    napari.run()

if __name__ == "__main__":
    dicom_folder = "/home/yeon/Documents/2025/MRI-matching/Pancreas_MRI_2023/MRI/Case_01_02_01/dicom"  # DICOM 폴더 경로
    volume = load_dicom_series(dicom_folder)
    visualize_with_napari(volume)
