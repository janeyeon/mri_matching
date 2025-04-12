import os
import numpy as np
import pydicom
import napari
from magicgui import magicgui
from qtpy.QtWidgets import QFileDialog, QWidget


def load_dicom_series(folder_path):
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".dcm")]
    if not files:
        raise FileNotFoundError(f"No DICOM files in {folder_path}")

    slices = [pydicom.dcmread(f) for f in files]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]) if "ImagePositionPatient" in x else 0.0)

    image_stack = np.stack([s.pixel_array for s in slices]).astype(np.float32)

    # Normalize
    image_stack -= image_stack.min()
    image_stack /= image_stack.max()

    return image_stack


@magicgui(call_button="Load DICOM Folder")
def dicom_loader_gui(folder_path: str = "/home/yeon/Documents/2025/MRI-matching/Pancreas_MRI_2023/MRI/Case_01_02_01/dicom"):
    if not folder_path:
        folder = QFileDialog.getExistingDirectory(QWidget(), "Select DICOM Folder")
        if not folder:
            print("No folder selected.")
            return
        folder_path = folder

    try:
        volume = load_dicom_series(folder_path)
        viewer.dims.ndisplay = 3  # 🔥 3D view 모드 켜기
        if "MRI Volume" in viewer.layers:
            viewer.layers["MRI Volume"].data = volume
        else:
            viewer.add_image(
                volume,
                name="MRI Volume",
                colormap="gray",
                rendering="attenuated_mip",  # 🔥 진짜 3D 렌더링
                scale=(1, 1, 1),  # 픽셀 spacing, 필요 시 z 간격 맞춰줄 것
                blending="translucent"
            )
        print(f"Loaded DICOM volume from {folder_path}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    viewer = napari.Viewer()
    viewer.window.add_dock_widget(dicom_loader_gui, area="right")
    viewer.dims.ndisplay = 3  # 🔥 시작부터 3D 모드
    napari.run()
