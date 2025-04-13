import os
import numpy as np
import pydicom
import napari
from magicgui import magicgui
from qtpy.QtWidgets import QFileDialog, QWidget

import numpy as np
from scipy.ndimage import map_coordinates

from skimage.measure import marching_cubes





def load_dicom_series(folder_path):
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".dcm")]
    if not files:
        raise FileNotFoundError(f"No DICOM files in {folder_path}")
    slices = [pydicom.dcmread(f) for f in files]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]) if "ImagePositionPatient" in x else 0.0)

    image_stack = np.stack([s.pixel_array for s in slices]).astype(np.float32)
    image_stack -= image_stack.min()
    image_stack /= image_stack.max()
    return image_stack

PATH = "/home/yeon/Documents/2025/MRI-matching/Pancreas_MRI_2023/MRI/Case_01_02_01/dicom"
SHAPE = load_dicom_series(PATH).shape


# 평면의 4점 좌표 계산 (직사각형)
def get_plane_outline(position, plane_x, plane_y, width=SHAPE[0], height=SHAPE[1]):
    # 중심 기준으로 네 꼭짓점 좌표 계산
    cx, cy = width / 2, height / 2
    corners = np.array([
        -cx * plane_x - cy * plane_y,
         cx * plane_x - cy * plane_y,
         cx * plane_x + cy * plane_y,
        -cx * plane_x + cy * plane_y
    ])
    corners += position
    return corners[:, [2, 1, 0]]  # napari: x, y, z → z, y, x




def slice_volume_with_plane(volume, plane_center, normal, plane_x=None, 
                              plane_size=(256, 256), spacing=1.0, in_plane_offset=(0.0, 0.0), order=1):

    # 일단 normalize부터 시작 
    normal = np.array(normal, dtype=float)
    normal /= (np.linalg.norm(normal) + 1e-8)

    # plane을 결정하는데 우선 주어진 normald에 가장 덜 평행한 축으로 자르게 만듦 -> v 
    if plane_x is None:
        
        v = np.array([0, 1, 0], dtype=float) if np.allclose(normal, [1, 0, 0]) else np.array([1, 0, 0], dtype=float)
        plane_x = np.cross(normal, v)
        plane_x /= (np.linalg.norm(plane_x) + 1e-8)
    else:
        plane_x = np.array(plane_x, dtype=float)
        # plane_x를 normal에서 projection d된 부분을 제거함  완전히 평면 위로
        plane_x -= np.dot(plane_x, normal) * normal
        norm_px = np.linalg.norm(plane_x)
        if norm_px < 1e-6:
            raise ValueError("plane_x is nearly parallel to normal or zero-length!")
        plane_x /= norm_px

    plane_y = np.cross(normal, plane_x)
    plane_y /= (np.linalg.norm(plane_y) + 1e-8)

    # offsㄷt 만큼 이동해줌 -> 근데 이게 지금 상황에 왜 필요한지 탐구할 필요 있음 
    pc = np.array(plane_center, dtype=float).copy()
    ox, oy = in_plane_offset
    pc = pc + ox * plane_x + oy * plane_y  # 평면 내부 offset

    # 원하는 plane 사이즈에맞는 grid를 생성하기 
    w, h = plane_size
    xs = np.arange(w, dtype=float) - (w - 1)/2.0  # -127.5 ~ +128.5 (예시)
    ys = np.arange(h, dtype=float) - (h - 1)/2.0
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")

    grid_x *= spacing
    grid_y *= spacing

    # grid에서 coordinate만들기 
    coords_z = pc[0] + grid_y * plane_y[0] + grid_x * plane_x[0]
    coords_y = pc[1] + grid_y * plane_y[1] + grid_x * plane_x[1]
    coords_x = pc[2] + grid_y * plane_y[2] + grid_x * plane_x[2]

    coords = np.array([coords_z, coords_y, coords_x], dtype=float)  # shape=(3, h, w)

    # volume 위치에  해당하는 slice를 추출하기 
    slice_img = my_map_coordinates(volume, coords, order=order, mode='nearest')
    # shape=(h, w)

    # 3d 전체에 먹일 plane의 3d point를 저장해서 return
    slice_points_3d = np.stack([coords_z, coords_y, coords_x], axis=-1)

    slice_points_3d = slice_points_3d.reshape(-1, 3)

    return slice_img, slice_points_3d


def my_map_coordinates(volume, coords, order=1, mode='nearest', cval=0.0):
    """
    Custom implementation for order=1 (trilinear interpolation) and mode 'nearest'
    volume: 3D np.ndarray with shape (Z, Y, X)
    coords: np.ndarray with shape (3, h, w) representing the (z, y, x) coordinates
            at which to sample the volume.
    Returns:
        An array of shape (h, w) with the interpolated values.
    Only order=1 and mode='nearest' are implemented.
    """
    if order != 1:
        raise NotImplementedError("Only order=1 is implemented.")
    if mode != 'nearest':
        raise NotImplementedError("Only mode='nearest' is implemented.")
    
    # volume dimensions
    Z, Y, X = volume.shape
    h, w = coords.shape[1:]  # output shape

    # Extract fractional coordinate arrays
    zf = coords[0]  # shape (h, w)
    yf = coords[1]
    xf = coords[2]
    
    # Compute floor (lower indices) and ceil (upper indices)
    z0 = np.floor(zf).astype(np.int64)
    y0 = np.floor(yf).astype(np.int64)
    x0 = np.floor(xf).astype(np.int64)
    z1 = z0 + 1
    y1 = y0 + 1
    x1 = x0 + 1

    # Clip indices to valid volume bounds (mode 'nearest')
    z0 = np.clip(z0, 0, Z-1)
    z1 = np.clip(z1, 0, Z-1)
    y0 = np.clip(y0, 0, Y-1)
    y1 = np.clip(y1, 0, Y-1)
    x0 = np.clip(x0, 0, X-1)
    x1 = np.clip(x1, 0, X-1)

    # Compute fractional differences
    dz = zf - z0
    dy = yf - y0
    dx = xf - x0

    # Retrieve corner voxel values for each coordinate (shape (h, w))
    c000 = volume[z0, y0, x0]
    c001 = volume[z0, y0, x1]
    c010 = volume[z0, y1, x0]
    c011 = volume[z0, y1, x1]
    c100 = volume[z1, y0, x0]
    c101 = volume[z1, y0, x1]
    c110 = volume[z1, y1, x0]
    c111 = volume[z1, y1, x1]

    # Trilinear interpolation
    c00 = c000 * (1 - dx) + c001 * dx
    c01 = c010 * (1 - dx) + c011 * dx
    c10 = c100 * (1 - dx) + c101 * dx
    c11 = c110 * (1 - dx) + c111 * dx

    c0 = c00 * (1 - dy) + c01 * dy
    c1 = c10 * (1 - dy) + c11 * dy

    result = c0 * (1 - dz) + c1 * dz
    return result


def rotate_vector(v, axis, angle_deg):
    """Rotate vector v around axis (x/y/z) by angle_deg"""
    angle = np.deg2rad(angle_deg)
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)
    c, s = np.cos(angle), np.sin(angle)
    R = (
        c * np.eye(3) +
        s * np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ]) +
        (1 - c) * np.outer(axis, axis)
    )
    return R @ v

def update_plane(position, normal):
    global volume
    if "MRI Volume" not in viewer.layers or volume is None:
        return

    viewer.layers["MRI Volume"].experimental_slicing_plane = {
        "position": position,
        "normal": normal,
        "enabled": True
    }

    try:
        
        # 1) 일단 한번 rot vector로 축을 바꾸고 시작 -> 안그러면 .. 축이 잘 안맞음 ㅠ 
        rotated_normal = rotate_vector(normal, axis=[0, 1, 0], angle_deg=90)

        # 2) slicing + 3D points
        plane_size = (int(SHAPE[1] *1.2), int(SHAPE[0] * 1.2))
        spacing = 1.0
        sliced, slice_pts_3d = slice_volume_with_plane(
            volume,
            plane_center=position,
            normal=rotated_normal,
            plane_x=None,
            plane_size=plane_size,
            spacing=spacing,
            in_plane_offset=(65,0),
            order=1
        )

        # points_xyz_napari = slice_pts_3d[:, [2,1,0]]
        points_xyz_napari = slice_pts_3d
      
        colors = sliced.flatten()

        if "Sliced Dots" in viewer.layers:
            viewer.layers["Sliced Dots"].data = points_xyz_napari
            viewer.layers["Sliced Dots"].features = {"intensity": colors}
            viewer.layers["Sliced Dots"].face_color = "intensity"
        else:
            viewer.add_points(
                points_xyz_napari,
                name="Sliced Dots",
                size=1,
                features={"intensity": colors},
                face_color="intensity",
                edge_width=0,
                opacity=0.6,
                blending="additive"
            )

        # #! 여기가 별도 2d vis를 키는 창임 동시에 키는 방법 강구해야함 
        # if hasattr(update_plane, "slice_window"):
        #     update_plane.slice_window.layers["Slice"].data = sliced
        # else:
        #     update_plane.slice_window = napari.Viewer()
        #     update_plane.slice_window.add_image(sliced, name="Slice", colormap="gray")
    except Exception as e:
        print(f"[Custom slice] Failed to compute sliced view: {e}")


@magicgui(call_button="Load DICOM Folder")
def dicom_loader_gui(folder_path: str = PATH):
    global volume
    if not folder_path:
        folder = QFileDialog.getExistingDirectory(QWidget(), "Select DICOM Folder")
        if not folder:
            print("No folder selected.")
            return
        folder_path = folder

    try:
        volume = load_dicom_series(folder_path)
        viewer.dims.ndisplay = 3
        viewer.add_image(volume, name="MRI Volume", rendering="attenuated_mip")
        center = [volume.shape[0] // 2, volume.shape[1] // 2, volume.shape[2] // 2]
        update_plane(center, [1, 0, 0], dtype=float)  # 초기 슬라이스
        print(f"Loaded volume from {folder_path}")
    except Exception as e:
        print(f"Error: {e}")


@magicgui(
    call_button="Update Slice Plane",
    position_x={"label": "X Pos"}, position_y={"label": "Y Pos"}, position_z={"label": "Z Pos"},
    normal_x={"label": "X Norm"}, normal_y={"label": "Y Norm"}, normal_z={"label": "Z Norm"},
)
def plane_controller(
    position_x: float = SHAPE[0] // 2,
    position_y: float = SHAPE[1] // 2,
    position_z: float = SHAPE[2] // 2,
    normal_x: float = 1.0,
    normal_y: float = 0.0,
    normal_z: float = 0.0,
):
    update_plane(
        [position_z, position_y, position_x],  # napari uses z, y, x order
        [normal_z, normal_y, normal_x]
    )
    
slice_count = 0  # 전역으로 선언

@magicgui(call_button="Select This Slice Plane")
def select_current_slice_plane():
    global slice_count, volume
    if volume is None or "MRI Volume" not in viewer.layers:
        print("No volume loaded.")
        return

    position = viewer.layers["MRI Volume"].experimental_slicing_plane.get("position", None)
    normal = viewer.layers["MRI Volume"].experimental_slicing_plane.get("normal", None)
    if position is None or normal is None:
        print("Slicing plane not set.")
        return

    try:
        rotated_normal = rotate_vector(normal, axis=[0, 1, 0], angle_deg=90)
        plane_size = (int(SHAPE[1] * 1.2), int(SHAPE[0] * 1.2))
        spacing = 1.0
        sliced, slice_pts_3d = slice_volume_with_plane(
            volume,
            plane_center=position,
            normal=rotated_normal,
            plane_x=None,
            plane_size=plane_size,
            spacing=spacing,
            in_plane_offset=(65, 0),
            order=1
        )

        points_xyz_napari = slice_pts_3d
        colors = sliced.flatten()

        layer_name = f"Sliced Dots {slice_count}"
        slice_count += 1

        viewer.add_points(
            points_xyz_napari,
            name=layer_name,
            size=1,
            features={"intensity": colors},
            face_color="intensity",
            edge_width=0,
            opacity=0.6,
            blending="additive"
        )
        print(f"Added new slice layer: {layer_name}")
    except Exception as e:
        print(f"[Select slice] Failed to extract: {e}")


# Main
if __name__ == "__main__":
    volume = None
    viewer = napari.Viewer()
    viewer.window.add_dock_widget(dicom_loader_gui, area="right")
    viewer.window.add_dock_widget(plane_controller, area="right")
    viewer.window.add_dock_widget(select_current_slice_plane, area="right")

    viewer.dims.ndisplay = 3
    napari.run()
