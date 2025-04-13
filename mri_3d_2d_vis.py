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

    # volume, position, normal, plane_size=(256, 256), spacing=1.0
    # ):
    # """
    # volume: 3D np.ndarray (Z, Y, X)
    # position: 3D point [z, y, x] on the plane
    # normal: normal vector [nz, ny, nx]
    # plane_size: output 2D plane size (height, width)
    # spacing: sampling resolution (in voxel units)
    # """

    # # 2. plane basis 계산
    # normal = np.array(normal, dtype=float)
    # normal /= np.linalg.norm(normal)

    # v = np.array([0, 1, 0], dtype=float) if np.allclose(normal, [1, 0, 0]) else np.array([1, 0, 0], dtype=float)
    # plane_x = np.cross(normal, v)
    # plane_x /= np.linalg.norm(plane_x)
    # plane_y = np.cross(normal, plane_x)
    # plane_y /= np.linalg.norm(plane_y)
    
    # # 6. Sliced image를 plane 위에 점으로 표시
    # # grid 좌표계 생성 (256x256 기준)
    # h, w = volume.shape[0],  volume.shape[1]
    # ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    # # 중심을 기준으로 offset (-128 ~ +128)
    # # grid_x = (xs - w // 2).astype(np.float32)
    # # grid_y = (ys - h // 2).astype(np.float32)
    
    # grid_x = (xs - w / 2 + 0.5).astype(np.float32)
    # grid_y = (ys - h / 2 + 0.5).astype(np.float32)      

    # # 3D 좌표로 투영
    # points = (
    #     position[0] + grid_y * plane_y[0] + grid_x * plane_x[0],
    #     position[1] + grid_y * plane_y[1] + grid_x * plane_x[1],
    #     position[2] + grid_y * plane_y[2] + grid_x * plane_x[2],
    # )

    # # Interpolate from volume
    # slice_img = map_coordinates(volume, points, order=1, mode='nearest')
    




# def slice_volume_with_plane(

#     # return slice_img
#         volume: np.ndarray,
#         plane_center: np.ndarray,
#         normal: np.ndarray,
#         plane_x: np.ndarray = None,
#         plane_size: tuple = (256, 256),
#         spacing: float = 1.0,
#         in_plane_offset: tuple = (0.0, 0.0)
#     ):
#     """
#     volume: 3D np.ndarray (Z, Y, X)
#     plane_center: 3D point [z, y, x], 평면의 중앙이 지날 위치
#     normal: plane normal (z, y, x)
#     plane_x: plane 상에서 x축이 될 벡터 (None이면 자동 결정)
#     plane_size: (width, height) in pixels
#     spacing: plane 상의 pixel spacing (확대/축소)
#     in_plane_offset: plane 상에서 (ox, oy)만큼 추가 이동 (픽셀 단위)
#     """

#     # 1) normal 정규화
#     normal = np.array(normal, dtype=float)
#     normal /= (np.linalg.norm(normal) + 1e-8)

#     # 2) plane_x가 없으면 default 지정
#     if plane_x is None:
#         # normal과 거의 평행하지 않은 벡터 하나 고름
#         if not np.allclose(normal, [1, 0, 0], atol=1e-6):
#             v = np.array([1, 0, 0], dtype=float)
#         else:
#             v = np.array([0, 1, 0], dtype=float)
#         plane_x = np.cross(normal, v)
#         plane_x /= (np.linalg.norm(plane_x) + 1e-8)
#     else:
#         plane_x = np.array(plane_x, dtype=float)
#         # 만약 plane_x가 normal과 기울어진 정도가 적으면 보정
#         plane_x -= np.dot(plane_x, normal) * normal
#         norm_px = np.linalg.norm(plane_x)
#         if norm_px < 1e-6:
#             raise ValueError("plane_x is nearly parallel to normal or zero-length!")
#         plane_x /= norm_px

#     # 3) plane_y = normal x plane_x
#     plane_y = np.cross(normal, plane_x)
#     plane_y /= (np.linalg.norm(plane_y) + 1e-8)

#     # 4) plane 내부에서 오프셋 적용
#     # plane_center + ox*plane_x + oy*plane_y
#     ox, oy = in_plane_offset
#     plane_center = plane_center + ox * plane_x + oy * plane_y

#     # 5) grid (pixel 좌표) 생성
#     w, h = plane_size
#     # 가로(width)·세로(height) 순서지만 아래 meshgrid는 (y, x) 형식으로 써줄 것
#     grid_x, grid_y = np.meshgrid(
#         np.linspace(-w/2, w/2, w) * spacing,
#         np.linspace(-h/2, h/2, h) * spacing,
#         indexing="xy"
#     )
#     # shape: (h, w)

#     # 6) 3D coords
#     # plane_x 방향에 grid_x, plane_y 방향에 grid_y 적용
#     coords = (
#         plane_center[0] + grid_y * plane_y[0] + grid_x * plane_x[0],
#         plane_center[1] + grid_y * plane_y[1] + grid_x * plane_x[1],
#         plane_center[2] + grid_y * plane_y[2] + grid_x * plane_x[2],
#     )

#     # 7) slice
#     slice_img = map_coordinates(volume, coords, order=1, mode='nearest')
#     return slice_img


def slice_volume_with_plane(volume, plane_center, normal, plane_x=None, 
                              plane_size=(256, 256), spacing=1.0, in_plane_offset=(0.0, 0.0), order=1):
    """
    1) 3D 볼륨에서 평면을 잘라 2D 슬라이스 생성
    2) 동시에, 그 슬라이스의 각 픽셀을 3D 좌표로 투영해 'Sliced Dots'형태로 시각화하기 위한 점들을 계산
       => 이 과정을 한 번에 하므로, 3D 평면과 2D 슬라이스가 정확히 동일 축을 공유함.

    Parameters
    ----------
    volume : (Z, Y, X)
    plane_center : (z, y, x)
        평면의 중심 좌표
    normal : (z, y, x)
        평면의 법선 벡터
    plane_x : (z, y, x), optional
        평면 상의 x축 방향. None이면 자동 결정.
    plane_size : (width, height)
        슬라이스 이미지 픽셀 크기
    spacing : float
        픽셀 사이 물리 간격
    in_plane_offset : (ox, oy)
        평면 내부에서 (plane_x, plane_y) 방향으로 추가 이동 (단위: 픽셀)
    order : int
        보간 차수(map_coordinates). 기본 1.

    Returns
    -------
    slice_img : 2D np.ndarray
        shape = (height, width)
    slice_points_3d : (N, 3) np.ndarray
        각 픽셀에 해당하는 3D 좌표 (z, y, x)
    """

    # 1) normal 정규화
    normal = np.array(normal, dtype=float)
    normal /= (np.linalg.norm(normal) + 1e-8)

    # 2) plane_x 결정 (없으면 자동)
    if plane_x is None:
        # normal과 거의 평행하지 않은 벡터 하나 선택
        # if not np.allclose(normal, [1, 0, 0], atol=1e-6):
        #     tmp = np.array([1, 0, 0], dtype=float)
        # else:
        #     tmp = np.array([0, 1, 0], dtype=float)
        
        v = np.array([0, 1, 0], dtype=float) if np.allclose(normal, [1, 0, 0]) else np.array([1, 0, 0], dtype=float)
        plane_x = np.cross(normal, v)
        plane_x /= (np.linalg.norm(plane_x) + 1e-8)
    else:
        plane_x = np.array(plane_x, dtype=float)
        # plane_x를 normal에 정사영된 성분 제거 → 완전히 평면 위로
        plane_x -= np.dot(plane_x, normal) * normal
        norm_px = np.linalg.norm(plane_x)
        if norm_px < 1e-6:
            raise ValueError("plane_x is nearly parallel to normal or zero-length!")
        plane_x /= norm_px

    # 3) plane_y = normal x plane_x
    plane_y = np.cross(normal, plane_x)
    plane_y /= (np.linalg.norm(plane_y) + 1e-8)

    # 4) plane_center + in_plane_offset
    pc = np.array(plane_center, dtype=float).copy()
    ox, oy = in_plane_offset
    pc = pc + ox * plane_x + oy * plane_y  # 평면 내부 offset

    # 5) 그리드 생성: (width, height)
    w, h = plane_size
    xs = np.arange(w, dtype=float) - (w - 1)/2.0  # -127.5 ~ +128.5 (예시)
    ys = np.arange(h, dtype=float) - (h - 1)/2.0
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")

    # 물리적 spacing 적용
    grid_x *= spacing
    grid_y *= spacing

    # 6) 3D coords (z, y, x)
    coords_z = pc[0] + grid_y * plane_y[0] + grid_x * plane_x[0]
    coords_y = pc[1] + grid_y * plane_y[1] + grid_x * plane_x[1]
    coords_x = pc[2] + grid_y * plane_y[2] + grid_x * plane_x[2]

    coords = np.array([coords_z, coords_y, coords_x], dtype=float)  # shape=(3, h, w)

    # 7) 슬라이스 추출
    slice_img = my_map_coordinates(volume, coords, order=order, mode='nearest')
    # shape=(h, w)

    # 8) 3D 점 좌표. shape=(h, w, 3)
    slice_points_3d = np.stack([coords_z, coords_y, coords_x], axis=-1)

    # (h,w,3) -> (h*w, 3)
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
        
        # 1) plane 구하기 (기본 normal 또는 rotate_vector로 기울이기)
        rotated_normal = rotate_vector(normal, axis=[0, 1, 0], angle_deg=90)

        # 2) slicing + 3D points
        plane_size = (256, 256)
        spacing = 1.0
        sliced, slice_pts_3d = slice_volume_with_plane(
            volume,
            plane_center=position,
            normal=rotated_normal,
            plane_x=None,
            plane_size=plane_size,
            spacing=spacing,
            in_plane_offset=(0,0),
            order=1
        )
        # (선택) 2D 슬라이스 추가 회전 => 만약 적용하면, 3D 좌표계와는 또 달라짐에 유의
        # sliced = np.rot90(sliced, k=3)

        # 2) 3D 점 찍기 (동일한 w,h, px, py, plane_center 사용)
        h, w = sliced.shape
        ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        # 중심을 (h-1)/2, (w-1)/2로 맞춤
        grid_x = (xs - (w - 1)/2).astype(np.float32)
        grid_y = (ys - (h - 1)/2).astype(np.float32)
        grid_x *= spacing
        grid_y *= spacing

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

        # # 예: Z축 기준으로 45도 회전된 normal
        # rotated_normal = rotate_vector(normal, axis=[0, 1, 0], angle_deg=90)
        # # # 1. sliced 생성
        # # sliced = slice_volume_with_plane(volume, position, rotated_normal, plane_size=(256, 256), spacing=1.0)
        
        # spacing = 1.3
        # plane_size = (SHAPE[1], SHAPE[0])
        # in_plane_offset = (0, 0)  # plane 내부에서 x방향 +50, y방향 -50 픽셀 이동

        # sliced = slice_volume_with_plane(
        #     volume, position, rotated_normal,
        #     plane_size=plane_size,
        #     spacing=spacing,
        #     in_plane_offset=in_plane_offset
        # )
                
        
        
        # sliced = np.rot90(sliced, k=3)  # 필요에 따라 k=1 또는 k=3을 사용
    


        # # 2. plane basis 계산
        # normal = np.array(normal, dtype=float)
        # normal /= np.linalg.norm(normal)

        # v = np.array([0, 1, 0], dtype=float) if np.allclose(normal, [1, 0, 0]) else np.array([1, 0, 0], dtype=float)
        # plane_x = np.cross(normal, v)
        # plane_x /= np.linalg.norm(plane_x)
        # plane_y = np.cross(normal, plane_x)
        # plane_y /= np.linalg.norm(plane_y)
        
        # # 6. Sliced image를 plane 위에 점으로 표시
        # # grid 좌표계 생성 (256x256 기준)
        # h, w = sliced.shape
        # ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

        # # 중심을 기준으로 offset (-128 ~ +128)
        # # grid_x = (xs - w // 2).astype(np.float32)
        # # grid_y = (ys - h // 2).astype(np.float32)
        
        # grid_x = (xs - w / 2 + 0.5).astype(np.float32)
        # grid_y = (ys - h / 2 + 0.5).astype(np.float32)      

        # # 3D 좌표로 투영
        # plane_points = (
        #     position[0] + grid_y * plane_y[0] + grid_x * plane_x[0],
        #     position[1] + grid_y * plane_y[1] + grid_x * plane_x[1],
        #     position[2] + grid_y * plane_y[2] + grid_x * plane_x[2],
        # )

        # points_xyz = np.stack(plane_points, axis=-1).reshape(-1, 3)
        # points_xyz_napari = points_xyz[:, [2, 1, 0]]  # z, y, x → x, y, z

        # # intensity를 colormap scalar로 설정
        # colors = sliced.flatten()

        # if "Sliced Dots" in viewer.layers:
        #     viewer.layers["Sliced Dots"].data = points_xyz_napari
        #     viewer.layers["Sliced Dots"].features = {"intensity": colors}
        #     viewer.layers["Sliced Dots"].face_color = "intensity"
        # else:
        #     viewer.add_points(
        #         points_xyz_napari,
        #         name="Sliced Dots",
        #         size=1,
        #         features={"intensity": colors},
        #         face_color="intensity",
        #         edge_width=0,
        #         opacity=0.6,
        #         blending="additive"
        #     )

        # # 5. sliced 이미지는 별도 2D window에만 시각화
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


# Main
if __name__ == "__main__":
    volume = None
    viewer = napari.Viewer()
    viewer.window.add_dock_widget(dicom_loader_gui, area="right")
    viewer.window.add_dock_widget(plane_controller, area="right")
    viewer.dims.ndisplay = 3
    napari.run()
