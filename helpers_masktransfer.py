import numpy as np
import cv2
import random
from scipy.spatial import Delaunay

# --- Explicit Imports ---
from helpers_cal import (
    qvec2rotmat,
    get_depth_from_map,
    back_project_point,
    project_point
)

# --- 1. Shape Classification and Fitting ---


def classify_component_shape(component_name):
    """Classifies a component as 'planar' or 'curved' based on a predefined dictionary."""
    shape_map = {
        "windshield_f": "planar",
        "windshield_b": "planar",
        "door_bl/window": "planar",
        "door_br/window": "planar",
        "door_fl/window": "planar",
        "door_fr/window": "planar",
        "rockerpanel": "planar",
        "step_pad": "planar",
        "glass_roof": "planar",
        "glass_r": "planar",
        "license_plate": "planar",
        "Front windshield": "planar",
        "bonnet": "curved",
        "bootlid": "curved",
        "bumper": "curved",
        "door_bl_assy": "curved",
        "door_br_assy": "curved",
        "door_fl_assy": "curved",
        "door_fr_assy": "curved",
        "grille": "curved",
        "headlamp": "curved",
        "mirror": "curved",
        "roof": "curved",
        "taillamp": "curved",
        "wheel": "curved",
        "wing": "curved",
        "fender": "curved",
        "moulding": "curved",
        "handle": "curved",
        "emblem": "curved",
        "cover": "curved",
        "fin": "curved",
        "stoplamp": "curved",
        "aux_grille": "curved",
        "exhaust_trim": "curved",
        "reflector": "curved",
        "fuel_filler_lid": "curved",
        "distance_sensor": "curved",
        "Bonnet": "curved",
        "Front bumper": "curved",
        "Left headlamp": "curved",
        "Left mirror": "curved",
        "Grille": "curved",
        "Front left door": "curved",
    }
    for keyword, shape in shape_map.items():
        if keyword in component_name:
            return shape
    return "curved"


def fit_plane_ransac(point_cloud, iterations=100, distance_threshold=0.02):
    """Fits a plane using RANSAC and returns the inlier points."""
    best_inliers = None
    best_inlier_count = 0
    for i in range(iterations):
        indices = np.random.choice(point_cloud.shape[0], 3, replace=False)
        p1, p2, p3 = point_cloud[indices]
        normal = np.cross(p2 - p1, p3 - p1)
        normal_norm = np.linalg.norm(normal)
        if normal_norm == 0:
            continue
        normal /= normal_norm
        d = -np.dot(normal, p1)
        distances = np.abs(np.dot(point_cloud, normal) + d)
        inliers = point_cloud[distances < distance_threshold]
        if len(inliers) > best_inlier_count:
            best_inlier_count = len(inliers)
            best_inliers = inliers
    return best_inliers


### Baseline transfer_mask


def transfer_mask(
    source_id, target_id, source_component_mask, images, cameras, image_id_to_rgb_path, image_id_to_depth_path
):
    """
    A simpler transfer that projects all pixels from a source mask and reconstructs it.
    """
    target_rgb_for_size = cv2.imread(image_id_to_rgb_path[target_id])
    if target_rgb_for_size is None:
        return None
    target_h, target_w, _ = target_rgb_for_size.shape

    source_rgb_for_size = cv2.imread(image_id_to_rgb_path[source_id])
    if source_rgb_for_size is None:
        return None
    source_h, source_w, _ = source_rgb_for_size.shape

    source_depth_map = cv2.imread(image_id_to_depth_path[source_id], cv2.IMREAD_UNCHANGED)
    if source_depth_map is None:
        return None
    if source_depth_map.shape != (source_h, source_w):
        source_depth_map = cv2.resize(source_depth_map, (source_w, source_h), interpolation=cv2.INTER_NEAREST)

    source_pixels_y, source_pixels_x = np.where(source_component_mask > 0)
    source_pixels = np.vstack((source_pixels_x, source_pixels_y)).T

    projected_pixels = []
    source_image_model = images[source_id]
    source_camera_model = cameras[source_image_model.camera_id]
    target_image_model = images[target_id]
    target_camera_model = cameras[target_image_model.camera_id]

    for u, v in source_pixels:
        depth, _ = get_depth_from_map(u, v, source_depth_map)
        if depth:
            p_3d = back_project_point(u, v, depth, source_camera_model, source_image_model)
            if p_3d is not None:
                p_2d_target = project_point(p_3d, target_camera_model, target_image_model)
                if p_2d_target:
                    projected_pixels.append(p_2d_target)

    if not projected_pixels:
        return None

    transferred_mask = np.zeros((target_h, target_w), dtype=np.uint8)
    valid_points = np.round(projected_pixels).astype(int)
    valid_indices = (
        (valid_points[:, 0] >= 0)
        & (valid_points[:, 0] < target_w)
        & (valid_points[:, 1] >= 0)
        & (valid_points[:, 1] < target_h)
    )
    valid_points = valid_points[valid_indices]
    transferred_mask[valid_points[:, 1], valid_points[:, 0]] = 255

    return transferred_mask


def transfer_mask_and_point(
    source_id, target_id, source_component_mask, images, cameras, image_id_to_rgb_path, image_id_to_depth_path
):
    """
    Transfers a full mask and a single sparse point from within that mask.
    """
    target_rgb_for_size = cv2.imread(image_id_to_rgb_path[target_id])
    target_h, target_w, _ = target_rgb_for_size.shape
    source_rgb_for_size = cv2.imread(image_id_to_rgb_path[source_id])
    source_h, source_w, _ = source_rgb_for_size.shape
    source_depth_map = cv2.imread(image_id_to_depth_path[source_id], cv2.IMREAD_UNCHANGED)
    if source_depth_map.shape != (source_h, source_w):
        source_depth_map = cv2.resize(source_depth_map, (source_w, source_h), interpolation=cv2.INTER_NEAREST)

    source_image_model = images[source_id]
    source_camera_model = cameras[source_image_model.camera_id]
    target_image_model = images[target_id]
    target_camera_model = cameras[target_image_model.camera_id]

    # Part 1: Select and project a sparse point from within the mask
    sparse_points_in_component = []
    for idx, point in enumerate(source_image_model.xys):
        if source_image_model.point3D_ids[idx] != -1:
            u, v = int(point[0]), int(point[1])
            if 0 <= v < source_h and 0 <= u < source_w and source_component_mask[v, u] > 0:
                sparse_points_in_component.append({'uv': (u, v), 'original_index': idx})

    projected_sparse_point = None
    source_point_uv = None
    if sparse_points_in_component:
        keypoint_data = random.choice(sparse_points_in_component)
        source_point_uv = keypoint_data['uv']
        depth, _ = get_depth_from_map(source_point_uv[0], source_point_uv[1], source_depth_map)
        if depth:
            p_3d = back_project_point(
                source_point_uv[0], source_point_uv[1], depth, source_camera_model, source_image_model
            )
            if p_3d is not None:
                projected_sparse_point = project_point(p_3d, target_camera_model, target_image_model)

    # Part 2: Transfer the full mask (using the simpler `transfer_mask` logic)
    transferred_mask = transfer_mask(
        source_id, target_id, source_component_mask, images, cameras, image_id_to_rgb_path, image_id_to_depth_path
    )

    return transferred_mask, (source_point_uv, projected_sparse_point)


# --- 2. Mask Transfer Pipelines (V1-V4) ---


def transfer_mask_v1(
    source_id, target_id, source_component_mask, images, cameras, image_id_to_rgb_path, image_id_to_depth_path
):
    """V1 ("Robust"): Projects all points then cleans up with morphological closing."""
    target_rgb_for_size = cv2.imread(image_id_to_rgb_path[target_id])
    target_h, target_w, _ = target_rgb_for_size.shape
    source_rgb_for_size = cv2.imread(image_id_to_rgb_path[source_id])
    source_h, source_w, _ = source_rgb_for_size.shape
    source_depth_map_raw = cv2.imread(image_id_to_depth_path[source_id], cv2.IMREAD_UNCHANGED)
    inpaint_mask = np.where(source_depth_map_raw == 0, 255, 0).astype(np.uint8)
    source_depth_map = cv2.inpaint(source_depth_map_raw, inpaint_mask, 3, cv2.INPAINT_NS)
    source_depth_map = cv2.resize(source_depth_map, (source_w, source_h), interpolation=cv2.INTER_NEAREST)
    source_pixels_y, source_pixels_x = np.where(source_component_mask > 0)
    source_pixels = np.vstack((source_pixels_x, source_pixels_y)).T
    transferred_mask_raw = np.zeros((target_h, target_w), dtype=np.uint8)
    depth_buffer = np.full((target_h, target_w), np.inf, dtype=np.float32)
    source_image_model, source_camera_model = images[source_id], cameras[images[source_id].camera_id]
    target_image_model, target_camera_model = images[target_id], cameras[images[target_id].camera_id]
    R_target, t_target = qvec2rotmat(target_image_model.qvec), target_image_model.tvec
    for u, v in source_pixels:
        depth, _ = get_depth_from_map(u, v, source_depth_map)
        if depth:
            p_3d = back_project_point(u, v, depth, source_camera_model, source_image_model)
            if p_3d is not None:
                p_cam_target = R_target @ p_3d + t_target
                target_z = p_cam_target[2]
                if target_z > 0:
                    p_2d_target = project_point(p_3d, target_camera_model, target_image_model)
                    if p_2d_target:
                        u_t, v_t = int(round(p_2d_target[0])), int(round(p_2d_target[1]))
                        if 0 <= v_t < target_h and 0 <= u_t < target_w and target_z < depth_buffer[v_t, u_t]:
                            transferred_mask_raw[v_t, u_t] = 255
                            depth_buffer[v_t, u_t] = target_z
    kernel = np.ones((5, 5), np.uint8)
    closed_mask = cv2.morphologyEx(transferred_mask_raw, cv2.MORPH_CLOSE, kernel, iterations=2)
    return closed_mask


def transfer_mask_v2(
    source_id,
    target_id,
    source_component_mask,
    component_name,
    images,
    cameras,
    image_id_to_rgb_path,
    image_id_to_depth_path,
):
    """V2 ("Advanced"): Fits a geometric model (RANSAC/Mesh) then projects."""
    target_rgb_for_size = cv2.imread(image_id_to_rgb_path[target_id])
    target_h, target_w, _ = target_rgb_for_size.shape
    source_rgb_for_size = cv2.imread(image_id_to_rgb_path[source_id])
    source_h, source_w, _ = source_rgb_for_size.shape
    source_depth_map_raw = cv2.imread(image_id_to_depth_path[source_id], cv2.IMREAD_UNCHANGED)
    inpaint_mask = np.where(source_depth_map_raw == 0, 255, 0).astype(np.uint8)
    source_depth_map = cv2.inpaint(source_depth_map_raw, inpaint_mask, 3, cv2.INPAINT_NS)
    source_depth_map = cv2.resize(source_depth_map, (source_w, source_h), interpolation=cv2.INTER_NEAREST)
    source_pixels_y, source_pixels_x = np.where(source_component_mask > 0)
    source_pixels = np.vstack((source_pixels_x, source_pixels_y)).T
    source_image_model, source_camera_model = images[source_id], cameras[images[source_id].camera_id]
    point_cloud_3d = []
    for u, v in source_pixels:
        depth, _ = get_depth_from_map(u, v, source_depth_map)
        if depth:
            p_3d = back_project_point(u, v, depth, source_camera_model, source_image_model)
            if p_3d is not None:
                point_cloud_3d.append(p_3d)
    if not point_cloud_3d:
        return None
    point_cloud_3d = np.array(point_cloud_3d)
    shape_type = classify_component_shape(component_name)
    points_to_project = None
    if shape_type == 'planar':
        points_to_project = fit_plane_ransac(point_cloud_3d)
    else:
        if len(point_cloud_3d) < 4:
            return None
        tri = Delaunay(point_cloud_3d)
        points_to_project = point_cloud_3d[np.unique(tri.simplices)]
    if points_to_project is None or len(points_to_project) == 0:
        return None
    transferred_mask_raw = np.zeros((target_h, target_w), dtype=np.uint8)
    depth_buffer = np.full((target_h, target_w), np.inf, dtype=np.float32)
    target_image_model, target_camera_model = images[target_id], cameras[images[target_id].camera_id]
    R_target, t_target = qvec2rotmat(target_image_model.qvec), target_image_model.tvec
    for p_3d in points_to_project:
        p_cam_target = R_target @ p_3d + t_target
        target_z = p_cam_target[2]
        if target_z > 0:
            p_2d_target = project_point(p_3d, target_camera_model, target_image_model)
            if p_2d_target:
                u_t, v_t = int(round(p_2d_target[0])), int(round(p_2d_target[1]))
                if 0 <= v_t < target_h and 0 <= u_t < target_w and target_z < depth_buffer[v_t, u_t]:
                    transferred_mask_raw[v_t, u_t] = 255
                    depth_buffer[v_t, u_t] = target_z
    kernel = np.ones((7, 7), np.uint8)
    closed_mask = cv2.morphologyEx(transferred_mask_raw, cv2.MORPH_CLOSE, kernel, iterations=3)
    return closed_mask


def transfer_mask_v3(
    source_id,
    target_id,
    source_component_mask,
    component_name,
    images,
    cameras,
    image_id_to_rgb_path,
    image_id_to_depth_path,
    image_id_to_mask_path,
):
    """V3 ("Ultimate"): V2 + Full visibility check and clipping."""
    target_rgb_for_size = cv2.imread(image_id_to_rgb_path[target_id])
    target_h, target_w, _ = target_rgb_for_size.shape
    source_rgb_for_size = cv2.imread(image_id_to_rgb_path[source_id])
    source_h, source_w, _ = source_rgb_for_size.shape
    source_depth_map_raw = cv2.imread(image_id_to_depth_path[source_id], cv2.IMREAD_UNCHANGED)
    inpaint_mask = np.where(source_depth_map_raw == 0, 255, 0).astype(np.uint8)
    source_depth_map = cv2.inpaint(source_depth_map_raw, inpaint_mask, 3, cv2.INPAINT_NS)
    source_depth_map = cv2.resize(source_depth_map, (source_w, source_h), interpolation=cv2.INTER_NEAREST)
    source_pixels_y, source_pixels_x = np.where(source_component_mask > 0)
    source_pixels = np.vstack((source_pixels_x, source_pixels_y)).T
    source_image_model, source_camera_model = images[source_id], cameras[images[source_id].camera_id]
    point_cloud_3d = []
    for u, v in source_pixels:
        depth, _ = get_depth_from_map(u, v, source_depth_map)
        if depth:
            p_3d = back_project_point(u, v, depth, source_camera_model, source_image_model)
            if p_3d is not None:
                point_cloud_3d.append(p_3d)
    if not point_cloud_3d:
        return None
    point_cloud_3d = np.array(point_cloud_3d)
    shape_type = classify_component_shape(component_name)
    points_to_project = None
    if shape_type == 'planar':
        points_to_project = fit_plane_ransac(point_cloud_3d)
    else:
        if len(point_cloud_3d) < 4:
            return None
        tri = Delaunay(point_cloud_3d)
        points_to_project = point_cloud_3d[np.unique(tri.simplices)]
    if points_to_project is None or len(points_to_project) == 0:
        return None
    target_depth_map_raw = cv2.imread(image_id_to_depth_path[target_id], cv2.IMREAD_UNCHANGED)
    if target_depth_map_raw is None:
        return None
    target_depth_map = cv2.resize(target_depth_map_raw, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    visibility_checked_mask = np.zeros((target_h, target_w), dtype=np.uint8)
    target_image_model, target_camera_model = images[target_id], cameras[images[target_id].camera_id]
    R_target, t_target = qvec2rotmat(target_image_model.qvec), target_image_model.tvec
    depth_tolerance = 0.05
    for p_3d in points_to_project:
        p_cam_target = R_target @ p_3d + t_target
        z_projected = p_cam_target[2]
        if z_projected > 0:
            p_2d_target = project_point(p_3d, target_camera_model, target_image_model)
            if p_2d_target:
                u_t, v_t = int(round(p_2d_target[0])), int(round(p_2d_target[1]))
                if 0 <= v_t < target_h and 0 <= u_t < target_w:
                    z_ground_truth, _ = get_depth_from_map(u_t, v_t, target_depth_map)
                    if z_ground_truth and z_projected <= (z_ground_truth + depth_tolerance):
                        visibility_checked_mask[v_t, u_t] = 255
    kernel = np.ones((7, 7), np.uint8)
    closed_mask = cv2.morphologyEx(visibility_checked_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    target_vehicle_mask = cv2.imread(image_id_to_mask_path.get(target_id, ''), cv2.IMREAD_GRAYSCALE)
    if target_vehicle_mask is not None:
        if target_vehicle_mask.shape != (target_h, target_w):
            target_vehicle_mask = cv2.resize(target_vehicle_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        return cv2.bitwise_and(closed_mask, target_vehicle_mask)
    else:
        return closed_mask


def transfer_mask_v3_fast(
    source_id,
    target_id,
    source_component_mask,
    component_name,
    images,
    cameras,
    image_id_to_rgb_path,
    image_id_to_depth_path,
    image_id_to_mask_path,
    subsample_step=9,
):
    """Optimized version of V3 that performs 2D subsampling."""
    target_rgb_for_size = cv2.imread(image_id_to_rgb_path[target_id])
    target_h, target_w, _ = target_rgb_for_size.shape
    source_rgb_for_size = cv2.imread(image_id_to_rgb_path[source_id])
    source_h, source_w, _ = source_rgb_for_size.shape
    source_depth_map_raw = cv2.imread(image_id_to_depth_path[source_id], cv2.IMREAD_UNCHANGED)
    inpaint_mask = np.where(source_depth_map_raw == 0, 255, 0).astype(np.uint8)
    source_depth_map = cv2.inpaint(source_depth_map_raw, inpaint_mask, 3, cv2.INPAINT_NS)
    source_depth_map = cv2.resize(source_depth_map, (source_w, source_h), interpolation=cv2.INTER_NEAREST)
    subsampled_mask = source_component_mask[::subsample_step, ::subsample_step]
    source_pixels_y_sub, source_pixels_x_sub = np.where(subsampled_mask > 0)
    source_pixels_y = source_pixels_y_sub * subsample_step
    source_pixels_x = source_pixels_x_sub * subsample_step
    source_pixels = np.vstack((source_pixels_x, source_pixels_y)).T
    source_image_model, source_camera_model = images[source_id], cameras[images[source_id].camera_id]
    point_cloud_3d = []
    for u, v in source_pixels:
        depth, _ = get_depth_from_map(u, v, source_depth_map)
        if depth:
            p_3d = back_project_point(u, v, depth, source_camera_model, source_image_model)
            if p_3d is not None:
                point_cloud_3d.append(p_3d)
    if not point_cloud_3d:
        return None
    point_cloud_3d = np.array(point_cloud_3d)
    shape_type = classify_component_shape(component_name)
    points_to_project = None
    if shape_type == 'planar':
        points_to_project = fit_plane_ransac(point_cloud_3d)
    else:
        if len(point_cloud_3d) < 4:
            return None
        tri = Delaunay(point_cloud_3d)
        points_to_project = point_cloud_3d[np.unique(tri.simplices)]
    if points_to_project is None or len(points_to_project) == 0:
        return None
    target_depth_map_raw = cv2.imread(image_id_to_depth_path[target_id], cv2.IMREAD_UNCHANGED)
    if target_depth_map_raw is None:
        return None
    target_depth_map = cv2.resize(target_depth_map_raw, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    visibility_checked_mask = np.zeros((target_h, target_w), dtype=np.uint8)
    target_image_model, target_camera_model = images[target_id], cameras[images[target_id].camera_id]
    R_target, t_target = qvec2rotmat(target_image_model.qvec), target_image_model.tvec
    depth_tolerance = 0.05
    for p_3d in points_to_project:
        p_cam_target = R_target @ p_3d + t_target
        z_projected = p_cam_target[2]
        if z_projected > 0:
            p_2d_target = project_point(p_3d, target_camera_model, target_image_model)
            if p_2d_target:
                u_t, v_t = int(round(p_2d_target[0])), int(round(p_2d_target[1]))
                if 0 <= v_t < target_h and 0 <= u_t < target_w:
                    z_ground_truth, _ = get_depth_from_map(u_t, v_t, target_depth_map)
                    if z_ground_truth and z_projected <= (z_ground_truth + depth_tolerance):
                        visibility_checked_mask[v_t, u_t] = 255
    kernel = np.ones((7, 7), np.uint8)
    closed_mask = cv2.morphologyEx(visibility_checked_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    target_vehicle_mask = cv2.imread(image_id_to_mask_path.get(target_id, ''), cv2.IMREAD_GRAYSCALE)
    if target_vehicle_mask is not None:
        if target_vehicle_mask.shape != (target_h, target_w):
            target_vehicle_mask = cv2.resize(target_vehicle_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        return cv2.bitwise_and(closed_mask, target_vehicle_mask)
    else:
        return closed_mask


def transfer_mask_v4(
    source_id,
    target_id,
    source_component_mask,
    component_name,
    images,
    cameras,
    image_id_to_rgb_path,
    image_id_to_depth_path,
    image_id_to_mask_path,
    erosion_kernel_size=7,
):
    """V4 ("Edge-Removed"): Erodes the source mask before calling V3."""
    kernel = np.ones((erosion_kernel_size, erosion_kernel_size), np.uint8)
    eroded_mask = cv2.erode(source_component_mask, kernel, iterations=1)
    return transfer_mask_v3(
        source_id,
        target_id,
        eroded_mask,
        component_name,
        images,
        cameras,
        image_id_to_rgb_path,
        image_id_to_depth_path,
        image_id_to_mask_path,
    )
