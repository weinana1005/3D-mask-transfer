import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pycocotools import mask as mask_utils
from scipy.spatial.distance import cdist


def qvec2rotmat(qvec):
    """Converts a quaternion vector to a 3x3 rotation matrix."""
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[1] * qvec[3] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[1] * qvec[3] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def get_depth_from_map(u, v, depth_map, scale_factor=1000.0):
    """Reads a depth value from a depth map at (u, v) and scales it to meters."""
    if depth_map is None:
        return None, "Depth map not loaded"
    h, w = depth_map.shape
    if not (0 <= v < h and 0 <= u < w):
        return None, "Coordinates out of bounds"
    pixel_value = depth_map[int(v), int(u)]
    if pixel_value == 0:
        return None, "Depth value is zero"
    return float(pixel_value) / scale_factor, None


def back_project_point(u, v, depth_m, camera, image):
    """Back-projects a 2D point with depth to a 3D world coordinate."""
    R = qvec2rotmat(image.qvec)
    t = image.tvec
    fx, fy, cx, cy = camera.params[0], camera.params[1], camera.params[2], camera.params[3]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    K_inv = np.linalg.inv(K)
    p_2d_homogeneous = np.array([u, v, 1.0])
    p_cam = depth_m * (K_inv @ p_2d_homogeneous)
    p_world = R.T @ (p_cam - t)
    return p_world


def project_point(point_3d, camera, image):
    """Projects a 3D world point onto a 2D image plane."""
    R = qvec2rotmat(image.qvec)
    t = image.tvec
    fx, fy, cx, cy = camera.params[0], camera.params[1], camera.params[2], camera.params[3]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    point_cam = R @ point_3d + t
    if point_cam[2] <= 1e-6:
        return None
    point_proj = K @ point_cam
    return (point_proj[0] / point_proj[2], point_proj[1] / point_proj[2])


# --- Calculate projection error ---


def analyze_reprojection_error(
    source_id,
    target_id,
    # Pass in all required data
    images,
    cameras,
    points3D,
    image_id_to_mask_path,
    image_id_to_rgb_path,
    image_id_to_depth_path,
):
    """
    Analyzes all projectable points and returns aggregated data,
    including the point with the maximum error.
    """
    print(f"Starting analysis between Source ID {source_id} and Target ID {target_id}...")
    source_mask = cv2.imread(image_id_to_mask_path[source_id], cv2.IMREAD_GRAYSCALE)
    source_rgb_for_size = cv2.imread(image_id_to_rgb_path[source_id])
    if source_rgb_for_size is None:
        print("Error: Could not load source RGB.")
        return None
    rgb_h, rgb_w, _ = source_rgb_for_size.shape
    source_depth_map = cv2.imread(image_id_to_depth_path[source_id], cv2.IMREAD_UNCHANGED)
    if source_depth_map is None:
        print("Error: Could not load source depth map.")
        return None
    if source_depth_map.shape[0] != rgb_h or source_depth_map.shape[1] != rgb_w:
        source_depth_map = cv2.resize(source_depth_map, (rgb_w, rgb_h), interpolation=cv2.INTER_NEAREST)

    keypoints_for_projection = []
    for idx, point in enumerate(images[source_id].xys):
        if images[source_id].point3D_ids[idx] != -1:
            u, v = int(point[0]), int(point[1])
            h, w = source_mask.shape
            if 0 <= v < h and 0 <= u < w and source_mask[v, u] > 128:
                keypoints_for_projection.append({'uv': (u, v), 'original_index': idx})

    if not keypoints_for_projection:
        print("Found no projectable keypoints on the source mask.")
        return None
    print(f"Found {len(keypoints_for_projection)} projectable keypoints. Processing...")

    results_data = []
    source_image_model = images[source_id]
    source_camera_model = cameras[source_image_model.camera_id]
    target_image_model = images[target_id]
    target_camera_model = cameras[target_image_model.camera_id]

    for keypoint_data in keypoints_for_projection:
        u_source, v_source = keypoint_data['uv']
        point_index = keypoint_data['original_index']

        point_3d_id = images[source_id].point3D_ids[point_index]
        point_3d_xyz_sparse = points3D[point_3d_id].xyz
        projected_point_sparse = project_point(point_3d_xyz_sparse, target_camera_model, target_image_model)

        depth_in_meters, _ = get_depth_from_map(u_source, v_source, source_depth_map)

        if projected_point_sparse and depth_in_meters:
            point_3d_xyz_dense = back_project_point(
                u_source, v_source, depth_in_meters, source_camera_model, source_image_model
            )
            if point_3d_xyz_dense is not None:
                projected_point_dense = project_point(point_3d_xyz_dense, target_camera_model, target_image_model)
                if projected_point_dense:
                    p_sparse = np.array(projected_point_sparse)
                    p_dense = np.array(projected_point_dense)
                    distance = np.linalg.norm(p_sparse - p_dense)
                    results_data.append(
                        {
                            "error": distance,
                            "depth": depth_in_meters,
                            "source_uv": (u_source, v_source),
                            "target_uv_sparse": projected_point_sparse,
                            "target_uv_dense": projected_point_dense,
                        }
                    )

    if not results_data:
        print("Could not successfully project any points with both methods.")
        return None

    errors = [r['error'] for r in results_data]
    max_error_index = np.argmax(errors)
    worst_offender_data = results_data[max_error_index]

    print("\n--- Reprojection Error Statistics ---")
    print(f"Points Analyzed: {len(errors)}")
    print(f"Mean Error:      {np.mean(errors):.2f} pixels")
    print(f"Median Error:    {np.median(errors):.2f} pixels")
    print(f"Max Error:       {np.max(errors):.2f} pixels")
    print("-------------------------------------\n")

    return {
        "errors": errors,
        "coordinates": np.array([r['target_uv_dense'] for r in results_data]),
        "depths": [r['depth'] for r in results_data],
        "worst_offender": worst_offender_data,
    }


def plot_worst_offender(
    source_id,
    target_id,
    worst_offender_data,
    # Pass in required data
    images,
    image_id_to_rgb_path,
):
    """Creates a dedicated plot showing the single point with the largest error."""
    print("--- Visualizing Point with Maximum Error ---")
    source_rgb = cv2.cvtColor(cv2.imread(image_id_to_rgb_path[source_id]), cv2.COLOR_BGR2RGB)
    target_rgb = cv2.cvtColor(cv2.imread(image_id_to_rgb_path[target_id]), cv2.COLOR_BGR2RGB)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle(f'Maximum Error Analysis: {worst_offender_data["error"]:.2f} pixels', fontsize=20, y=1.02)

    u_source, v_source = worst_offender_data["source_uv"]
    ax1.imshow(source_rgb)
    ax1.scatter(u_source, v_source, s=250, c='red', marker='X')
    ax1.set_title(f'Source Point on: {images[source_id].name}', fontsize=16)
    ax1.axis('off')

    u_s, v_s = worst_offender_data["target_uv_sparse"]
    u_d, v_d = worst_offender_data["target_uv_dense"]

    ax2.imshow(target_rgb)
    ax2.scatter(u_s, v_s, s=150, c='magenta', marker='X', label='Sparse (COLMAP)')
    ax2.scatter(u_d, v_d, s=200, c='lime', marker='+', label='Dense (Ground Truth)')
    ax2.plot([u_s, u_d], [v_s, v_d], 'r-', lw=2, alpha=0.8)
    ax2.set_title(f'Error Visualization on: {images[target_id].name}', fontsize=16)
    ax2.legend()
    ax2.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# --- Data Loading and Handling ---


def load_component_data(json_path):
    """Loads COCO-style data and uses a NORMALIZED basename for matching."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    image_name_to_anns = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        img_file_name = next((img['file_name'] for img in data['images'] if img['id'] == img_id), None)
        if img_file_name:
            normalized_name = os.path.splitext(os.path.basename(img_file_name))[0]
            if normalized_name not in image_name_to_anns:
                image_name_to_anns[normalized_name] = []
            image_name_to_anns[normalized_name].append(ann)
    cat_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
    return image_name_to_anns, cat_id_to_name


def get_component_mask(image_name, component_name, image_name_to_anns, cat_id_to_name):
    """Retrieves and decodes the mask for a specific component in a specific image."""
    normalized_name = os.path.splitext(os.path.basename(image_name))[0]
    if normalized_name not in image_name_to_anns:
        return None, "Image name not found in annotations."

    cat_id = next((cid for cid, name in cat_id_to_name.items() if name == component_name), None)
    if cat_id is None:
        return None, f"Component name '{component_name}' not found."

    for ann in image_name_to_anns[normalized_name]:
        if ann['category_id'] == cat_id:
            return mask_utils.decode(ann['segmentation']), None
    return None, "Component not found for this image."


def get_component_mask_robust(image_name, component_name, image_name_to_anns, cat_id_to_name, h, w):
    """A robust function to get a mask, which can handle both RLE and polygon formats."""
    normalized_name = os.path.splitext(os.path.basename(image_name))[0]
    if normalized_name not in image_name_to_anns and len(image_name_to_anns) == 1:
        annotations = next(iter(image_name_to_anns.values()), None)
    else:
        annotations = image_name_to_anns.get(normalized_name)
    if annotations is None:
        return None, "No annotations found."
    cat_id = next((cid for cid, name in cat_id_to_name.items() if name == component_name), None)
    if cat_id is None:
        return None, f"Component '{component_name}' not found."
    for ann in annotations:
        if ann['category_id'] == cat_id:
            segmentation = ann['segmentation']
            if isinstance(segmentation, dict) and 'counts' in segmentation:
                return mask_utils.decode(segmentation), None
            elif isinstance(segmentation, list):
                rles = mask_utils.frPyObjects(segmentation, h, w)
                mask = mask_utils.decode(rles)
                if len(mask.shape) > 2:
                    mask = np.sum(mask, axis=2).astype(bool)
                return mask, None
            else:
                return None, "Unknown format."
    return None, "Component not found in annotations."


def find_corner_views(component_name, component_to_views_map, images):
    """Finds the best available view from each of the four corners of the car."""
    available_views = component_to_views_map.get(component_name, [])
    if not available_views:
        return []
    view_positions = {img_id: images[img_id].tvec for _, img_id in available_views}
    all_pos = np.array(list(view_positions.values()))
    min_x, max_x = np.min(all_pos[:, 0]), np.max(all_pos[:, 0])
    min_z, max_z = np.min(all_pos[:, 2]), np.max(all_pos[:, 2])
    center_x, center_z = (min_x + max_x) / 2, (min_z + max_z) / 2
    quadrants = {
        "Front-Left": lambda p: p[0] < center_x and p[2] > center_z,
        "Front-Right": lambda p: p[0] > center_x and p[2] > center_z,
        "Rear-Left": lambda p: p[0] < center_x and p[2] < center_z,
        "Rear-Right": lambda p: p[0] > center_x and p[2] < center_z,
    }
    corner_views = []
    for quad_name, is_in_quadrant in quadrants.items():
        quad_views = [(area, img_id) for area, img_id in available_views if is_in_quadrant(view_positions[img_id])]
        if quad_views:
            corner_views.append(quad_views[0][1])
    return list(set(corner_views))


# --- Evaluation Metrics ---


def calculate_iou(mask1, mask2):
    """Calculates the Intersection over Union (IoU) for two masks."""
    if mask1 is None or mask2 is None:
        return 0.0
    mask1_bool, mask2_bool = mask1.astype(bool), mask2.astype(bool)
    intersection = np.logical_and(mask1_bool, mask2_bool)
    union = np.logical_or(mask1_bool, mask2_bool)
    if np.sum(union) == 0:
        return 1.0 if np.sum(intersection) == 0 else 0.0
    return np.sum(intersection) / np.sum(union)


def calculate_projection_spread(transferred_mask, target_vehicle_mask):
    """Calculates the percentage of projected pixels that fall outside the main vehicle body."""
    if transferred_mask is None or target_vehicle_mask is None:
        return 0.0
    transferred_bool, vehicle_bool = transferred_mask.astype(bool), target_vehicle_mask.astype(bool)
    false_positives = np.logical_and(transferred_bool, np.logical_not(vehicle_bool))
    total_projected_pixels = np.sum(transferred_bool)
    if total_projected_pixels == 0:
        return 0.0
    return np.sum(false_positives) / total_projected_pixels


def calculate_average_contour_distance(mask_pred, mask_gt):
    """Calculates the average distance from the predicted mask's contour to the ground truth mask's contour."""
    if mask_pred is None or mask_gt is None or not np.any(mask_pred) or not np.any(mask_gt):
        return float('inf')
    contours_pred, _ = cv2.findContours(mask_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_gt, _ = cv2.findContours(mask_gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours_pred or not contours_gt:
        return float('inf')
    points_pred = np.squeeze(contours_pred[0])
    points_gt = np.squeeze(contours_gt[0])
    if points_pred.ndim == 1:
        points_pred = np.expand_dims(points_pred, axis=0)
    if points_gt.ndim == 1:
        points_gt = np.expand_dims(points_gt, axis=0)
    distances = cdist(points_pred, points_gt)
    min_distances = np.min(distances, axis=1)
    return np.mean(min_distances)


def sharpen_image(image):
    """
    Sharpens an image using the Unsharp Masking technique.
    """
    # Create a blurred version of the image
    blurred = cv2.GaussianBlur(image, (0, 0), 3)

    # Subtract the blurred image from the original to create the "mask"
    # The '1.5' is the sharpening amount, and the '0' is a constant to add
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

    return sharpened


def get_mask_size_threshold(component_name):
    """
    Returns the minimum mask size ratio for a given component.
    """
    filter_threshold_map = {
        # Using the specific values you provided
        "mirror_l_assy": 0.00075,
        "mirror_r_assy": 0.00075,
        "grille": 0.002,
        "headlamp_l_assy": 0.001,
        "headlamp_r_assy": 0.001,
        "taillamp_l_assy": 0.001,
        "taillamp_r_assy": 0.001,
        "bonnet": 0.004,
        "bootlid": 0.004,
        "bumper_f/cover": 0.004,
        "bumper_b/cover": 0.004,
        "door_fl_assy": 0.004,
        "door_fr_assy": 0.004,
        "door_bl_assy": 0.004,
        "door_br_assy": 0.004,
        "windshield_f": 0.004,
        "windshield_b": 0.004,
        "wing_fl": 0.004,
        "wing_fr": 0.004,
        "wing_bl": 0.004,
        "wing_br": 0.004,
    }
    # Return the specific threshold, or a safe default (0.05%) if not in the map
    return filter_threshold_map.get(component_name, 0.0005)
