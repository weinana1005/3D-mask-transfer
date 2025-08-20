import random
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

# --- Explicit Imports ---
from helpers_cal import calculate_iou, find_corner_views, get_component_mask
from helpers_masktransfer import transfer_mask_v3, transfer_mask_v3_fast


def run_subsample_analysis_on_fixed_targets(
    component_name,
    subsample_step,
    targets_to_test,
    # Pass in all required data
    component_to_views_map,
    images,
    image_name_to_anns,
    cat_id_to_name,
    image_id_to_rgb_path,
    image_id_to_depth_path,
    image_id_to_mask_path,
    cameras,
):
    """
    Runs the 4-corner analysis for a specific component and subsample size
    on a predefined list of target images.
    Returns the list of IoU scores and the total time taken.
    """
    print(f"  - Analyzing '{component_name}' with subsample_step={subsample_step}...")

    source_ids = find_corner_views(component_name, component_to_views_map, images)
    if len(source_ids) < 2:
        return None, 0

    iou_scores = []

    start_time = time.time()
    for target_id in targets_to_test:
        target_gt_mask, _ = get_component_mask(
            images[target_id].name, component_name, image_name_to_anns, cat_id_to_name
        )

        projected_masks = []
        for sid in source_ids:
            source_mask, _ = get_component_mask(images[sid].name, component_name, image_name_to_anns, cat_id_to_name)
            if source_mask is not None:
                # Pass all required arguments to the transfer function
                transferred_mask = transfer_mask_v3_fast(
                    sid,
                    target_id,
                    source_mask,
                    component_name,
                    images,
                    cameras,
                    image_id_to_rgb_path,
                    image_id_to_depth_path,
                    image_id_to_mask_path,
                    subsample_step=subsample_step,
                )
                if transferred_mask is not None:
                    projected_masks.append(transferred_mask)

        if len(projected_masks) < len(source_ids):
            continue

        target_h, target_w, _ = cv2.imread(image_id_to_rgb_path[target_id]).shape
        union_mask = np.zeros((target_h, target_w), dtype=np.uint8)
        for mask in projected_masks:
            union_mask = cv2.bitwise_or(union_mask, mask)

        iou_scores.append(calculate_iou(union_mask, target_gt_mask))

    end_time = time.time()
    total_time = end_time - start_time

    return iou_scores, total_time


def run_multi_view(
    num_source_views,
    example_num,
    specific_components,
    # Pass in all required data
    component_to_views_map,
    images,
    image_name_to_anns,
    cat_id_to_name,
    image_id_to_rgb_path,
    image_id_to_depth_path,
    image_id_to_mask_path,
    cameras,
):
    """
    Runs the multi-view check for a component from the specified list,
    guaranteeing a target with a ground truth mask.
    """
    print(f"\n{'='*20} Running {num_source_views}-View IoU Evaluation Example {example_num} {'='*20}\n")

    source_ids, target_id, component_name = [], None, None
    num_required_views = num_source_views + 1

    potential_components = [
        name
        for name in specific_components
        if name in component_to_views_map and len(component_to_views_map[name]) >= num_required_views
    ]

    if not potential_components:
        print(
            f"Error: Could not find any of your specific components with at least {num_required_views} views. Skipping."
        )
        return

    component_name = random.choice(potential_components)
    sorted_views = component_to_views_map[component_name]

    source_ids = [view[1] for view in sorted_views[:num_source_views]]
    target_id = random.choice([view[1] for view in sorted_views[num_source_views:]])

    print(f"Found valid set for component '{component_name}':")
    for i, sid in enumerate(source_ids):
        print(f"  - Source {chr(65+i)}: {images[sid].name}")
    print(f"  - Target:   {images[target_id].name}")

    projected_masks = []
    for i, sid in enumerate(source_ids):
        print(f"\n--- Projecting from Source {chr(65+i)} to Target ---")
        source_mask, error_msg = get_component_mask(
            images[sid].name, component_name, image_name_to_anns, cat_id_to_name
        )

        if source_mask is not None:
            transferred_mask = transfer_mask_v3(
                sid,
                target_id,
                source_mask,
                component_name,
                images,
                cameras,
                image_id_to_rgb_path,
                image_id_to_depth_path,
                image_id_to_mask_path,
            )
            if transferred_mask is not None:
                projected_masks.append(transferred_mask)
        else:
            print(f"Could not get source mask. Error: {error_msg}")

    if len(projected_masks) < 2:
        print("Failed to get enough successful projections. Skipping.")
        return

    target_h, target_w, _ = cv2.imread(image_id_to_rgb_path[target_id]).shape
    heatmap = np.zeros((target_h, target_w), dtype=np.uint8)
    for mask in projected_masks:
        heatmap += mask // 255

    agreement_threshold = int(np.ceil(len(projected_masks) / 2))
    final_mask = np.where(heatmap >= agreement_threshold, 255, 0).astype(np.uint8)

    target_gt_mask, _ = get_component_mask(images[target_id].name, component_name, image_name_to_anns, cat_id_to_name)
    iou_score = calculate_iou(final_mask, target_gt_mask)
    print(f"\n>>> Ground Truth Found! IoU Score: {iou_score:.4f} <<<")

    target_rgb = cv2.cvtColor(cv2.imread(image_id_to_rgb_path[target_id]), cv2.COLOR_BGR2RGB)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))
    fig.suptitle(f"{num_source_views}-View Check for '{component_name}' (IoU: {iou_score:.4f})", fontsize=24, y=1.0)

    ax1.imshow(target_rgb)
    ax1.imshow(heatmap, cmap='hot', alpha=0.6)
    ax1.set_title("1. Confidence Heatmap", fontsize=18)
    ax1.axis('off')

    ax2.imshow(target_rgb)
    overlay_res = target_rgb.copy()
    roi_res = overlay_res[final_mask > 0]
    blended_res = (roi_res * 0.5 + np.array([0, 255, 0], dtype=np.uint8) * 0.5).astype(np.uint8)
    overlay_res[final_mask > 0] = blended_res
    ax2.imshow(overlay_res)
    ax2.set_title(f"2. Final Mask (Agreement >= {agreement_threshold})", fontsize=18)
    ax2.axis('off')

    ax3.imshow(target_rgb)
    overlay_gt = target_rgb.copy()
    roi_gt = overlay_gt[target_gt_mask > 0]
    blended_gt = (roi_gt * 0.5 + np.array([255, 255, 0], dtype=np.uint8) * 0.5).astype(np.uint8)
    overlay_gt[target_gt_mask > 0] = blended_gt
    ax3.imshow(overlay_gt)
    ax3.set_title("3. Ground Truth Mask", fontsize=18)
    ax3.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
