import json
import os
from nuscenes.nuscenes import NuScenes
from tqdm import tqdm
import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
import cv2

def bbox_to_corner3d(bbox):
    min_x, min_y, min_z = bbox[0]
    max_x, max_y, max_z = bbox[1]
    
    corner3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corner3d

def get_bound_2d_mask(corners_3d, K, pose, H, W):
    mask = np.zeros((H, W), dtype=np.uint8)
    corners_3d = np.dot(corners_3d, pose[:3, :3].T) + pose[:3, 3:].T
    if np.all(corners_3d[:, 2] < 0):
        return mask
    corners_3d[..., 2] = np.clip(corners_3d[..., 2], a_min=1e-3, a_max=None)
    corners_3d = np.dot(corners_3d, K.T)
    corners_2d = corners_3d[:, :2] / corners_3d[:, 2:]
    corners_2d = np.round(corners_2d).astype(int)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 5]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    return mask

def collect_nuscenes_image_paths(nuscenes_root, version='v1.0-trainval'):
    nusc = NuScenes(version=version, dataroot=nuscenes_root, verbose=True)
    output = {}

    mask_dir = os.path.join('./data/nuscenes/')

    for scene in tqdm(nusc.scene, desc="Processing scenes"):
        scene_name = scene['name']
        output[scene_name] = {}

        # 获取第一个sample的token
        sample_token = scene['first_sample_token']

        frame_cnt = 0
        while sample_token:
            frame_cnt += 1
            sample = nusc.get('sample', sample_token)

            bbox_world_list = []
            for ann_token in sample['anns']:
                ann = nusc.get('sample_annotation', ann_token)

                if len(ann['attribute_tokens']) > 0:
                    attr = nusc.get('attribute', ann['attribute_tokens'][0])
                    attr_name = attr['name']

                    if 'moving' in attr_name:
                        center = np.array(ann['translation'])
                        bbox_scale_factor = 1. # for fear of the shadow
                        size = np.array(ann['size']) * bbox_scale_factor
                        local2world = np.eye(4)
                        rotation = Quaternion(ann['rotation'])
                        local2world[:3, :3] = rotation.rotation_matrix
                        local2world[:3, 3] = center

                        width, length, height = size
                        bbox = [[-length / 2, -width / 2, -height / 2], [length / 2, width / 2, height / 2]]
                        bbox_corners_local = bbox_to_corner3d(bbox)
                        bbox_corners_world = bbox_corners_local @ local2world[:3, :3].T + local2world[:3, 3]
                        bbox_world_list.append(bbox_corners_world)

            for cam_name in sample['data']:
                if 'CAM' not in cam_name:
                    continue  # 跳过非相机数据

                cam_data = nusc.get('sample_data', sample['data'][cam_name])
                img_path = os.path.join(nuscenes_root, cam_data['filename'])

                cs = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
                ixt = np.array(cs['camera_intrinsic'])
                pose = nusc.get('ego_pose', cam_data['ego_pose_token'])
                sensor2ego = transform_matrix(cs['translation'], Quaternion(cs['rotation']), inverse=False)
                ego2world = transform_matrix(pose['translation'], Quaternion(pose['rotation']), inverse=False)
                c2w = ego2world @ sensor2ego

                # 初始化相机的image和mask字典
                if cam_name not in output[scene_name]:
                    output[scene_name][cam_name] = {
                        'image': [],
                        'mask': []
                    }

                # 添加图像路径到image列表
                output[scene_name][cam_name]['image'].append(img_path)

                # Project 3D bounding box to 2D image plane
                obj_bound = np.zeros((900, 1600), dtype=bool)
                for bbox_world in bbox_world_list:
                    obj_bound = obj_bound | get_bound_2d_mask(bbox_world, ixt, np.linalg.inv(c2w), 900, 1600)
                
                mask_path = os.path.join(mask_dir, scene_name, cam_name, img_path.split('/')[-1].replace('.jpg', '_mask.png'))
                os.makedirs(os.path.dirname(mask_path), exist_ok=True)
                cv2.imwrite(mask_path, obj_bound.astype(np.uint8) * 255)
                output[scene_name][cam_name]['mask'].append(os.path.abspath(mask_path))

            # 下一帧
            sample_token = sample['next'] if sample['next'] != "" else None
        print(f"Processed {frame_cnt} frames for scene {scene_name}")

    return output

if __name__ == "__main__":
    nuscenes_root = '/SSD_DISK/datasets/nuscenes'  # 替换为你的 NuScenes 数据集根目录
    version = 'v1.0-trainval'
    output_json = "nuscenes_image_paths.json"

    paths_dict = collect_nuscenes_image_paths(nuscenes_root, version)
    with open(output_json, 'w') as f:
        json.dump(paths_dict, f, indent=2)

    print(f"Image paths saved to {output_json}")
