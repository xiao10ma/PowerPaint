import os
from PIL import Image
import numpy as np
from torch.utils.data import IterableDataset
import json
from torchvision import transforms
import torch
import random

CAM_LIST = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

class NuScenesIterJsonDataset(IterableDataset):
    def __init__(
        self,
        transforms,
        pipeline,
        task_prompt,
        name=None,
        img_meta_path=None,
        resolution=None,
        **kwargs,
    ):
        super().__init__()
        self.img_meta_path = img_meta_path
        self.img_meta = json.load(open(img_meta_path))

        self.transforms = transforms
        self.pipeline = pipeline
        self.task_prompt = task_prompt
        self.resolution = resolution

        total_len = 0
        for scene_name, cam_dict in self.img_meta.items():
            for cam_name, cam_meta in cam_dict.items():
                total_len += len(cam_meta["image"])
        self.total_len = total_len

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        # 获取数据索引列表
        all_indices = []
        for scene_name, cam_dict in self.img_meta.items():
            for cam_name, cam_meta in cam_dict.items():
                for idx in range(len(cam_meta["image"])):
                    all_indices.append((scene_name, cam_name, idx))
        
        # 如果是多进程，划分数据
        if worker_info is not None:
            per_worker = int(np.ceil(len(all_indices) / worker_info.num_workers))
            worker_id = worker_info.id
            start_idx = worker_id * per_worker
            end_idx = min(start_idx + per_worker, len(all_indices))
            # 只处理该worker负责的部分数据
            indices = all_indices[start_idx:end_idx]
        else:
            indices = all_indices
        
        while True:  # 循环读取数据
            for scene_name, cam_name, idx in indices:
                cam_meta = self.img_meta[scene_name][cam_name]
                img_path = cam_meta["image"][idx]
                mask_path = cam_meta["mask"][idx]
                
                output = {}
                output["pixel_values"] = self.transforms(Image.open(img_path))
                
                mask = Image.open(mask_path).convert("L")
                mask = mask.resize((self.resolution, self.resolution), Image.LANCZOS)
                mask = np.array(mask).astype(np.float32)

                if len(mask.shape) == 3:
                    mask = mask[:, :, 0]
                mask[mask > 128] = 255
                mask[mask <= 128] = 0

                mask = Image.fromarray(mask.astype("uint8"))
                mask = transforms.ToTensor()(mask)
                mask[mask != 0] = 1
                output["mask"] = mask

                alpha = torch.tensor((1.0, 0.0))
                output["tradeoff"] = alpha
                
                # 10% probability to drop all conditions for unconditional generation
                if random.random() < 0.1:
                    cur_promptA = cur_promptB = cur_prompt = ""
                else:
                    cur_promptA = self.task_prompt.context_inpainting.placeholder_tokens
                    cur_promptB = self.task_prompt.context_inpainting.placeholder_tokens
                    cur_prompt = ""
                
                # IMPORTANT, remember to convert prompt for multi-vector embeddings
                cur_promptA = self.pipeline.maybe_convert_prompt(cur_promptA, self.pipeline.tokenizer)
                cur_promptB = self.pipeline.maybe_convert_prompt(cur_promptB, self.pipeline.tokenizer)
                cur_prompt = self.pipeline.maybe_convert_prompt(cur_prompt, self.pipeline.tokenizer)

                output["input_idsA"], output["input_idsB"], output["input_ids"] = self.pipeline.tokenizer(
                    [cur_promptA, cur_promptB, cur_prompt],
                    max_length=self.pipeline.tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).input_ids

                yield output

    def __len__(self):
        return self.total_len
