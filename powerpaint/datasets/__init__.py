import random
from typing import List
import numpy as np
import torch
from torch.utils.data import IterableDataset

# from .laion import LaionIterJsonDataset
# from .openimage import OpenImageBLIPaug_Dataset
from .nuscenes import NuScenesIterJsonDataset


class ProbPickingDataset(IterableDataset):
    def __init__(self, datasets):
        """
        Args:
            datasets: list of dict containing {"dataset": dataset, "prob": probability}
        """
        super().__init__()
        self.datasets = datasets
        probs = [d["prob"] for d in datasets]
        self.probs = np.array(probs) / np.sum(probs)
        
    def __len__(self):
        return sum(len(d["dataset"]) for d in self.datasets)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        # 为每个worker创建独立的迭代器和随机种子
        if worker_info is not None:
            # 为每个worker设置不同的随机种子
            np.random.seed(worker_info.seed % np.iinfo(np.uint32).max)
        
        iterators = [iter(d["dataset"]) for d in self.datasets]
        
        while True:
            # 随机选择一个数据集
            dataset_idx = np.random.choice(len(self.datasets), p=self.probs)
            
            try:
                # 尝试从选中的数据集获取数据
                yield next(iterators[dataset_idx])
            except StopIteration:
                # 如果当前数据集已经遍历完，重新初始化该数据集的迭代器
                iterators[dataset_idx] = iter(self.datasets[dataset_idx]["dataset"])
                yield next(iterators[dataset_idx])


__all__ = ["ProbPickingDataset", "NuScenesIterJsonDataset"]
