from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp
import glob
import re
import random
from collections import defaultdict

from torchreid.data import ImageDataset


class AnonymizedReIDTSH(ImageDataset):

    def __init__(self, root="", max_samples_per_person=100, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root)
        self.max_samples_per_person = max_samples_per_person

        # Define paths for different sets
        self.train_dir = osp.join(self.dataset_dir, "reid_diff/train")
        self.anon_dir = osp.join(self.dataset_dir, "reid_dp2/test")
        self.non_anon_dir = osp.join(self.dataset_dir, "reid_ori/test")

        # Pattern to extract person ID and camera ID from filenames
        pattern = re.compile(r".*_p(\d+)_.*_c(\d+)\.jpg")

        # Process training images
        train = []
        train_paths = glob.glob(osp.join(self.train_dir, "*.jpg"))
        c = 0
        for img_path in train_paths:
            match = pattern.match(img_path)
            if match:
                pid = int(match.group(1))
                camid = c
                c += 1
                train.append((img_path, pid, camid))

        # Process anonymized images (query set)
        query = []
        anon_paths = glob.glob(osp.join(self.anon_dir, "*.jpg"))
        c = 0
        for img_path in anon_paths:
            match = pattern.match(img_path)
            if match:
                pid = int(match.group(1))
                camid = c
                c += 1
                query.append((img_path, pid, camid))

        # Process non-anonymized images (gallery set)
        gallery = []
        non_anon_paths = glob.glob(osp.join(self.non_anon_dir, "*.jpg"))
        c = 0
        for img_path in non_anon_paths:
            match = pattern.match(img_path)
            if match:
                pid = int(match.group(1))
                camid = c
                c += 1
                gallery.append((img_path, pid, camid))

        # Balance data
        print(f"Before balancing - Train: {len(train)}, Query: {len(query)}, Gallery: {len(gallery)}")
        train = self.balance_data(train)
        query = self.balance_data(query)
        gallery = self.balance_data(gallery)
        print(f"After balancing - Train: {len(train)}, Query: {len(query)}, Gallery: {len(gallery)}")

        # Relabel PIDs
        train = self.relabel_pid(train)
        # Keep original PIDs for query and gallery but ensure they match
        self.ensure_pid_consistency(query, gallery)

        print(f"Final dataset sizes - Train: {len(train)}, Query: {len(query)}, Gallery: {len(gallery)}")

        # Print some statistics
        self.print_dataset_statistics(train, query, gallery)

        super(AnonymizedReIDTSH, self).__init__(train, query, gallery, **kwargs)

    def balance_data(self, data_list):
        """Balance the number of images per person."""
        pid_dict = defaultdict(list)
        for item in data_list:
            pid_dict[item[1]].append(item)

        balanced_data = []
        for pid, items in pid_dict.items():
            if len(items) > self.max_samples_per_person:
                items = random.sample(items, self.max_samples_per_person)
            balanced_data.extend(items)
        return balanced_data

    def relabel_pid(self, data_list):
        """Relabel PIDs to ensure they are continuous."""
        pid_container = set()
        for _, pid, _ in data_list:
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        return [(img_path, pid2label[pid], camid) for img_path, pid, camid in data_list]

    def ensure_pid_consistency(self, query, gallery):
        """Ensure that all query PIDs exist in gallery."""
        query_pids = set(pid for _, pid, _ in query)
        gallery_pids = set(pid for _, pid, _ in gallery)

        missing_pids = query_pids - gallery_pids
        if missing_pids:
            raise ValueError(f"The following PIDs are in query but not in gallery: {missing_pids}")

    def print_dataset_statistics(self, train, query, gallery):
        """Print dataset statistics."""
        num_train_pids = len(set(t[1] for t in train))
        num_train_cams = len(set(t[2] for t in train))
        num_query_pids = len(set(q[1] for q in query))
        num_query_cams = len(set(q[2] for q in query))
        num_gallery_pids = len(set(g[1] for g in gallery))
        num_gallery_cams = len(set(g[2] for g in gallery))

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print(f"  train    | {num_train_pids:5d} | {len(train):8d} | {num_train_cams:9d}")
        print(f"  query    | {num_query_pids:5d} | {len(query):8d} | {num_query_cams:9d}")
        print(f"  gallery  | {num_gallery_pids:5d} | {len(gallery):8d} | {num_gallery_cams:9d}")
        print("  ----------------------------------------")
