import math
from typing import List, cast

import matplotlib.pyplot as plt
import numpy as np
import nrrd
from numpy import ndarray
import pandas as pd

from swcgeom.core import Tree
from swcgeom.analysis import draw
from swcgeom.core import to_subtree


def get_sampling(head: Tree.Node, tail: Tree.Node, sample_num):
    if head.is_tip():
        return None
    distance = head.distance(tail)
    direction = tail.xyz() - head.xyz()
    direction = direction / np.linalg.norm(direction)
    iter_time = math.floor(distance)
    iter_time = min(sample_num, iter_time)
    points = []
    for i in range(iter_time):
        points.append(head.xyz() + direction * i)
    return points


def sampling_segment(root: Tree.Node, node: Tree.Node, sample_num):
    segment = []
    if sample_num > 0:
        if not root.is_tip():
            points = get_sampling(root, node, sample_num)
            sample_num -= len(points)
            segment += points
        else:
            segment.append(node.xyz())

    while sample_num > 0:
        if not node.is_tip():
            points = get_sampling(node, node.children()[0], sample_num)
            sample_num -= len(points)
            segment += points
        else:
            segment.append(node.xyz())
            break
        node = node.children()[0]

    return segment


class BifurcationHandler:
    tree: Tree
    img: ndarray

    def __init__(self, tree: Tree, img_dir: str):
        self.tree = tree
        data, header = nrrd.read(filename=img_dir)
        self.img = data

    def handle(self, idx: int | np.integer, threshold) -> Tree:
        node = self.tree.node(idx)
        children = node.children()
        left = children[0]
        right = children[1]

        sample_num = 10
        left_segment = sampling_segment(node, left, sample_num)
        right_segment = sampling_segment(node, right, sample_num)

        left_intensity = self.segment_intensity(left_segment)
        right_intensity = self.segment_intensity(right_segment)
        node = left if left_intensity < right_intensity else right

        low_intensity = min(right_intensity,left_intensity)
        difference = abs(left_intensity - right_intensity) / low_intensity
        #print(left_intensity, right_intensity)
        #draw(self.tree)
        #draw(node.get_branch())

        if difference > threshold or low_intensity < 100:
            print(left_intensity, right_intensity)
            return node.idx

    def get_intensity(self, idx):
        x, y, z = self.tree.node(idx).xyz()

        return self.img[round(x)][round(y)][round(z)]

    def segment_intensity(self, segment):
        intensity_sum = 0
        for pos in segment:
            x, y, z = pos
            intensity_sum += self.img[round(x)][round(y)][round(z)]
        return intensity_sum


if __name__ == '__main__':
    log_dir = r"C:\Users\80121\Desktop\dendriteImageNrrd\log.txt"
    csv_dir = r"C:\Users\80121\Desktop\Brain_t_game_record_after0201.csv"
    df = pd.read_csv(csv_dir)
    wrong_bp = {}

    for row_index, row in df.iterrows():
        swc_id = row['SWCId']
        bp = [x for x in row['WrongBP'].split(',') if x != ' ']
        if wrong_bp.get(swc_id) is None:
            wrong_bp[swc_id] = set(bp)
        else:
            wrong_bp[swc_id] = wrong_bp[swc_id].union(set(bp))

    print(wrong_bp)

    with open(log_dir, 'a+') as test:
        test.truncate(0)

    for swc_id, points in wrong_bp.items():
        with open(log_dir, 'a') as f:
            f.write(swc_id)
        try:
            tree = Tree.from_swc(rf"C:\Users\80121\Desktop\dendriteSWC\{swc_id}swc_sorted.swc")
        except :
            print("swc error,", swc_id)
            with open(log_dir, 'a') as f:
                f.write(" swc error\n")
            continue

        img_dir = rf"C:\Users\80121\Desktop\dendriteImageNrrd\{swc_id}nrrd"
        ch = BifurcationHandler(tree, img_dir)
        idxs = []
        for node in points:
            result = ch.handle(int(node) - 1, 5)
            if result is not None:
                idxs.append(result)
        ch.tree = to_subtree(ch.tree, idxs)
        print(ch.tree.number_of_nodes())
        ch.tree.to_swc(rf"C:\Users\80121\Desktop\dendriteImageNrrd\{swc_id}swc")
        print(swc_id, "done")
        with open(r"C:\Users\80121\Desktop\dendriteImageNrrd\log.txt", 'a') as f:
            for idx in idxs:
                f.write(f" {idx}")
            f.write("\n")
    # draw(tree, camera='yz')
    # plt.show()
