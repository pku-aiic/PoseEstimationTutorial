"""
训练数据集地址: /data1/lxxgck/coco17
测试数据集地址: /data1/lxxgck/testset
    共8位测试用户，ID分别为30、33、41、44、45、48、49、50。每个用户一个子目录。
    每位用户目录下有7个动作的子文件夹。按顺序命名，如30_1代表30号用户的第一个动作。
    keyframes目录下为本次要inference的图片 kpt.csv为标注的关键点
    标注格式为关键点名称,gt_x,gt_y,图片名,图片宽,图片高
测试脚本如下:
    由于大家的inference代码各不相同, 请大家根据自己的代码自行填充get_normed_kpt_dist_list中相关部分
"""

import os
import math
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


def compute_dist(point1, point2):
    # 计算两个点的距离
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def load_gt_dict(user_id, action_id):
    # 给定当前标注的csv格式, 转换成dict
    anno_dict = {}
    csv_path = os.path.join(TESTSET_PATH, str(
        user_id), f'{user_id}_{action_id}', 'kpt.csv')
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line == '':
            continue
        kpt_name, gt_x, gt_y, jpg_name, image_w, image_h = line.split(',')
        if jpg_name not in anno_dict.keys():
            anno_dict[jpg_name] = {
                "image_w": 0,
                "image_h": 0,
                "keypoint": {}
            }
        anno_dict[jpg_name]["keypoint"][kpt_name] = {
            "gt_x": int(gt_x),
            "gt_y": int(gt_y),
        }
        anno_dict[jpg_name]["image_w"] = int(image_w)
        anno_dict[jpg_name]["image_h"] = int(image_h)
    return anno_dict


def get_normed_kpt_dist_list(user_id, action_id):
    # 计算预测点坐标与lable的正则化过的距离
    dist_normed_list = []
    keyframe_dir = os.path.join(TESTSET_PATH, str(
        user_id), f'{user_id}_{action_id}', 'keyframes')
    # load gt as dict
    anno_dict_all = load_gt_dict(user_id, action_id)
    for img_name in os.listdir(keyframe_dir):
        anno_dict = anno_dict_all[img_name]
        # 取图片宽度作正则化

        try:
            img_w = anno_dict["image_w"]
        except:
            print(anno_dict)
            assert 1 == 0
        # TODO: given img_path, inference image
        img_path = os.path.join(keyframe_dir, img_name)

        # TODO: convert network output to standard format for comparison with gt

        kpt_anno_dict = anno_dict["keypoint"]
        for kpt_name in kpt_anno_dict.keys():
            gt_x = kpt_anno_dict[kpt_name]["gt_x"]
            gt_y = kpt_anno_dict[kpt_name]["gt_y"]

            # TODO: assign pred_x, pred_y
            pred_x = 100
            pred_y = 200

            dist = compute_dist([gt_x, gt_y], [pred_x, pred_y])
            dist_normed = dist / img_w
            dist_normed_list.append(dist_normed)
    return dist_normed_list


TESTSET_PATH = "/data1/lxxgck/testset"
USER_ID_LIST = [30, 33, 41, 44, 45, 48, 49, 50]
ACTION_ID_LIST = [1, 2, 3, 4, 5, 6, 7]

res_list = []
for user_id in tqdm(USER_ID_LIST):
    for action_id in ACTION_ID_LIST:
        normed_kpt_list = get_normed_kpt_dist_list(user_id, action_id)
        res_list.extend(normed_kpt_list)
print(np.mean(res_list))
