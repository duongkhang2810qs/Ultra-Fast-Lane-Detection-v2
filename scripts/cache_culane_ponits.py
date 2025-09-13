import os
import cv2
import numpy as np
import tqdm
import json
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='The root of the dataset')
    return parser


if __name__ == '__main__':
    args = get_args().parse_args()
    culane_root = args.root
    train_list = os.path.join(culane_root, 'list/train_gt.txt')
    with open(train_list, 'r') as fp:
        res = fp.readlines()
    cache_dict = {}
    for line in tqdm.tqdm(res):
        info = line.split(' ')

        label_path = os.path.join(culane_root, info[1][1:])
        # label_img = cv2.imread(label_path)[:,:,0]
        label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # CHANGED

        txt_path = info[0][1:].replace('jpg','lines.txt')
        txt_path = os.path.join(culane_root, txt_path)
        lanes = open(txt_path, 'r').readlines()

        all_points = np.zeros((4,35,2), dtype=np.float32)
        the_anno_row_anchor = np.array([
            250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390,
            400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540,
            550, 560, 570, 580, 590
        ], dtype=np.float32)
        all_points[:,:,1] = np.tile(the_anno_row_anchor, (4,1))
        all_points[:,:,0] = -99999  # init using no lane

        # Ảnh không có lane -> bỏ qua (tránh crash)
        if len(lanes) == 0:  # ADDED
            cache_dict[info[0][1:]] = all_points.tolist()  # ADDED
            continue  # ADDED

        h, w = label_img.shape  # ADDED

        for lane_idx , lane in enumerate(lanes):
            ll = lane.strip().split(' ')
            if len(ll) < 4:  # không đủ cặp (x,y) -> bỏ qua lane này  # ADDED
                continue  # ADDED

            point_x = ll[::2]
            point_y = ll[1::2]

            mid_idx = len(point_x) // 2  # ADDED
            # mid_x = int(float(point_x[int(len(point_x)/2)]))
            # mid_y = int(float(point_y[int(len(point_x)/2)]))
            mid_x = int(round(float(point_x[mid_idx])))  # CHANGED
            mid_y = int(round(float(point_y[mid_idx])))  # CHANGED

            # Kiểm tra biên; nếu điểm giữa ra ngoài ảnh -> bỏ lane  # ADDED
            if not (0 <= mid_x < w and 0 <= mid_y < h):  # ADDED
                continue  # ADDED

            # lane_order từ mask mong đợi 1..4  # ADDED
            lane_order = int(label_img[mid_y, mid_x])  # CHANGED (bỏ -1, -1)
            if not (1 <= lane_order <= 4):  # ADDED
                continue  # ADDED

            for i in range(len(point_x)):
                try:  # ADDED (chống ValueError từng điểm)
                    p1x = float(point_x[i])  # CHANGED (ép float)
                    # pos = (float(point_y[i]) - 250) / 10
                    pos = int(round((float(point_y[i]) - 250.0) / 10.0))  # CHANGED
                except ValueError:
                    continue  # ADDED

                # Kiểm tra biên anchor Y trước khi gán  # ADDED
                if 0 <= pos < all_points.shape[1]:
                    all_points[lane_order - 1, pos, 0] = p1x  # giữ float
                # else: ngoài dải anchor -> bỏ qua  # ADDED

        cache_dict[info[0][1:]] = all_points.tolist()

    with open(os.path.join(culane_root, 'culane_anno_cache.json'), 'w') as f:
        json.dump(cache_dict, f)
