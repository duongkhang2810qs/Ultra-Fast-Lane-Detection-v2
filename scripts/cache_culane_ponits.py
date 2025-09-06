import os
import cv2
import numpy as np
import tqdm
import json
import argparse
import re

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='The root of the dataset')
    return parser

def norm_key(img_rel: str) -> str:
    rel = img_rel.replace("\\", "/").lstrip("/")
    if rel.startswith("images/"):
        rel = "image/" + rel[len("images/"):]
    if not rel.startswith("image/"):
        rel = "image/" + os.path.basename(rel)
    return rel

def parse_line_points(line: str):
    # bắt số robust (kể cả 1e-3)
    nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', line)
    if len(nums) < 4 or len(nums) % 2 != 0:
        return None
    xs = [float(nums[i]) for i in range(0, len(nums), 2)]
    ys = [float(nums[i+1]) for i in range(0, len(nums), 2)]
    return list(zip(xs, ys))

def interp_x_at_y(pts_xy, y):
    n = len(pts_xy)
    for i in range(n - 1):
        x1, y1 = pts_xy[i]
        x2, y2 = pts_xy[i + 1]
        if (y1 <= y <= y2) or (y2 <= y <= y1):
            if y2 == y1:
                return float(x1)
            t = (y - y1) / (y2 - y1)
            return float(x1 + t * (x2 - x1))
    return -1.0

if __name__ == '__main__':
    args = get_args().parse_args()
    culane_root = args.root
    train_list = os.path.join(culane_root, 'list/train_gt.txt')
    with open(train_list, 'r') as fp:
        lines = [ln.strip() for ln in fp if ln.strip()]

    # 35 mốc y CULane (ảnh cao ~590)
    the_anno_row_anchor = np.arange(250, 591, 10).astype(np.float32)

    cache_dict = {}

    for line in tqdm.tqdm(lines):
        parts = line.split()  # tách theo whitespace
        if len(parts) < 2:
            continue

        img_rel  = parts[0].lstrip('/')
        mask_rel = parts[1].lstrip('/')

        img_path  = os.path.join(culane_root, img_rel)
        mask_path = os.path.join(culane_root, mask_rel)

        # .lines.txt cạnh ảnh, cùng tên (không phụ thuộc đuôi)
        img_base  = os.path.splitext(os.path.join(culane_root, img_rel))[0]
        txt_path  = img_base + '.lines.txt'

        img = cv2.imread(img_path)
        if img is None:
            continue
        H, W = img.shape[:2]

        label_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if label_img is None or label_img.shape[:2] != (H, W):
            label_img = None

        # đọc .lines.txt
        lane_lines = []
        if os.path.isfile(txt_path):
            with open(txt_path, 'r') as f:
                for raw in f:
                    pts = parse_line_points(raw)
                    if not pts:
                        continue
                    pts = sorted(pts, key=lambda t: t[1])
                    if len(pts) >= 2:
                        lane_lines.append(pts)

        # init
        all_points = np.zeros((4, 35, 2), dtype=np.float32)
        all_points[:, :, 1] = np.tile(the_anno_row_anchor, (4, 1))
        all_points[:, :, 0] = -99999.0

        if not lane_lines:
            cache_dict[norm_key(img_rel)] = all_points.tolist()
            continue

        # lane order ưu tiên từ mask
        lane_orders = [-1] * len(lane_lines)  # 0..3
        if label_img is not None:
            u = np.unique(label_img)
            ids = [int(v) for v in u if v > 0]
            ids.sort()
            id_map = {old: i + 1 for i, old in enumerate(ids)}  # -> 1..K
            for li, pts in enumerate(lane_lines):
                mid = pts[len(pts) // 2]
                mx = int(round(mid[0])); my = int(round(mid[1]))
                if 0 <= mx < W and 0 <= my < H:
                    v = int(label_img[my, mx])
                    if v in id_map:
                        lane_orders[li] = id_map[v] - 1  # về 0..3

        # fallback: trái→phải theo x ở đáy
        unassigned = [i for i, o in enumerate(lane_orders) if o < 0]
        if unassigned:
            bottoms = []
            for li in unassigned:
                xb = interp_x_at_y(lane_lines[li], H - 1)
                if xb >= 0:
                    bottoms.append((xb, li))
            bottoms.sort(key=lambda t: t[0])
            used = set([o for o in lane_orders if o >= 0])
            free_slots = [s for s in range(4) if s not in used]
            for slot, (_, li) in zip(free_slots, bottoms[:len(free_slots)]):
                lane_orders[li] = slot

        # điền x tại 35 mốc y
        for li, pts in enumerate(lane_lines):
            slot = lane_orders[li]
            if slot < 0 or slot > 3:
                continue
            for x, y in pts:
                pos = int(round((y - 250.0) / 10.0))
                if 0 <= pos < 35:
                    all_points[slot, pos, 0] = float(x)

        cache_dict[norm_key(img_rel)] = all_points.tolist()

    out_path = os.path.join(culane_root, 'culane_anno_cache.json')
    with open(out_path, 'w') as f:
        json.dump(cache_dict, f)
    print("✅ wrote", out_path, "entries:", len(cache_dict))
