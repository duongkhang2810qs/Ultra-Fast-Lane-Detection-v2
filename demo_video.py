import os, cv2, torch
import argparse
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from utils.common import merge_config, get_model
from utils.dist_utils import dist_print

# ==== chuyển logits → toạ độ (giữ nguyên logic bạn đang dùng) ====
def pred2coords(pred, row_anchor, col_anchor, local_width=1,
                original_image_width=1640, original_image_height=590):
    # shapes
    b, num_grid_row, num_cls_row, num_lane_row = pred['loc_row'].shape
    b, num_grid_col, num_cls_col, num_lane_col = pred['loc_col'].shape

    max_indices_row = pred['loc_row'].argmax(1).cpu()
    valid_row       = pred['exist_row'].argmax(1).cpu()

    max_indices_col = pred['loc_col'].argmax(1).cpu()
    valid_col       = pred['exist_col'].argmax(1).cpu()

    # move to cpu for softmax over axis 0
    pred['loc_row'] = pred['loc_row'].cpu()
    pred['loc_col'] = pred['loc_col'].cpu()

    coords = []

    # theo UFLD v2: 2 lane dọc (row branch) ở idx [1,2], 2 lane ngang (col branch) ở idx [0,3]
    row_lane_idx = [1, 2]
    col_lane_idx = [0, 3]

    # row branch: dự đoán x theo các y-anchor
    for i in row_lane_idx:
        tmp = []
        if valid_row[0, :, i].sum() > num_cls_row / 2:
            for k in range(valid_row.shape[1]):
                if valid_row[0, k, i]:
                    all_ind = torch.arange(
                        max(0,   max_indices_row[0, k, i] - local_width),
                        min(num_grid_row - 1, max_indices_row[0, k, i] + local_width) + 1
                    )
                    out_tmp = (pred['loc_row'][0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_row - 1) * original_image_width
                    tmp.append((int(out_tmp), int(row_anchor[k] * original_image_height)))
            coords.append(tmp)

    # col branch: dự đoán y theo các x-anchor
    for i in col_lane_idx:
        tmp = []
        if valid_col[0, :, i].sum() > num_cls_col / 4:
            for k in range(valid_col.shape[1]):
                if valid_col[0, k, i]:
                    all_ind = torch.arange(
                        max(0,   max_indices_col[0, k, i] - local_width),
                        min(num_grid_col - 1, max_indices_col[0, k, i] + local_width) + 1
                    )
                    out_tmp = (pred['loc_col'][0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_col - 1) * original_image_height
                    tmp.append((int(col_anchor[k] * original_image_width), int(out_tmp)))
            coords.append(tmp)

    return coords

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('config', help='path to config file (ví dụ: configs/culane_res34.py)')
    p.add_argument('--test_model', required=True, help='đường dẫn epXXX.pth bạn muốn dùng để infer')
    p.add_argument('--video', required=True, help='đường dẫn video đầu vào (mp4/avi...)')
    p.add_argument('--out', default='out.avi', help='đường dẫn video xuất (mặc định out.avi)')
    p.add_argument('--local_rank', type=int, default=0)
    return p.parse_args()

def build_transform(cfg):
    # resize theo (train_height / crop_ratio, train_width), rồi CẮT ĐÁY về train_height
    h_resize = int(cfg.train_height / cfg.crop_ratio)
    w_resize = cfg.train_width

    def bottom_crop(pil_img):
        w, h = pil_img.size
        top = max(0, h - cfg.train_height)
        return TF.crop(pil_img, top=top, left=0, height=cfg.train_height, width=w)

    return T.Compose([
        T.Resize((h_resize, w_resize), interpolation=T.InterpolationMode.BILINEAR),
        T.Lambda(bottom_crop),
        T.ToTensor(),
        T.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
    ])

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    # lấy cfg từ repo + args riêng
    args = get_args()
    # merge_config() đọc args trong sys.argv, nên “nhét” config vào tương thích:
    import sys
    sys.argv = [sys.argv[0], args.config, '--test_model', args.test_model, '--local_rank', str(args.local_rank)]
    args_cfg, cfg = merge_config()

    # batch 1 cho demo
    cfg.batch_size = 1
    dist_print('start video inference...')

    # model
    assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide','34fca']
    net = get_model(cfg)
    sd = torch.load(cfg.test_model, map_location='cpu')
    state = sd['model'] if 'model' in sd else sd
    # bỏ "module." nếu có
    new_state = {}
    for k,v in state.items():
        new_state[k.replace('module.', '')] = v
    net.load_state_dict(new_state, strict=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device).eval()

    # video io
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0  # mặc định
    out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # .avi
    vout = cv2.VideoWriter(args.out, fourcc, fps, (out_w, out_h))

    # transform cho model
    tfm = build_transform(cfg)

    # chạy từng frame
    frame_idx = 0
    with torch.no_grad():
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_idx += 1

            # Chuẩn bị input
            img_h, img_w = frame_bgr.shape[:2]
            # UFLD dùng RGB → PIL
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(frame_rgb)
            inp = tfm(pil).unsqueeze(0).to(device)

            # forward
            pred = net(inp)

            # toạ độ theo ảnh gốc
            coords = pred2coords(
                pred,
                cfg.row_anchor, cfg.col_anchor,
                original_image_width=img_w,
                original_image_height=img_h
            )

            # vẽ lên frame_bgr
            for lane in coords:
                for (x, y) in lane:
                    cv2.circle(frame_bgr, (int(x), int(y)), 4, (0, 255, 0), -1)

            vout.write(frame_bgr)

            if frame_idx % 50 == 0:
                print(f'Processed {frame_idx} frames...')

    cap.release()
    vout.release()
    print(f'Done. Saved to: {args.out}')
