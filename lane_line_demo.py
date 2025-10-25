import os, cv2, torch
import argparse
import numpy as np
from collections import deque, defaultdict
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from utils.common import merge_config, get_model
from utils.dist_utils import dist_print

# ========================= UFLDv2: logits -> coords =========================
def pred2coords(pred, row_anchor, col_anchor, local_width=1,
                original_image_width=1640, original_image_height=590):
    b, num_grid_row, num_cls_row, num_lane_row = pred['loc_row'].shape
    b, num_grid_col, num_cls_col, num_lane_col = pred['loc_col'].shape

    max_indices_row = pred['loc_row'].argmax(1).cpu()
    valid_row       = pred['exist_row'].argmax(1).cpu()

    max_indices_col = pred['loc_col'].argmax(1).cpu()
    valid_col       = pred['exist_col'].argmax(1).cpu()

    pred['loc_row'] = pred['loc_row'].cpu()
    pred['loc_col'] = pred['loc_col'].cpu()

    coords = []
    row_lane_idx = [1, 2]
    col_lane_idx = [0, 3]

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
            if tmp:
                coords.append(tmp)

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
            if tmp:
                coords.append(tmp)

    return coords

# ========================= Feature extract for MLP =========================
def patch_presence(bgr, cx, cy, patch_half=6, canny=(50,150),
                   white_v=200, white_s=40,
                   yellow_h_low=15, yellow_h_high=40, yellow_s=80, yellow_v=120,
                   edge_thr=0.10, paint_thr=0.08) -> int:
    h,w = bgr.shape[:2]
    x0,x1 = max(0,int(cx)-patch_half), min(w,int(cx)+patch_half+1)
    y0,y1 = max(0,int(cy)-patch_half), min(h,int(cy)+patch_half+1)
    if x1<=x0 or y1<=y0: 
        return 0
    patch = bgr[y0:y1, x0:x1]
    gray  = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny[0], canny[1])
    edge_ratio = (edges>0).mean()

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    H,S,V = cv2.split(hsv)
    white  = (V>=white_v) & (S<=white_s)
    yellow = (H>=yellow_h_low) & (H<=yellow_h_high) & (S>=yellow_s) & (V>=yellow_v)
    paint_ratio = (white|yellow).mean()
    return 1 if (edge_ratio>edge_thr or paint_ratio>paint_thr) else 0

def bits_from_lane(bgr, lane_xy, stride=1, **kw):
    if not lane_xy:
        return np.zeros((0,), dtype=np.uint8)
    return np.array([patch_presence(bgr, x, y, **kw) for (x,y) in lane_xy[::max(1,stride)]],
                    dtype=np.uint8)

def rle_fft_feats(bits: np.ndarray):
    N = len(bits)
    if N == 0:
        return dict(active_ratio=0.0, mean_run=0.0, std_run=0.0, gap_ratio=0.0, mean_gap=0.0, fft_band=0.0)
    runs1, runs0 = [], []
    cur, cnt = bits[0], 0
    for v in bits:
        if v == cur: cnt += 1
        else:
            (runs1 if cur==1 else runs0).append(cnt)
            cur, cnt = v, 1
    (runs1 if cur==1 else runs0).append(cnt)
    active_ratio = bits.mean()
    mean_run = float(np.mean(runs1)) if runs1 else 0.0
    std_run  = float(np.std(runs1))  if runs1 else 0.0
    gap_ratio = (np.sum(runs0)/N) if runs0 else 0.0
    mean_gap  = float(np.mean(runs0)) if runs0 else 0.0
    if N >= 8:
        z = bits.astype(np.float32) - active_ratio
        fr = np.fft.rfft(z)
        psd = (fr*np.conj(fr)).real
        lo, hi = max(1,int(0.1*len(psd))), max(2,int(0.4*len(psd)))
        fft_band = float(psd[lo:hi].sum())
    else:
        fft_band = 0.0
    return dict(active_ratio=active_ratio, mean_run=mean_run, std_run=std_run,
                gap_ratio=gap_ratio, mean_gap=mean_gap, fft_band=fft_band)

def geom_feats(poly):
    if len(poly) < 3:
        return dict(n=len(poly), y_span=0.0, x_span=0.0, mean_dx=0.0, std_dx=0.0, mean_abs_d2x=0.0)
    arr = np.array(poly, dtype=np.float32)
    xs, ys = arr[:,0], arr[:,1]
    dx = np.diff(xs)
    d2x = np.diff(dx)
    return dict(
        n=len(poly),
        y_span=float(ys.max()-ys.min()),
        x_span=float(xs.max()-xs.min()),
        mean_dx=float(dx.mean()),
        std_dx =float(dx.std()),
        mean_abs_d2x=float(np.abs(d2x).mean()) if len(d2x)>0 else 0.0
    )

def extract_feature_vector_from_lane(bgr, lane_xy):
    bits = bits_from_lane(
        bgr, lane_xy, stride=1,
        patch_half=6, canny=(50,150),
        white_v=200, white_s=40,
        yellow_h_low=15, yellow_h_high=40, yellow_s=80, yellow_v=120,
        edge_thr=0.10, paint_thr=0.08
    )
    f1 = rle_fft_feats(bits)
    f2 = geom_feats(lane_xy)
    feats = np.array([
        f1["active_ratio"], f1["mean_run"], f1["std_run"], f1["gap_ratio"], f1["mean_gap"], f1["fft_band"],
        f2["n"], f2["y_span"], f2["x_span"], f2["mean_dx"], f2["std_dx"], f2["mean_abs_d2x"]
    ], dtype=np.float32)
    return feats

def apply_standardizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (X - mean) / (std + 1e-6)

# ========================= MLP Definition & Loader =========================
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim=12, hid1=64, hid2=32, out_dim=2, p_drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid1), nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hid1, hid2), nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hid2, out_dim)
        )
    def forward(self, x): 
        return self.net(x)

# ========================= Args & Transform =========================
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('config', help='path to config file (ví dụ: configs/culane_res34.py)')
    p.add_argument('--test_model', required=True, help='đường dẫn epXXX.pth bạn muốn dùng để infer')
    p.add_argument('--video', required=True, help='đường dẫn video đầu vào (mp4/avi...)')
    p.add_argument('--out', default='out.avi', help='đường dẫn video xuất (mặc định out.avi)')
    p.add_argument('--local_rank', type=int, default=0)

    # === NEW: MLP & drawing args ===
    p.add_argument('--mlp_ckpt', required=True, help='đường dẫn MLP checkpoint .pt (Cell 7)')
    p.add_argument('--std_npz',  required=True, help='đường dẫn standardizer.npz (mean/std)')
    p.add_argument('--draw', choices=['line','dot'], default='line', help='kiểu vẽ lane đã phân loại')
    p.add_argument('--thickness', type=int, default=3, help='độ dày polyline khi --draw line')
    p.add_argument('--dot_radius', type=int, default=3, help='bán kính chấm khi --draw dot')
    p.add_argument('--smooth_win', type=int, default=1, help='cửa sổ smoothing (majority vote); 1 = tắt')
    return p.parse_args()

def build_transform(cfg):
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

# ========================= Drawing Helpers =========================
COLOR_SOLID  = (0,   0, 255)  # đỏ (BGR)
COLOR_DASHED = (255, 0,   0)  # xanh dương (BGR)

def draw_lane(frame_bgr, lane, label, mode='line', thickness=3, dot_radius=3):
    color = COLOR_DASHED if label==1 else COLOR_SOLID
    if not lane:
        return
    if mode == 'line':
        pts = np.array([(int(x), int(y)) for (x,y) in lane], dtype=np.int32).reshape((-1,1,2))
        cv2.polylines(frame_bgr, [pts], isClosed=False, color=color, thickness=thickness)
    else:
        for (x,y) in lane:
            cv2.circle(frame_bgr, (int(x), int(y)), dot_radius, color, -1, lineType=cv2.LINE_AA)
    x0,y0 = lane[0]
    tag = 'dashed' if label==1 else 'solid'
    cv2.putText(frame_bgr, tag, (int(x0)+6, int(y0)-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

# ========================= Main =========================
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args = get_args()

    # merge_config() dùng sys.argv → inject tương thích
    import sys
    sys.argv = [sys.argv[0], args.config, '--test_model', args.test_model, '--local_rank', str(args.local_rank)]
    args_cfg, cfg = merge_config()

    cfg.batch_size = 1
    dist_print('start video inference...')

    # UFLDv2 backbone
    assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide','34fca']
    net = get_model(cfg)
    sd = torch.load(args.test_model, map_location='cpu')
    state = sd['model'] if 'model' in sd else sd
    new_state = { k.replace('module.', ''): v for k,v in state.items() }
    net.load_state_dict(new_state, strict=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device).eval()

    # Load MLP + standardizer
    ckpt = torch.load(args.mlp_ckpt, map_location=device)
    in_dim  = ckpt.get('in_dim', 12)
    hid1    = ckpt.get('hid1', 64)
    hid2    = ckpt.get('hid2', 32)
    out_dim = ckpt.get('out_dim', 2)
    mlp = MLP(in_dim=in_dim, hid1=hid1, hid2=hid2, out_dim=out_dim).to(device)
    mlp.load_state_dict(ckpt['model_state'])
    mlp.eval()
    st = np.load(args.std_npz)
    mean_loaded = st['mean'].astype(np.float32)
    std_loaded  = st['std'].astype(np.float32)

    # video io
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    vout = cv2.VideoWriter(args.out, fourcc, fps, (out_w, out_h))

    tfm = build_transform(cfg)

    # optional temporal smoothing (per-lane-index buffer)
    smooth_win = max(1, int(args.smooth_win))
    buffers = defaultdict(lambda: deque(maxlen=smooth_win))

    def classify_lane(frame_bgr, lane_xy):
        feats = extract_feature_vector_from_lane(frame_bgr, lane_xy)
        xn = apply_standardizer(feats[None,:], mean_loaded, std_loaded)
        with torch.no_grad():
            logits = mlp(torch.from_numpy(xn).float().to(device))
            pred = int(torch.argmax(logits, dim=1).item())
        return pred  # 0=solid, 1=dashed

    frame_idx = 0
    with torch.no_grad():
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_idx += 1

            img_h, img_w = frame_bgr.shape[:2]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(frame_rgb)
            inp = tfm(pil).unsqueeze(0).to(device)

            pred = net(inp)

            coords = pred2coords(
                pred,
                cfg.row_anchor, cfg.col_anchor,
                original_image_width=img_w,
                original_image_height=img_h
            )

            # Phân loại & vẽ
            for i, lane in enumerate(coords):
                if not lane:
                    continue
                yhat = classify_lane(frame_bgr, lane)
                if smooth_win > 1:
                    buffers[i].append(yhat)
                    # majority vote
                    vals = list(buffers[i])
                    yhat = int(round(np.mean(vals))) if vals else yhat
                draw_lane(frame_bgr, lane, yhat, mode=args.draw, thickness=args.thickness, dot_radius=args.dot_radius)

            vout.write(frame_bgr)

            if frame_idx % 50 == 0:
                print(f'Processed {frame_idx} frames...')

    cap.release()
    vout.release()
    print(f'Done. Saved to: {args.out}')
