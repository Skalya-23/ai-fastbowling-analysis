# save as: scripts/check_stride.py
import os
import math
import argparse
import pandas as pd

def resolve_repo_root():
    # .../Cricket-YOLO-Pose/scripts -> .../Cricket-YOLO-Pose
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def resolve_csv_path(kpts_path: str) -> str:
    # If absolute or exists as given, use it. Otherwise, treat as repo-root relative.
    if os.path.isabs(kpts_path) and os.path.exists(kpts_path):
        return kpts_path
    if os.path.exists(kpts_path):
        return kpts_path
    repo_root = resolve_repo_root()
    candidate = os.path.join(repo_root, kpts_path)
    return candidate

def need_cols(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected columns: {missing}")

def get_frame_row(df, frame_num: int):
    sel = df[df["frame"] == frame_num]
    if sel.empty:
        raise IndexError(f"No row found for frame == {frame_num}. "
                         f"Available frame range: [{int(df['frame'].min())}, {int(df['frame'].max())}]")
    return sel.iloc[0]

def main():
    parser = argparse.ArgumentParser(description="Stride sanity check (BFC→FFC).")
    parser.add_argument("--kpts", default="data/videos/James2_kpts.csv",
                        help="Path to keypoints CSV (repo-root relative or absolute).")
    parser.add_argument("--bfc", type=int, default=66, help="BFC frame index.")
    parser.add_argument("--ffc", type=int, default=72, help="FFC frame index.")
    parser.add_argument("--fps", type=float, default=30.0, help="Video FPS.")
    parser.add_argument("--m_per_px", type=float, default=0.00363, help="Scale: meters per pixel.")
    parser.add_argument("--handed", choices=["right","left"], default="right",
                        help="Bowling handedness (affects which ankle is back/front).")
    args = parser.parse_args()

    csv_path = resolve_csv_path(args.kpts)
    print(f"Loading: {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Required columns
    need_cols(df, [
        "frame",
        "rank_x","rank_y","lank_x","lank_y",
        "rhip_x","rhip_y","lhip_x","lhip_y"
    ])

    bfc = get_frame_row(df, args.bfc)
    ffc = get_frame_row(df, args.ffc)

    print(f"\nBFC (frame {args.bfc}):")
    print(f"  Right ankle: ({bfc['rank_x']:.1f}, {bfc['rank_y']:.1f})")
    print(f"  Left  ankle: ({bfc['lank_x']:.1f}, {bfc['lank_y']:.1f})")

    print(f"\nFFC (frame {args.ffc}):")
    print(f"  Right ankle: ({ffc['rank_x']:.1f}, {ffc['rank_y']:.1f})")
    print(f"  Left  ankle: ({ffc['lank_x']:.1f}, {ffc['lank_y']:.1f})")

    # Distances at each event
    def dist(ax, ay, bx, by):
        return math.hypot(bx - ax, by - ay)

    right_to_left_bfc = dist(bfc["rank_x"], bfc["rank_y"], bfc["lank_x"], bfc["lank_y"])
    right_to_left_ffc = dist(ffc["rank_x"], ffc["rank_y"], ffc["lank_x"], ffc["lank_y"])

    print(f"\nDistance between ankles:")
    print(f"  At BFC: {right_to_left_bfc:.1f} px")
    print(f"  At FFC: {right_to_left_ffc:.1f} px")

    # Stride logic: for right-handed, back=right @BFC; front=left @FFC (swap if left-handed)
    if args.handed == "right":
        back_ankle_BFC = (bfc["rank_x"], bfc["rank_y"])
        front_ankle_FFC = (ffc["lank_x"], ffc["lank_y"])
    else:
        back_ankle_BFC = (bfc["lank_x"], bfc["lank_y"])
        front_ankle_FFC = (ffc["rank_x"], ffc["rank_y"])

    stride_px = dist(back_ankle_BFC[0], back_ankle_BFC[1], front_ankle_FFC[0], front_ankle_FFC[1])
    stride_m = stride_px * args.m_per_px

    print(f"\nStride length ({'right' if args.handed=='right' else 'left'}@BFC → "
          f"{'left' if args.handed=='right' else 'right'}@FFC):")
    print(f"  {stride_px:.1f} px = {stride_m:.3f} m  (m/px={args.m_per_px})")

    # Hip center displacement → run-up speed over (FFC - BFC)
    hip_bfc_x = (bfc["lhip_x"] + bfc["rhip_x"]) / 2.0
    hip_ffc_x = (ffc["lhip_x"] + ffc["rhip_x"]) / 2.0
    hip_dist_px = abs(hip_ffc_x - hip_bfc_x)
    hip_dist_m = hip_dist_px * args.m_per_px

    frame_gap = abs(args.ffc - args.bfc)
    dt = frame_gap / args.fps if args.fps > 0 else float("nan")

    print("\n=== HIP POSITIONS (for run-up speed) ===")
    print(f"Hip center at BFC: {hip_bfc_x:.1f}")
    print(f"Hip center at FFC: {hip_ffc_x:.1f}")
    print(f"Hip displacement: {hip_dist_px:.1f} px = {hip_dist_m:.3f} m")
    print(f"Time gap: {frame_gap} frames @ {args.fps:g} fps = {dt:.3f} s")
    if dt > 0:
        v = hip_dist_m / dt
        print(f"Run-up speed: {v:.2f} m/s = {v*3.6:.2f} km/h")
    else:
        print("Run-up speed: N/A (fps or frame gap invalid)")

if __name__ == "__main__":
    main()
