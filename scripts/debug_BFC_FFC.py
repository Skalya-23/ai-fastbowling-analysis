"""
Debug script to understand why BFC/FFC detection is failing.
Shows exactly what the algorithm sees and where it's failing.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.signal import savgol_filter


def smooth_series(y, win=7):
    """Smooth a series to reduce jitter."""
    return pd.Series(y).rolling(window=win, center=True, min_periods=1).mean().to_numpy()


def analyze_contacts(y, fps=30, vel_thresh=3.5, min_len=2, ground_pct=70, name="Ankle"):
    """
    Analyze ankle data to find contact segments.
    Shows step-by-step what the algorithm is doing.
    """
    print(f"\n{'='*60}")
    print(f"ANALYZING {name}")
    print(f"{'='*60}\n")
    
    # Remove NaN values
    valid_mask = np.isfinite(y)
    if not valid_mask.any():
        print(f"ERROR: No valid data for {name}")
        return []
    
    # Smooth the data
    y_smooth = smooth_series(y, win=7)
    
    # Calculate velocity (change per frame)
    dy = np.gradient(y_smooth)
    speed = np.abs(dy)
    
    # Find "ground" threshold (high Y = more grounded)
    y_thresh = np.nanpercentile(y_smooth[valid_mask], ground_pct)
    
    # Check conditions
    speed_ok = speed < vel_thresh
    y_ok = y_smooth >= y_thresh
    combined = speed_ok & y_ok & valid_mask
    
    print(f"Data stats:")
    print(f"  Valid frames: {valid_mask.sum()} / {len(y)}")
    print(f"  Y range: {np.nanmin(y):.1f} to {np.nanmax(y):.1f} px")
    print(f"  Ground threshold (Y > {y_thresh:.1f}): {y_ok.sum()} frames")
    print(f"  Low speed (<{vel_thresh}px/frame): {speed_ok.sum()} frames")
    print(f"  BOTH conditions: {combined.sum()} frames\n")
    
    # Find contiguous segments
    segments = []
    i = 0
    n = len(combined)
    
    while i < n:
        if combined[i]:
            j = i
            while j + 1 < n and combined[j + 1]:
                j += 1
            if (j - i + 1) >= min_len:
                segments.append((i, j))
                duration_ms = (j - i + 1) / fps * 1000
                print(f"  Contact segment: frames {i}-{j} ({j-i+1} frames, {duration_ms:.0f}ms)")
                print(f"    Y range: {y_smooth[i]:.1f} to {y_smooth[j]:.1f}")
                print(f"    Avg speed: {speed[i:j+1].mean():.2f} px/frame")
            i = j + 1
        else:
            i += 1
    
    if not segments:
        print(f"  ❌ NO CONTACT SEGMENTS FOUND")
        print(f"\n  Diagnosis:")
        if combined.sum() > 0:
            print(f"    - Some frames meet criteria but not {min_len} consecutive")
            print(f"    → TRY: Reduce min_len to 1")
        elif speed_ok.sum() > 0 and y_ok.sum() == 0:
            print(f"    - Speed is OK but foot never reaches ground threshold")
            print(f"    → TRY: Increase ground_pct (e.g., 85 or 90)")
        elif y_ok.sum() > 0 and speed_ok.sum() == 0:
            print(f"    - Foot reaches ground but speed always too high")
            print(f"    → TRY: Increase vel_thresh (e.g., 5.0 or 6.0)")
        else:
            print(f"    - Foot never near ground AND always moving too fast")
            print(f"    → TRY: Increase BOTH thresholds significantly")
    else:
        print(f"  ✅ Found {len(segments)} contact segment(s)")
    
    return segments, y_smooth, speed, y_thresh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kpts_csv", required=True, help="Path to keypoints CSV")
    parser.add_argument("--release_frame", type=int, default=74, help="Release frame")
    parser.add_argument("--fps", type=float, default=30, help="Video FPS")
    parser.add_argument("--vel_thresh", type=float, default=3.5, help="Velocity threshold")
    parser.add_argument("--min_len", type=int, default=2, help="Minimum contact length")
    parser.add_argument("--ground_pct", type=float, default=70, help="Ground percentile")
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.kpts_csv)
    
    print(f"\n{'#'*60}")
    print(f"BFC/FFC DETECTION DIAGNOSTIC")
    print(f"{'#'*60}")
    print(f"\nVideo info:")
    print(f"  Frames: {len(df)}")
    print(f"  FPS: {args.fps}")
    print(f"  Release frame: {args.release_frame}")
    print(f"  Release time: {args.release_frame / args.fps:.3f}s")
    print(f"\nThresholds:")
    print(f"  vel_thresh: {args.vel_thresh} px/frame")
    print(f"  min_len: {args.min_len} frames")
    print(f"  ground_pct: {args.ground_pct}%")
    
    # Analyze right ankle (back foot for right-hander)
    right_segments, right_smooth, right_speed, right_thresh = analyze_contacts(
        df['rank_y'].values, 
        args.fps,
        args.vel_thresh,
        args.min_len,
        args.ground_pct,
        "RIGHT ANKLE (Back Foot)"
    )
    
    # Analyze left ankle (front foot for right-hander)
    left_segments, left_smooth, left_speed, left_thresh = analyze_contacts(
        df['lank_y'].values,
        args.fps,
        args.vel_thresh,
        args.min_len,
        args.ground_pct,
        "LEFT ANKLE (Front Foot)"
    )
    
    # Check for valid BFC/FFC pair
    print(f"\n{'='*60}")
    print(f"BFC/FFC PAIR ANALYSIS")
    print(f"{'='*60}\n")
    
    if not right_segments:
        print("❌ No BFC detected (no right ankle contacts)")
    if not left_segments:
        print("❌ No FFC detected (no left ankle contacts)")
    
    if right_segments and left_segments:
        print("Checking for valid pairs (BFC before FFC, gap 0.05-0.50s):\n")
        
        found_pair = False
        for i, (rb, re) in enumerate(right_segments):
            for j, (lb, le) in enumerate(left_segments):
                if lb > rb:  # FFC after BFC
                    gap_s = (lb - rb) / args.fps
                    if 0.05 <= gap_s <= 0.50:
                        print(f"  ✅ Valid pair found:")
                        print(f"     BFC: frames {rb}-{re} (t={rb/args.fps:.3f}s)")
                        print(f"     FFC: frames {lb}-{le} (t={lb/args.fps:.3f}s)")
                        print(f"     Gap: {gap_s:.3f}s")
                        print(f"     Before release: {args.release_frame - rb} frames")
                        found_pair = True
        
        if not found_pair:
            print("  ❌ No valid pairs (either wrong order or gap outside 0.05-0.50s)")
            print("\n  All right ankle contacts:")
            for rb, re in right_segments:
                print(f"    frames {rb}-{re} (t={rb/args.fps:.3f}s)")
            print("\n  All left ankle contacts:")
            for lb, le in left_segments:
                print(f"    frames {lb}-{le} (t={lb/args.fps:.3f}s)")
    
    # Create visualization
    print(f"\n{'='*60}")
    print(f"GENERATING VISUALIZATION")
    print(f"{'='*60}\n")
    
    fig, axes = plt.subplots(4, 1, figsize=(16, 12))
    frames = df['frame'].values
    
    # Plot 1: Raw ankle positions
    ax = axes[0]
    ax.plot(frames, df['rank_y'], 'b-', alpha=0.3, label='Right Ankle (raw)')
    ax.plot(frames, right_smooth, 'b-', linewidth=2, label='Right Ankle (smoothed)')
    ax.plot(frames, df['lank_y'], 'r-', alpha=0.3, label='Left Ankle (raw)')
    ax.plot(frames, left_smooth, 'r-', linewidth=2, label='Left Ankle (smoothed)')
    ax.axhline(y=right_thresh, color='b', linestyle='--', alpha=0.5, label=f'Right ground thresh ({right_thresh:.0f}px)')
    ax.axhline(y=left_thresh, color='r', linestyle='--', alpha=0.5, label=f'Left ground thresh ({left_thresh:.0f}px)')
    ax.axvline(x=args.release_frame, color='g', linestyle='--', linewidth=2, label='Release')
    ax.set_ylabel('Y Position (px)\n[Higher = More Grounded]')
    ax.set_title('Ankle Positions (Raw vs Smoothed)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Ankle velocities
    ax = axes[1]
    ax.plot(frames, right_speed, 'b-', linewidth=1.5, label='Right Ankle Speed')
    ax.plot(frames, left_speed, 'r-', linewidth=1.5, label='Left Ankle Speed')
    ax.axhline(y=args.vel_thresh, color='k', linestyle='--', label=f'Velocity threshold ({args.vel_thresh} px/frame)')
    ax.axvline(x=args.release_frame, color='g', linestyle='--', linewidth=2, label='Release')
    ax.set_ylabel('Speed (px/frame)')
    ax.set_title('Ankle Velocities (Must be BELOW threshold for contact)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, min(20, right_speed.max(), left_speed.max()))
    
    # Plot 3: Contact segments - Right ankle
    ax = axes[2]
    ax.plot(frames, right_smooth, 'b-', linewidth=2, label='Right Ankle')
    for (start, end) in right_segments:
        ax.axvspan(start, end, alpha=0.3, color='blue', label='Contact' if start == right_segments[0][0] else '')
        ax.plot(start, right_smooth[start], 'go', markersize=10, label='BFC candidate' if start == right_segments[0][0] else '')
    ax.axhline(y=right_thresh, color='b', linestyle='--', alpha=0.5)
    ax.axvline(x=args.release_frame, color='g', linestyle='--', linewidth=2, label='Release')
    ax.set_ylabel('Y Position (px)')
    ax.set_title('Right Ankle (Back Foot) - Detected Contacts')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Contact segments - Left ankle
    ax = axes[3]
    ax.plot(frames, left_smooth, 'r-', linewidth=2, label='Left Ankle')
    for (start, end) in left_segments:
        ax.axvspan(start, end, alpha=0.3, color='red', label='Contact' if start == left_segments[0][0] else '')
        ax.plot(start, left_smooth[start], 'go', markersize=10, label='FFC candidate' if start == left_segments[0][0] else '')
    ax.axhline(y=left_thresh, color='r', linestyle='--', alpha=0.5)
    ax.axvline(x=args.release_frame, color='g', linestyle='--', linewidth=2, label='Release')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Y Position (px)')
    ax.set_title('Left Ankle (Front Foot) - Detected Contacts')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = 'bfc_ffc_diagnostic.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved visualization: {output_file}")
    
    # Print recommendations
    print(f"\n{'='*60}")
    print(f"RECOMMENDATIONS")
    print(f"{'='*60}\n")
    
    if not right_segments and not left_segments:
        print("⚠️  NO CONTACTS DETECTED AT ALL")
        print("\nTry these settings:")
        print("  --vel_thresh 6.0      (allow more movement)")
        print("  --min_len 1           (accept brief contacts)")
        print("  --ground_pct 85       (lower ground threshold)")
        print("\nCommand:")
        print(f"python scripts/debug_bfc_ffc.py --kpts_csv {args.kpts_csv} "
              f"--release_frame {args.release_frame} --vel_thresh 6.0 --min_len 1 --ground_pct 85")
    
    elif not right_segments:
        print("⚠️  BFC NOT DETECTED (but FFC was)")
        print("\nRight ankle specific issue. Try:")
        print("  --vel_thresh 6.0")
        print("  --ground_pct 85")
    
    elif not left_segments:
        print("⚠️  FFC NOT DETECTED (but BFC was)")
        print("\nLeft ankle specific issue. Try:")
        print("  --vel_thresh 6.0")
        print("  --ground_pct 85")
    
    else:
        print("✅ Both BFC and FFC detected!")
        print("\nUpdate event_detect.py with these thresholds:")
        print(f"  vel_thresh={args.vel_thresh}")
        print(f"  min_len={args.min_len}")
        print(f"  ground_pct={args.ground_pct}")


if __name__ == "__main__":
    main()