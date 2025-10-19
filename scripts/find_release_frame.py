"""
Helper script to find the correct release frame visually.
Shows video frame-by-frame around the expected release time.
"""

import cv2
import argparse
import sys

def find_release_frame(video_path, start_time=2.0, fps=30):
    """
    Display frames around release time to identify exact release frame.
    
    Args:
        video_path: Path to video
        start_time: Approximate release time in seconds
        fps: Frames per second
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {video_path}")
    print(f"FPS: {actual_fps}")
    print(f"Total frames: {total_frames}")
    print(f"Starting search around {start_time}s")
    print("\nControls:")
    print("  SPACE: Mark this frame as release")
    print("  RIGHT ARROW: Next frame")
    print("  LEFT ARROW: Previous frame")
    print("  Q: Quit")
    print("-" * 50)
    
    # Start a few frames before expected release
    start_frame = int(start_time * actual_fps) - 10
    start_frame = max(0, start_frame)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame
    
    marked_frame = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video reached")
            break
        
        # Create display
        display = frame.copy()
        h, w = display.shape[:2]
        
        # Add frame info
        time_s = current_frame / actual_fps
        text = f"Frame: {current_frame} | Time: {time_s:.3f}s"
        
        if marked_frame is not None:
            text += f" | MARKED: {marked_frame}"
            cv2.rectangle(display, (0, 0), (w, h), (0, 255, 0), 10)
        
        cv2.putText(display, text, (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        
        # Instructions
        cv2.putText(display, "SPACE=Mark | ARROWS=Navigate | Q=Quit", 
                    (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Show
        cv2.imshow("Find Release Frame", display)
        
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q') or key == 27:  # Q or ESC
            break
        elif key == ord(' '):  # SPACE
            marked_frame = current_frame
            marked_time = current_frame / actual_fps
            print(f"\n✓ MARKED: Frame {marked_frame} (t={marked_time:.3f}s)")
            print(f"\nUse this in your command:")
            print(f"  --manual_release_frame {marked_frame}")
        elif key == 83 or key == ord('d'):  # RIGHT ARROW or D
            current_frame += 1
            if current_frame < total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            else:
                print("End of video")
                current_frame -= 1
        elif key == 81 or key == ord('a'):  # LEFT ARROW or A
            current_frame = max(0, current_frame - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        elif key == ord('j'):  # J - jump forward 5 frames
            current_frame = min(total_frames - 1, current_frame + 5)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        elif key == ord('k'):  # K - jump back 5 frames
            current_frame = max(0, current_frame - 5)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    
    cap.release()
    cv2.destroyAllWindows()
    
    if marked_frame is not None:
        print(f"\n{'='*50}")
        print(f"RELEASE FRAME: {marked_frame}")
        print(f"RELEASE TIME: {marked_frame / actual_fps:.3f}s")
        print(f"{'='*50}")
        print("\nNext steps:")
        print("1. Copy the frame number above")
        print("2. Run combined_pipeline.py with:")
        print(f"   --manual_release_frame {marked_frame}")
        print(f"   --use_hybrid_tracking")
        return marked_frame
    else:
        print("\nNo frame marked. Run again to find release frame.")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find exact release frame in bowling video")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--start_time", type=float, default=2.0, 
                        help="Approximate release time in seconds (default: 2.0)")
    parser.add_argument("--fps", type=float, default=30, 
                        help="Video FPS (default: 30)")
    
    args = parser.parse_args()
    
    release_frame = find_release_frame(args.video, args.start_time, args.fps)
    
    if release_frame:
        sys.exit(0)
    else:
        sys.exit(1)