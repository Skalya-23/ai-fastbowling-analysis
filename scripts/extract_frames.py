import cv2, os, glob

def extract_frames(video_path, outdir, step=2):
    os.makedirs(outdir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    i = 0
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % step == 0:  # save every Nth frame
            fname = os.path.join(outdir, f"{os.path.splitext(os.path.basename(video_path))[0]}_{frame_id:05d}.jpg")
            cv2.imwrite(fname, frame)
            frame_id += 1
        i += 1
    cap.release()
    print(f"[DONE] {frame_id} frames saved to {outdir}")

if __name__ == "__main__":
    videos = glob.glob("data/videos/New/*.mp4")
    for v in videos:
        outdir = "datasets/cricket_ball/frames"
        extract_frames(v, outdir, step=1)  # adjust step (1=all frames, 2=half, 5=every 5th)
