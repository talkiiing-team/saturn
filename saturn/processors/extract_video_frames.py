import cv2
import numpy as np
from numpy.typing import ArrayLike
from concurrent.futures import ThreadPoolExecutor


def extract_video_frames(
    video_path: str,
    resize_frames=(224, 224),
    n=10,
    interval=30,
) -> ArrayLike:
    frames = select_top_n_from_equal_intervals(
        video_path=video_path, n=n, interval=interval
    )

    if resize_frames is not None:
        frames = [cv2.resize(x, resize_frames) for x in frames]

    return frames

def calculate_movement(prev_gray, gray):
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return np.sum(magnitude)

def process_frame(frame, prev_gray):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_gray is not None:
        movement_score = calculate_movement(prev_gray, gray)
    else:
        movement_score = 0

    return gray, movement_score

def select_top_n_from_equal_intervals(video_path, n=10, interval=30):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_movement = []
    prev_gray = None
    frame_count = 0

    with ThreadPoolExecutor() as executor:
        futures = []
        while cap.isOpened() and frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % interval == 0:
                futures.append(executor.submit(process_frame, frame, prev_gray))
                frame_count += 1
            else:
                frame_count += 1

        for future in futures:
            gray, movement_score = future.result()
            if movement_score is not None:
                frame_movement.append((movement_score, frame, frame_count))

            # Update prev_gray for the next iteration
            prev_gray = gray

    frame_movement.sort(reverse=True, key=lambda x: x[0])
    top_n_frames = [frame for _, frame, _ in frame_movement[:n]]

    cap.release()
    return top_n_frames

