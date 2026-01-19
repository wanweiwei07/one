"""
Utility to convert video files to GIF format.
Supports common video formats (mp4, avi, mov, etc.)
"""
import os
import sys
from pathlib import Path

try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("Error: PIL (Pillow) is required. Install with: pip install Pillow")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("Error: opencv-python is required. Install with: pip install opencv-python")
    sys.exit(1)


def video_to_gif(video_path, output_path=None, fps=10, scale=1.0, 
                 start_time=None, duration=None, optimize=True):
    """
    Convert video file to GIF.
    :param video_path: path to input video file
    :param output_path: path to output GIF file (default: same name as video with .gif extension)
    :param fps: frames per second for output GIF (default: 10)
    :param scale: scale factor for output size (default: 1.0, range: 0.1-1.0)
    :param start_time: start time in seconds (default: None, start from beginning)
    :param duration: duration in seconds (default: None, entire video)
    :param optimize: optimize GIF file size (default: True)
    :return: path to created GIF file
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if output_path is None:
        output_path = Path(video_path).with_suffix('.gif')
    else:
        output_path = Path(output_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(video_fps / fps))
    start_frame = int(start_time * video_fps) if start_time else 0
    if duration:
        end_frame = start_frame + int(duration * video_fps)
    else:
        end_frame = total_frames
    end_frame = min(end_frame, total_frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    frame_count = start_frame
    print(f"Processing video: {video_path}")
    print(f"  Video FPS: {video_fps:.2f}, Output FPS: {fps}")
    print(f"  Frame range: {start_frame}-{end_frame} (every {frame_interval} frames)")
    while frame_count < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if (frame_count - start_frame) % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if scale != 1.0:
                new_width = int(frame_rgb.shape[1] * scale)
                new_height = int(frame_rgb.shape[0] * scale)
                frame_rgb = cv2.resize(frame_rgb, (new_width, new_height), 
                                      interpolation=cv2.INTER_AREA)
            frames.append(Image.fromarray(frame_rgb))
        frame_count += 1
    cap.release()
    if not frames:
        raise RuntimeError("No frames extracted from video")
    print(f"  Extracted {len(frames)} frames")
    print(f"  Output size: {frames[0].width}x{frames[0].height}")
    print(f"  Writing GIF to: {output_path}")
    frame_duration = int(1000 / fps)
    frames[0].save(output_path, save_all=True, append_images=frames[1:],
                   duration=frame_duration, loop=0, optimize=optimize)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Done! File size: {file_size_mb:.2f} MB")
    return str(output_path)


def main():
    """Command-line interface for video to GIF conversion."""
    import argparse
    parser = argparse.ArgumentParser(description="Convert video files to GIF format")
    parser.add_argument("video", help="Input video file path")
    parser.add_argument("-o", "--output", help="Output GIF file path (default: same name as video)")
    parser.add_argument("-f", "--fps", type=int, default=10, 
                       help="Output FPS (default: 10)")
    parser.add_argument("-s", "--scale", type=float, default=1.0,
                       help="Scale factor for output size (default: 1.0, range: 0.1-1.0)")
    parser.add_argument("-t", "--start", type=float, default=None,
                       help="Start time in seconds (default: 0)")
    parser.add_argument("-d", "--duration", type=float, default=None,
                       help="Duration in seconds (default: entire video)")
    parser.add_argument("--no-optimize", action="store_true",
                       help="Disable GIF optimization (faster but larger file)")
    args = parser.parse_args()
    if args.scale <= 0 or args.scale > 1.0:
        print("Error: scale must be between 0.1 and 1.0")
        sys.exit(1)
    try:
        output_path = video_to_gif(args.video, args.output, args.fps, args.scale,
                                   args.start, args.duration, not args.no_optimize)
        print(f"\nSuccess! GIF created: {output_path}")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        print("Video to GIF Converter")
        print("=" * 50)
        print("\nUsage examples:")
        print("  python test_video_gif.py video.mp4")
        print("  python test_video_gif.py video.mp4 -o output.gif")
        print("  python test_video_gif.py video.mp4 -f 15 -s 0.5")
        print("  python test_video_gif.py video.mp4 -t 5 -d 10")
        print("\nFor more options, run: python test_video_gif.py --help")
        print("\nDemo: Converting a sample video (if available)...")
        sample_videos = list(Path('.').glob('*.mp4')) + list(Path('.').glob('*.avi'))
        if sample_videos:
            sample = sample_videos[0]
            print(f"\nFound sample video: {sample}")
            print("Converting first 3 seconds to demo.gif...")
            try:
                video_to_gif(str(sample), "demo.gif", fps=10, scale=0.5, 
                           start_time=0, duration=3)
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("\nNo sample videos found. Place a .mp4 or .avi file in the current directory.")
