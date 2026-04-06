"""
model_downloader.py
Downloads YOLOv3 model files if not already present.
"""

import os
import urllib.request


YOLOV3_FILES = {
    "yolov3.cfg": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
    "yolov3.weights": "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov3.weights",
    "coco.names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
}

MIN_FILE_SIZES = {
    "yolov3.cfg": 7000,
    "yolov3.weights": 220 * 1024 * 1024,
    "coco.names": 100,
}


def _download(url: str, dest: str):
    print(f"[Downloader] Downloading {os.path.basename(dest)} …")
    print(f"[Downloader] URL: {url}")
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0",
        "Accept": "*/*",
    })
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            total_size = int(response.getheader('Content-Length', 0))
            downloaded = 0
            with open(dest, "wb") as out_file:
                while True:
                    chunk = response.read(65536)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        mb_downloaded = downloaded / 1024 / 1024
                        print(f"[Downloader] {os.path.basename(dest)}: {percent:.1f}% ({mb_downloaded:.1f} MB)")
    except Exception as e:
        if os.path.exists(dest):
            os.remove(dest)
        raise e
    print(f"[Downloader] Saved to {dest}")


def _normalize_names_file(model_dir: str, names: str) -> str:
    """Rename a legacy coco.name file to coco.names if needed."""
    old_path = os.path.join(model_dir, "coco.name")
    if os.path.exists(old_path) and not os.path.exists(names):
        os.rename(old_path, names)
        print(f"[Downloader] Renamed legacy coco.name to {os.path.basename(names)}")
    return names


def _file_is_valid(path: str) -> bool:
    if not os.path.exists(path):
        return False
    key = os.path.basename(path)
    min_size = MIN_FILE_SIZES.get(key)
    if min_size is None:
        return True
    size = os.path.getsize(path)
    if size < min_size:
        print(f"[Downloader] Invalid {key}: size {size} < expected {min_size}. Redownloading.")
        return False
    return True


def ensure_models_downloaded(model_dir: str, cfg: str, weights: str, names: str) -> bool:
    """
    Checks for model files; downloads any that are missing.
    Returns True if all files are present after the attempt.
    """
    os.makedirs(model_dir, exist_ok=True)
    names = _normalize_names_file(model_dir, names)
    file_map = {
        cfg: YOLOV3_FILES["yolov3.cfg"],
        weights: YOLOV3_FILES["yolov3.weights"],
        names: YOLOV3_FILES["coco.names"],
    }

    for dest, url in file_map.items():
        if not _file_is_valid(dest):
            try:
                if os.path.exists(dest):
                    os.remove(dest)
                _download(url, dest)
            except Exception as e:
                print(f"[Downloader] [ERROR] Failed to download {url}: {e}")
                return False
        else:
            print(f"[Downloader] Found valid {os.path.basename(dest)}")

    return all(_file_is_valid(p) for p in file_map)
