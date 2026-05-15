#!/usr/bin/env python3
"""
生成 metadata_cache.json，只包含图片尺寸，不包含 caption。

用法:
    python create_metadata_cache.py --image_dir /path/to/images
    python create_metadata_cache.py --image_dir /path/to/images --output /path/to/metadata_cache.json
"""

import argparse
import json
import os
import sys

import imagesize
from PIL import Image
from tqdm import tqdm

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def get_image_size(image_path: str):
    """
    获取图片尺寸 (width, height)。
    优先使用 imagesize 库，失败时用 PIL 作为 fallback。
    与 train_util.py 中 get_image_size() 实现一致。
    """
    image_size = imagesize.get(image_path)
    if image_size[0] <= 0:
        # imagesize 对部分图片不生效，回退到 PIL
        try:
            with Image.open(image_path) as img:
                image_size = img.size
        except Exception:
            print(f"Warning: failed to get image size: {image_path}", file=sys.stderr)
            image_size = (0, 0)
    return image_size


def collect_images(image_dir: str):
    """递归扫描 image_dir，收集所有支持的图片文件路径。"""
    image_paths = []
    for root, _, files in os.walk(image_dir):
        for filename in files:
            _, ext = os.path.splitext(filename)
            if ext.lower() in SUPPORTED_EXTENSIONS:
                image_paths.append(os.path.join(root, filename))
    return image_paths


def main():
    parser = argparse.ArgumentParser(
        description="生成 metadata_cache.json（仅包含图片尺寸）"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="图片目录路径（必填）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出 JSON 文件路径（默认: image_dir 下的 metadata_cache.json）",
    )
    args = parser.parse_args()

    image_dir = os.path.abspath(args.image_dir)
    if not os.path.isdir(image_dir):
        print(f"Error: image_dir does not exist or is not a directory: {image_dir}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output
    if output_path is None:
        output_path = os.path.join(image_dir, "metadata_cache.json")

    # 确保 output 的父目录存在
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    # 收集所有图片
    print(f"Scanning images in: {image_dir}")
    image_paths = collect_images(image_dir)
    print(f"Found {len(image_paths)} images")

    if len(image_paths) == 0:
        print("No supported images found, writing empty cache.")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2)
        print(f"Empty metadata_cache saved to: {output_path}")
        return

    # 获取图片尺寸
    metadata = {}
    for full_path in tqdm(image_paths, desc="Getting image sizes", unit="img"):
        # 使用相对于 image_dir 的路径作为 key，统一使用 / 分隔符
        rel_path = os.path.relpath(full_path, image_dir).replace(os.sep, "/")
        width, height = get_image_size(full_path)
        metadata[rel_path] = {"resolution": [width, height]}

    # 写入 JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Metadata cache saved to: {output_path}")
    print(f"Total entries: {len(metadata)}")


if __name__ == "__main__":
    main()
