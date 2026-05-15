#!/usr/bin/env python3
"""
将 captions 转换为 Hugging Face datasets (基于 PyArrow/parquet) 格式，
支持高效的内存映射读取。

用法:
    # JSON 模式
    python create_caption_dataset.py --caption_json captions.json --output ./hf_dataset

    # 目录模式
    python create_caption_dataset.py --image_dir /path/to/images --output ./hf_dataset
    python create_caption_dataset.py --image_dir /path/to/images --caption_extension .txt --output ./hf_dataset
"""

import argparse
import glob
import json
import os
import sys

from datasets import Dataset
from tqdm import tqdm

# 与 train_util.py:104 保持一致的图片扩展名
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp",
                    ".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP"}


def collect_images(image_dir: str):
    """递归扫描 image_dir，收集所有支持的图片文件路径。"""
    image_paths = []
    for root, _, files in os.walk(image_dir):
        for filename in files:
            _, ext = os.path.splitext(filename)
            if ext in IMAGE_EXTENSIONS:
                image_paths.append(os.path.join(root, filename))
    return image_paths


def read_caption_file(caption_path: str):
    """
    读取一个 caption 文件，每行作为一个独立的 caption。
    跳过空行。

    参考 train_util.py:2001-2025 中的 read_caption 逻辑。
    """
    captions = []
    try:
        with open(caption_path, "rt", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    captions.append(stripped)
    except UnicodeDecodeError as e:
        print(f"Error: illegal char in file (not UTF-8): {caption_path}", file=sys.stderr)
        raise e
    return captions


def collect_captions_for_image(image_path: str, caption_extension: str):
    """
    为单个图片收集所有关联的 caption 内容。

    扫描模式: {basename}{caption_extension}*
    即匹配所有以图片基础名+caption后缀开头的文件。
    每个匹配到的文件中，每行非空内容作为一个独立的 caption。

    返回 list of str (可能为空)。
    """
    base_name = os.path.splitext(image_path)[0]
    pattern = base_name + caption_extension + "*"
    caption_files = sorted(glob.glob(pattern))

    all_captions = []
    for cap_file in caption_files:
        captions = read_caption_file(cap_file)
        all_captions.extend(captions)

    return all_captions


def process_directory_mode(image_dir: str, caption_extension: str):
    """
    目录模式：
    1. 递归扫描 image_dir 中所有支持的图片
    2. 为每张图片读取关联的 caption 文件
    3. 返回 (image_keys, captions_list) 两个列表
    """
    image_dir = os.path.abspath(image_dir)
    if not os.path.isdir(image_dir):
        print(f"Error: image_dir does not exist or is not a directory: {image_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning images in: {image_dir}")
    image_paths = collect_images(image_dir)
    print(f"Found {len(image_paths)} images")

    image_keys = []
    captions_list = []
    skipped_no_caption = 0

    for full_path in tqdm(image_paths, desc="Reading captions", unit="img"):
        caps = collect_captions_for_image(full_path, caption_extension)
        if not caps:
            skipped_no_caption += 1
            continue

        # 使用相对于 image_dir 的路径作为 key，统一使用 / 分隔符
        # 与 create_metadata_cache.py 保持一致
        rel_path = os.path.relpath(full_path, image_dir).replace(os.sep, "/")
        image_keys.append(rel_path)
        captions_list.append(caps)

    if skipped_no_caption > 0:
        print(f"Skipped {skipped_no_caption} images without caption files")

    return image_keys, captions_list


def process_json_mode(json_path: str):
    """
    JSON 模式：
    读取 JSON 文件，格式为 {"image_key": ["caption1", "caption2", ...], ...}
    返回 (image_keys, captions_list) 两个列表。
    """
    json_path = os.path.abspath(json_path)
    if not os.path.isfile(json_path):
        print(f"Error: caption_json does not exist: {json_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading captions from: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_keys = []
    captions_list = []

    for key, captions in tqdm(data.items(), desc="Processing entries", unit="entry"):
        if not isinstance(captions, list):
            print(f"Warning: captions for '{key}' is not a list, wrapping in list", file=sys.stderr)
            captions = [str(captions)]
        # 确保所有 caption 都是字符串
        captions = [str(c) for c in captions]
        image_keys.append(key)
        captions_list.append(captions)

    print(f"Loaded {len(image_keys)} entries")
    return image_keys, captions_list


def main():
    parser = argparse.ArgumentParser(
        description="将 captions 转换为 Hugging Face datasets 格式（PyArrow/parquet）"
    )

    input_group = parser.add_argument_group("输入（二选一）")
    input_group.add_argument(
        "--caption_json",
        type=str,
        default=None,
        help="输入 JSON 文件路径。JSON 格式: {\"image_key\": [\"caption1\", \"caption2\", ...], ...}",
    )
    input_group.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="图片目录路径。将扫描目录中所有支持的图片文件，读取对应的 .caption 文件。",
    )

    parser.add_argument(
        "--caption_extension",
        type=str,
        default=".caption",
        help="caption 文件后缀名（仅目录模式），默认: .caption",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出目录路径（HF dataset 保存路径）",
    )

    args = parser.parse_args()

    # 验证输入模式
    if args.caption_json is None and args.image_dir is None:
        parser.error("必须指定 --caption_json 或 --image_dir 之一")
    if args.caption_json is not None and args.image_dir is not None:
        parser.error("--caption_json 和 --image_dir 不能同时指定")

    # 确保 caption_extension 以 . 开头
    caption_ext = args.caption_extension
    if not caption_ext.startswith("."):
        caption_ext = "." + caption_ext

    # 处理输入
    if args.caption_json is not None:
        image_keys, captions_list = process_json_mode(args.caption_json)
    else:
        image_keys, captions_list = process_directory_mode(args.image_dir, caption_ext)

    if len(image_keys) == 0:
        print("No entries found. Creating empty dataset.")
        dataset = Dataset.from_dict({"image_key": [], "captions": []})
    else:
        # 构建 HF Dataset，包含两列:
        #   - image_key: 字符串（图片文件名/相对路径）
        #   - captions: 字符串列表（list of str）
        dataset = Dataset.from_dict({
            "image_key": image_keys,
            "captions": captions_list,
        })

    # 保存到磁盘
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving dataset to: {output_dir}")
    dataset.save_to_disk(output_dir)

    print(f"Dataset saved successfully!")
    print(f"  Total entries: {len(dataset)}")
    print(f"  Columns: {dataset.column_names}")
    if len(dataset) > 0:
        total_captions = sum(len(caps) for caps in captions_list)
        print(f"  Total captions: {total_captions}")


if __name__ == "__main__":
    main()
