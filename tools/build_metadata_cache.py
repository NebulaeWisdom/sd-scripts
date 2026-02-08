import argparse
import json
import os
from pathlib import Path

import imagesize
from tqdm import tqdm

import library.train_util as train_util
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def main(args):
    image_dir = Path(args.train_data_dir)
    assert image_dir.is_dir(), f"train_data_dir is not directory: {image_dir}"

    output_path = Path(args.output) if args.output else image_dir / "metadata_cache.json"

    image_paths = train_util.glob_images(str(image_dir), "*")
    logger.info(f"found {len(image_paths)} images")

    metadata = {}
    for image_path in tqdm(image_paths, desc="build metadata_cache"):
        image_path = Path(image_path)
        caption_path = image_path.with_suffix(args.caption_extension)

        caption = ""
        if caption_path.is_file():
            caption = caption_path.read_text(encoding="utf-8").strip()

        width, height = imagesize.get(str(image_path))
        if width == -1 or height == -1:
            logger.warning(f"failed to get image size, skip: {image_path}")
            continue

        metadata[str(image_path)] = {
            "caption": caption,
            "resolution": [width, height],
        }

    output_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"wrote metadata cache: {output_path}")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_data_dir", type=str, help="directory for train images")
    parser.add_argument("--caption_extension", type=str, default=".txt", help="caption file extension")
    parser.add_argument("--output", type=str, default=None, help="output path (default: <train_data_dir>/metadata_cache.json)")
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    main(parser.parse_args())
