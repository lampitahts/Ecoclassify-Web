import argparse
import os
import random
import shutil
from collections import Counter
from pathlib import Path

IMG_EXTS = ('.jpg', '.jpeg', '.png')


def list_class_files(data_dir):
    data_dir = Path(data_dir).resolve()
    class_files = {}
    # walk recursively and treat each leaf directory that contains images as a class
    for root, dirs, files in os.walk(data_dir):
        imgs = [Path(root) / f for f in files if Path(f).suffix.lower() in IMG_EXTS]
        if not imgs:
            continue
        # class label normalized to relative path with underscores
        rel = Path(root).relative_to(data_dir)
        class_name = str(rel).replace(os.sep, '_') if str(rel) != '.' else data_dir.name
        class_files[class_name] = imgs
    # ensure deterministic ordering
    return dict(sorted(class_files.items()))


def print_counts(class_files):
    for cls, files in class_files.items():
        print(f"{cls}: {len(files)}")


def undersample_copy(class_files, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # find min count
    min_count = min(len(v) for v in class_files.values())
    print('Undersampling to', min_count, 'per class')
    for cls, files in class_files.items():
        cls_out = out_dir / cls
        cls_out.mkdir(parents=True, exist_ok=True)
        chosen = random.sample(list(files), min_count)
        for src in chosen:
            shutil.copy(src, cls_out / src.name)


def oversample_copy(class_files, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    max_count = max(len(v) for v in class_files.values())
    print('Oversampling to', max_count, 'per class')
    for cls, files in class_files.items():
        cls_out = out_dir / cls
        cls_out.mkdir(parents=True, exist_ok=True)
        # copy all existing
        for src in files:
            shutil.copy(src, cls_out / src.name)
        # sample with replacement to reach max_count
        files_list = list(files)
        if len(files_list) == 0:
            continue
        while len(list(cls_out.iterdir())) < max_count:
            src = random.choice(files_list)
            dst_name = f"os_{random.randint(0,10**9)}_{src.name}"
            shutil.copy(src, cls_out / dst_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--counts', action='store_true')
    parser.add_argument('--out_dir', default=None)
    parser.add_argument('--strategy', choices=['undersample','oversample'], default='undersample')
    args = parser.parse_args()

    class_files = list_class_files(args.data_dir)
    if args.counts:
        print_counts(class_files)
    if args.out_dir:
        if args.strategy == 'undersample':
            undersample_copy(class_files, args.out_dir)
        else:
            oversample_copy(class_files, args.out_dir)


if __name__ == '__main__':
    main()
