import argparse

DEFAULT_DATASETS = [
    "Lab_day",
    "Lab_night",
    "Indoors_day",
    "Indoors_night",
    "Showroom",
    "Outdoors",
]

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--src_data_path",
        type=str,
        default="../datasets",
        help="Root folder containing the original datasets.",
    )

    parser.add_argument(
        "--data_for",
        type=str,
        default="training_hsi",
        help="Subfolder inside each dataset, e.g. training_hsi.",
    )

    parser.add_argument(
        "--save_data_path",
        type=str,
        default="data/train",
        help="Output root. Use data/train to match LightFieldDataset.",
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Dataset folders to process.",
    )

    parser.add_argument(
        "--patch_size_lr",
        type=int,
        default=32,
        help="Low-resolution patch size. HR patch size is scale * patch_size_lr.",
    )

    parser.add_argument(
        "--scale",
        type=int,
        default=4,
        help="Super-resolution scale factor.",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Stride for HR patches. Defaults to patch_size_lr * 2.",
    )

    parser.add_argument(
        "--num_partitions",
        type=int,
        default=None,
        help="Number of Spark partitions. Defaults to min(number of files, CPU cores).",
    )

    parser.add_argument(
        "--master",
        type=str,
        default="local[*]",
        help="Spark master. Use local[*] for local development.",
    )

    return parser.parse_args()