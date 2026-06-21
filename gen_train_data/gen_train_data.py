import os
from pathlib import Path

import h5py
import numpy as np
from pyspark.sql import Row, SparkSession
from pyspark.sql import functions as F

from parse_args import parse_args


def build_jobs(args):
    """
    Build one Spark job per input HDF5 file.

    Each job is a dictionary. Later, Spark turns this into a distributed DataFrame.
    """
    src_root = Path(args.src_data_path)
    save_root = Path(args.save_data_path)

    patch_size_hr = args.patch_size_lr * args.scale
    stride = args.stride if args.stride is not None else args.patch_size_lr * 2

    jobs = []

    ### loop over all the datasets that are provided
    for dataset in args.datasets:
        input_dir = src_root / dataset / args.data_for
        output_dir = save_root / dataset

        output_dir.mkdir(parents=True, exist_ok=True)

        if not input_dir.exists():
            print(f"[WARN] Skipping missing directory: {input_dir}")
            continue

        ### loop over each scene in the dataset
        ### for each scene create a job
        for input_path in sorted(input_dir.iterdir()):
            if not input_path.is_file():
                continue

            jobs.append(
                {
                    "dataset": dataset,
                    "scene_name": input_path.stem,
                    "input_path": str(input_path.resolve()),
                    "output_dir": str(output_dir.resolve()),
                    "patch_size_hr": patch_size_hr,
                    "stride": stride,
                    "overwrite": bool(args.overwrite),
                }
            )

    return jobs


def process_scene(job):
    """
    Process one HDF5 scene file.

    This function runs on a Spark worker.
    It reads the HR light field, extracts patches, and writes HDF5 patch files.
    """
    dataset = job["dataset"]
    scene_name = job["scene_name"]
    input_path = job["input_path"]
    output_dir = Path(job["output_dir"])
    patch_size_hr = int(job["patch_size_hr"])
    stride = int(job["stride"])

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with h5py.File(input_path, "r") as hf:

            lf = np.asarray(hf["HR"], dtype=np.float32)

        channels, height, width = lf.shape

        usable_h = height - ((height - patch_size_hr) % stride)
        usable_w = width - ((width - patch_size_hr) % stride)
        num_i = (usable_h - patch_size_hr) // stride
        num_j = (usable_w - patch_size_hr) // stride

        patches_written = 0
        patches_skipped = 0

        for i in range(num_i + 1):
            for j in range(num_j + 1):
                x = i * stride
                y = j * stride

                patch = lf[
                    :,
                    x : x + patch_size_hr,
                    y : y + patch_size_hr,
                ]

                expected_shape = (25, patch_size_hr, patch_size_hr)
                if patch.shape != expected_shape:
                    raise RuntimeError(
                        f"Bad patch shape {patch.shape}, expected {expected_shape}"
                    )

                output_path = output_dir / f"{scene_name}_{i}-{j}.h5"
                with h5py.File(output_path, "w") as out:
                    out.create_dataset("LF", data=patch, dtype="single")

                patches_written += 1

        return Row(
            dataset=dataset,
            scene_name=scene_name,
            status="ok",
            patches_written=patches_written,
            patches_skipped=patches_skipped,
            message="",
        )

    except Exception as exc:
        return Row(
            dataset=dataset,
            scene_name=scene_name,
            status="error",
            patches_written=0,
            patches_skipped=0,
            message=repr(exc),
        )


def process_partition(rows):
    """
    Process one Spark partition.

    A partition is just a chunk of rows.
    Each row describes one HDF5 scene file.
    """
    for row in rows:
        job = row.asDict()
        yield process_scene(job)


def main():
    args = parse_args()

    jobs = build_jobs(args)

    if not jobs:
        raise RuntimeError(
            "No input files found. Check --src_data_path, --data_for, and --datasets."
        )

    num_partitions = args.num_partitions
    if num_partitions is None:
        num_partitions = min(len(jobs), os.cpu_count() or 1)

    spark = (
        SparkSession.builder
        .appName("HSLFSR patch generation")
        .master(args.master)
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    ### split the jobs into multiple partitions
    jobs_df = spark.createDataFrame(jobs).repartition(num_partitions)

    print("\nInput scenes:")
    jobs_df.select("dataset", "scene_name", "input_path").show(20, truncate=False)

    print("\nGenerating patches...")

    ### execute the jobs
    results = jobs_df.rdd.mapPartitions(process_partition).collect()

    results_df = spark.createDataFrame(results)

    print("\nPer-scene results:")
    results_df.orderBy("dataset", "scene_name").show(100, truncate=False)

    print("\nSummary:")
    (
        results_df.groupBy("dataset", "status")
        .agg(
            F.count("*").alias("scenes"),
            F.sum("patches_written").alias("patches_written"),
            F.sum("patches_skipped").alias("patches_skipped"),
        )
        .orderBy("dataset", "status")
        .show(truncate=False)
    )

    spark.stop()


if __name__ == "__main__":
    main()