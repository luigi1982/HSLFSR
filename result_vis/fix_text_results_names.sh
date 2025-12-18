#!/usr/bin/env bash
set -e

model=$1
exp=$2

path=results/$model/evaluate/$exp/id/Texts

mv $path/scene_9.h5 $path/scene_8.h5
mv $path/scene_1.h5 $path/scene_9.h5
mv $path/scene_2.h5 $path/scene_1.h5