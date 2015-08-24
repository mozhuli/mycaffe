#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=examples/myself/solver.prototxt \
    --snapshot=examples/myself/caffenet_train_5000.solverstate
