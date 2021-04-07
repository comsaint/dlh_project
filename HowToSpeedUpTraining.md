# How to speed up training
This is a collection of hints on how to speed up model training.

## Benchmark
On my machine (CPU: Intel i5 3.3GHz 12core, GPU:1xGTX3070(8G), SSD disk), it takes about 3 minutes/epoch to fine tune a pre-trained model (thus ~30 mins for 10 epochs).

## Test your machine
For testing, I prepared a script under `./speed_test/script.py` for speed test. The test script will download and train a model (DenseNet by default) for 1 epoch with an image subset, which takes a few minutes on my machine. If it does not finish within reasonable time, chances are there is a bottleneck with disk I/O. Please refer to the following sections for optimization actions.

To use, activate the correct environment, then `cd` into `notebook` and run `python -m torch.utils.bottleneck script.py` in terminal. While the script is running, the first thing to check is to look at the prints on console. You should see the following lines:
> Running environment analysis...
> Running your script with cProfile
> **Device: cuda**
> Number of images: 112120
> Number of labels: 15
Note the **Device**: if it says `cuda`, then PyTorch can detect your GPU; otherwise it will show `cpu` which means only CPU is used. If this is the case, check (here)[https://pytorch.org/get-started/locally/] for how to properly setup PyTorch.

You may want to monitor your GPU usage to see if it is busy. Open a terminal and run `watch -n 0.5 nvidia-smi` will show the temperature, power, utilization... of your GPU(s). I found the power consumption a good indicator.

If the script finishes, first thing to check is the last few lines. You should see something that looks like:
> Self CPU time total: 14.327s
> Self CUDA time total: 14.378s

In my case, CPU and GPU time are close which is a good sign (neither CPU or GPU are overstressed).

Most likely, you will find your CPU time much more that GPU time. In this case, check the print under the `autograd profiler output (CPU mode)` section. The top items are where most time was spent. For me they are all `DataLoader` objects, which says my bottleneck is on disk I/O.

To dive deeper, you can find more information of `torch.utils.bottleneck` online.

## Hardware
1. Make sure your data is stored on an SSD disk. Otherwise the CPU will be busy fetching data, leaving GPU idle.
2. Make sure you have GPU(s). Tough luck without one.

## Software
1. In my case, the bottleneck was on loading images (even from SSD). I have not find a solution on Windows machine, but on Ubuntu or Mac a big boost is to replace the `pillow` package with `pillow-SIMD`, which is the default package in `requirements.txt`. In my case, it shortens training time by **10x**.
2. Increasing `NUM_WORKERS` and/or lowering `TRAIN_BATCH_SIZE` may also help. However, note that changing `TRAIN_BATCH_SIZE` may impact accuracy.


> More to come...