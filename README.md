# pytorch-apple-silicon-benchmarks

Benchmarks of PyTorch on Apple Silicon.

This is a work in progress, if there is a dataset or model you would like to add just open an issue or a PR.

# Prepare environment

Create conda env with python compiled for `osx-arm64` and activate it with:

```bash
CONDA_SUBDIR=osx-arm64 conda create -n native python -c conda-forge
conda activate native
```

and install `pytorch` nightly build with:

```bash
pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

and finally install `datasets` and `transformers` with:

```bash
pip install transformers datasets
```


# Devices

- M1 Max CPU 32GB: 10 cores, 2 efficient + 8 performance up to ~3GHz; Peak measured power consuption: `30W`.
- M1 Max GPU 32GB: 32 cores; Peak measured power consuption: `46W`.
- NVIDIA V100 16GB (SXM2): 5,120 CUDA cores + 640 tensor cores; Peak measured power consuption: `310W`.


# Results

## BERT Transformers in Sequence Classification.

Run the experiments by yourself with:

```bash
python tests/transformers_sequence_classification.py \
    --device <cpu|cuda|mps> \
    --pre_trained_name <bert-base-cased|bert-large-cased> \
    --batch_size <32|64|128> \
    --mode <training|inference> \
    --steps 100 \
    --sequence_length <128|512>
```

The following tables show the time needed to complete 100 steps without gradient accumulation. `-` means that the script went out of memory. All experiments have been run with `float32`.


### `bert-base-cased`

**Training**:

| Batch size | Sequence length | M1 Max CPU (32GB)   | M1 Max GPU (32GB) | V100 (16GB) |
| ---------- | --------------- | ------------------- | ----------------- | ----------- |
| 16         | 128             | 2m 29s              | 1m 3s             | 12s         |
| 64         | 128             | 8m 32s              | 2m 57s            | 41s         |
| 256        | 128             | 50m 10s             | 1h 49m 9s         | -           |
| 16         | 512             | 11m 22s             | 9m 28s            | 47s         |
| 64         | 512             | 1h 21m 2s           | 3h 26m 4s         | -           |
| 256        | 512             | 6h 33m 7s           | -                 | -           |


**Inference**:

| Batch size | Sequence length | M1 Max CPU (32GB) | M1 Max GPU (32GB) | V100 (16GB) |
| ---------- | --------------- | ----------------- | ----------------- | ----------- |
| 16         | 128             | 52s               | 16s               | 4s          |
| 64         | 128             | 3m 2s             | 50s               | 13s         |
| 256        | 128             | 11m 25s           | 3m 22s            | 54s         |
| 16         | 512             | 4m 22s            | 1m 1s             | 16s         |
| 64         | 512             | 17m 51s           | 3m 59s            | 1m 4s       |
| 256        | 512             | 1h 10m 41s        | 15m 47s           | 4m 10s      |



# Considerations

- This is the first alpha ever to support the M1 family of processors, so you should expect performance to increase further in the next months since many optimizations will be added to the MPS backed.
- ~~At the moment I experienced a progressive slowdown with MPS such that the first iteration took more than half the time than the last.~~ (seems solved in latest release)
- Before deciding whether the M1 Max could be your best choice, consider that it has no `float64` support and neither `fp16` tensor cores.
- It seems that there is no real limit to the batch size with the M1 Max because it is able to use the swap also for the 'GPU' memory. However, this really slows down training.


# FAQ

- If you cannot install `tokenizers` because `rust` is missing, do the following:
```
brew install rustup
rustup-init
source ~/.cargo/env
```
