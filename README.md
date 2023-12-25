This script runs a toy task on all available GPUs (according to `nvidia-smi`).

While running, `GPU Util` and `Memory used` as reported by `nvidia-smi` should be nonzero for all GPUs.

```sh
$ pip install cupy
$ python main.py
```

### Notes

`cupy` install version may depend on your CUDA driver:

```sh
$ nvcc --version
$ pip install --pre cupy-cudaXYZ
```

where `XYZ` is your CUDA version. For example, CUDA 10.1 -> `cupy-cuda101`.
