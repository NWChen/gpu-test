# This script runs a toy task on all available GPUs (according to `nvidia-smi`).
# While running, `GPU Util` and `Memory used` as reported by `nvidia-smi` should be nonzero for all GPUs.

import multiprocessing
import cupy as cp
import os
import subprocess

class single_gpu():
    def __init__(self, gpu_id):
        self.gpu_id = str(gpu_id)
        self.all_gpu_ids = os.environ['CUDA_VISIBLE_DEVICES']

    def __enter__(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu_id

    def __exit__(self, _exc_type, _exc_value, _exc_traceback):
        os.environ['CUDA_VISIBLE_DEVICES'] = self.all_gpu_ids


def run_task(gpu_id):
    with single_gpu(gpu_id) as c:
        print(f'Task started on GPU {gpu_id}')
        DIM = 10000
        a = cp.random.rand(DIM, DIM)
        b = cp.random.rand(DIM, DIM)
        result = cp.matmul(a, b)
        print(f'Task completed on GPU {gpu_id}')


def get_gpu_ids():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index', '--format=csv,nounits,noheader'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.stdout.strip().split('\n')
    except (subprocess.CalledProcessError, ValueError):
        return []


gpu_ids = get_gpu_ids()
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    print(f'Found no $CUDA_VISIBLE_DEVICES, setting $CUDA_VISIBLE_DEVICES={gpu_ids}')
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_ids)


processes = []
for gpu_id in gpu_ids:
    p = multiprocessing.Process(target=run_task, args=(gpu_id,))
    processes.append(p)
    p.start()

for p in processes:
    p.join()

print(f'Finished')

