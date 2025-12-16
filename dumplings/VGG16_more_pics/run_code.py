import pandas as pd
import torch as th
import numpy as np
th.device("cuda" if th.cuda.is_available() else "cpu")

# ==================================================

t = th.rand(2,3,4)
t.shape, t.dtype 

# ==================================================

mask = [1,0,1]

t[:,mask]

# ==================================================

t[0].shape

# ==================================================

t = th.rand(4,4)
display(t)

t[:,1::2] +=1
t[:,::2] -=1

t

# ==================================================

t = th.zeros([100000, 10], dtype=th.float32)
t[th.arange(100000),th.randint(0, 10, (100000,))] = 1
t.mean(0)

# ==================================================

t = th.randint(0,11,(10,10))
mask = np.bool([i[::-1] for i in np.vectorize(np.logical_xor)(np.triu(np.ones(t.shape),2),np.triu(np.ones(t.shape),1))])

display(t)
t[mask]

# ==================================================

n = 10
rows, cols = np.ogrid[:n, :n]
mask = rows + cols < n - 1
t[mask]

# ==================================================

t = th.randint(0,101,(5,5))
row_max, col_max = th.unravel_index(th.argmax(t), t.shape)

row_start = max(0, row_max - 1)
row_end = min(5, row_max + 2)
col_start = max(0,col_max - 1)
col_end = min(5, col_max + 2)

mask = th.zeros_like(t, dtype=th.bool)
mask[row_start:row_end, col_start:col_end] = True

result = t.clone()
result[~mask] = 0

t, result

# ==================================================

t_u = th.stack([t, result], dim=0)
display(t_u)
th.save(t_u, 'tensor.pt')

load_t=th.load('tensor.pt')
display(load_t)
th.all(t_u == load_t)

# ==================================================

t5 = th.randint(1,11,(2, 3, 5, 5), dtype=th.float32)
t5.mean(dim=(3,2), keepdim=False).reshape(2,3,1)

# ==================================================

import matplotlib.pyplot as plt
import time
N=100_000_000
lambda_param = 5.0

start_cpu = time.time()
exponential_dist = th.distributions.Exponential(th.tensor(lambda_param))
sample_tensor = exponential_dist.sample([N])
end_cpu = time.time() - start_cpu

print(f'{end_cpu=} sec')

plt.figure(figsize=(10, 6))
plt.hist(sample_tensor.numpy(), bins=100, density=True, alpha=0.7, color='skyblue', edgecolor='black', linewidth=0.5, label='Эмпирическая гистограмма')
x = np.linspace(0, 3.5, 1000)
y = lambda_param * np.exp(-lambda_param * x)
plt.plot(x, y, 'r-', linewidth=2, label=f'Теоретическая плотность ({lambda_param=})')
plt.legend()
plt.show()



if th.cuda.is_available():
    print('Cuda available!')
    start_gpu = time.time()
    exponential_dist = th.distributions.Exponential(th.tensor(lambda_param, device='cuda'))
    sample_tensor = exponential_dist.sample([N])
    end_gpu = time.time() - start_gpu

    print(f'{end_gpu=} sec')

# ==================================================

t = th.randint(0,256,(10, 6, 6, 3))
t[:, :, :, 0][:,:,:2] =0
t[:, :, :, 1][:,:,[2,3]] =0
t[:, :, :, 2][:,:,[4,5]] =0

t

# ==================================================

