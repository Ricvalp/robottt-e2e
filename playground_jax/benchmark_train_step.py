#!/usr/bin/env python
"""Benchmark train_step performance to identify bottlenecks."""
import jax
import jax.numpy as jnp
import time
import sys

print('=' * 60)
print('JAX Performance Benchmark')
print('=' * 60)
print(f'JAX backend: {jax.default_backend()}')
print(f'JAX devices: {jax.devices()}')
print()

from conditional_model import ConditionalUNet, DDPM
from train_cond_maml_mnist import select_fast_params, create_fast_mask, train_step, maml_step

# Initialize model
key = jax.random.PRNGKey(42)
model = ConditionalUNet(in_channels=1, base_channels=64, image_size=32)
dummy_x = jnp.ones((1, 32, 32, 1))
dummy_t = jnp.ones((1,))
dummy_cond = jnp.ones((1, 32, 32, 1))
params = model.init(key, dummy_x, dummy_t, cond=dummy_cond, train=False)

fast_paths = select_fast_params(params, 'up_down_mid_head_gn')
fast_mask = create_fast_mask(params, fast_paths)
param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
print(f'Model params: {param_count/1e6:.2f}M')
print()

# Create data on device
batch = jax.random.normal(key, (32, 32, 32, 1))
cond = jax.random.normal(key, (32, 32, 32, 1))
ddpm = DDPM()

# Benchmark 1: Forward pass only
print('--- Forward pass only ---')
@jax.jit
def forward_only(params, x, t, cond):
    return model.apply(params, x, t, cond=cond, train=False)

t = jnp.ones((32,)) * 0.5
_ = forward_only(params, batch, t, cond)  # warmup
jax.block_until_ready(_)

start = time.time()
for _ in range(100):
    out = forward_only(params, batch, t, cond)
jax.block_until_ready(out)
fwd_time = (time.time() - start) / 100 * 1000
print(f'Forward: {fwd_time:.2f}ms')

# Benchmark 2: Forward + backward (single loss)
print('\n--- Forward + backward (no MAML) ---')
@jax.jit
def loss_and_grad(params, x, t, noise, cond):
    def loss_fn(params):
        v_pred = model.apply(params, x, t, cond=cond, train=False)
        return jnp.mean(v_pred ** 2)
    return jax.value_and_grad(loss_fn)(params)

_ = loss_and_grad(params, batch, t, batch, cond)  # warmup
jax.block_until_ready(_)

start = time.time()
for _ in range(50):
    loss, grads = loss_and_grad(params, batch, t, batch, cond)
jax.block_until_ready(loss)
fwdbwd_time = (time.time() - start) / 50 * 1000
print(f'Forward+backward: {fwdbwd_time:.2f}ms')

# Benchmark 3: Full MAML train_step (JIT compiled)
print('\n--- Full MAML train_step (JIT) ---')
key, step_key = jax.random.split(key)
outer_loss, grads = train_step(
    model.apply, params, batch, cond, fast_mask, step_key,
    0.001, 1, ddpm.beta_start, ddpm.beta_end, ddpm.train_steps,
)
jax.block_until_ready(outer_loss)
print(f'Warmup complete')

start = time.time()
for i in range(20):
    key, step_key = jax.random.split(key)
    outer_loss, grads = train_step(
        model.apply, params, batch, cond, fast_mask, step_key,
        0.001, 1, ddpm.beta_start, ddpm.beta_end, ddpm.train_steps,
    )
jax.block_until_ready(outer_loss)
maml_time = (time.time() - start) / 20 * 1000
print(f'MAML train_step: {maml_time:.2f}ms')
print(f'Throughput: {1000/maml_time:.1f} iter/s')

print('\n' + '=' * 60)
print('Summary:')
print(f'  Forward only:      {fwd_time:.2f}ms')
print(f'  Forward+backward:  {fwdbwd_time:.2f}ms')
print(f'  MAML train_step:   {maml_time:.2f}ms')
print(f'  MAML/FwdBwd ratio: {maml_time/fwdbwd_time:.1f}x')
print('=' * 60)
