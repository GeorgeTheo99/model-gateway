#!/usr/bin/env python3
"""Convert split BF16 GGUF files to HuggingFace safetensors format.

Specific to Qwen3.5-397B MoE (qwen35moe architecture).
Streams tensors from GGUF, maps names to HF convention, fuses gate+up expert
weights, and writes sharded safetensors with proper bfloat16 dtype via torch.

Memory-efficient: processes one layer at a time, writes shards incrementally.

Usage:
    python gguf_to_safetensors.py \
        --gguf-dir /local_disk0/models/gguf \
        --config-dir /local_disk0/models/qwen3.5-397b-base-config \
        --output-dir /local_disk0/models/safetensors \
        --shard-size 5
"""

import argparse
import glob
import json
import os
import re
import shutil
import sys
import time

import numpy as np
import torch
from gguf import GGUFReader
from safetensors.torch import save_file

# Layer type pattern from config: every 4th layer starting at 3 is full_attention
FULL_ATTENTION_LAYERS = set(range(3, 60, 4))  # {3, 7, 11, 15, ..., 59}


def gguf_tensor_to_torch(tensor):
    """Convert a GGUF tensor to a PyTorch tensor with correct dtype."""
    shape = tuple(int(s) for s in tensor.shape)
    dtype_val = tensor.tensor_type.value if hasattr(tensor.tensor_type, 'value') else int(tensor.tensor_type)

    if dtype_val == 30:  # BF16
        raw_bytes = tensor.data.tobytes()
        t = torch.frombuffer(bytearray(raw_bytes), dtype=torch.bfloat16).reshape(shape).clone()
        return t
    elif dtype_val == 0 or dtype_val == 32:  # F32
        raw_bytes = tensor.data.tobytes()
        t = torch.frombuffer(bytearray(raw_bytes), dtype=torch.float32).reshape(shape).clone()
        return t
    elif dtype_val == 1:  # F16
        raw_bytes = tensor.data.tobytes()
        t = torch.frombuffer(bytearray(raw_bytes), dtype=torch.float16).reshape(shape).clone()
        return t
    else:
        # Fallback: read as numpy float32
        data = tensor.data.astype(np.float32).reshape(shape)
        return torch.from_numpy(data.copy())


def map_gguf_to_hf(gguf_name):
    """Map GGUF tensor name to HF tensor name."""
    if gguf_name == 'token_embd.weight':
        return 'model.language_model.embed_tokens.weight'
    if gguf_name == 'output.weight':
        return 'lm_head.weight'
    if gguf_name == 'output_norm.weight':
        return 'model.language_model.norm.weight'

    # Block tensors
    m = re.match(r'blk\.(\d+)\.(.+)', gguf_name)
    if m:
        layer_idx = int(m.group(1))
        rest = m.group(2)
        prefix = f'model.language_model.layers.{layer_idx}'
        return _map_block_tensor(rest, prefix, layer_idx, is_mtp=False)

    # MTP tensors
    mtp_m = re.match(r'mtp\.(\d+)\.blk\.(\d+)\.(.+)', gguf_name)
    if mtp_m:
        mtp_idx = mtp_m.group(1)
        blk_idx = int(mtp_m.group(2))
        rest = mtp_m.group(3)
        prefix = f'mtp.layers.{mtp_idx}'
        return _map_block_tensor(rest, prefix, blk_idx, is_mtp=True)

    if re.match(r'mtp\.\d+\.output_norm\.weight', gguf_name):
        return 'mtp.norm.weight'
    if 'mtp' in gguf_name and 'pre_fc_norm' in gguf_name:
        if 'embd' in gguf_name:
            return 'mtp.pre_fc_norm_embedding.weight'
        if 'hidden' in gguf_name:
            return 'mtp.pre_fc_norm_hidden.weight'

    return None  # unmapped


def _map_block_tensor(rest, prefix, layer_idx, is_mtp=False):
    """Map a block-level tensor to HF name."""
    is_full_attn = layer_idx in FULL_ATTENTION_LAYERS

    # Layer norms
    if rest == 'attn_norm.weight':
        return f'{prefix}.input_layernorm.weight'
    if rest == 'post_attention_norm.weight':
        return f'{prefix}.post_attention_layernorm.weight'

    # Linear attention (non-full-attention layers, non-MTP)
    if not is_full_attn and not is_mtp:
        linear_mapping = {
            'ssm_a': f'{prefix}.linear_attn.A_log',
            'ssm_conv1d.weight': f'{prefix}.linear_attn.conv1d.weight',
            'ssm_dt.bias': f'{prefix}.linear_attn.dt_bias',
            'ssm_alpha.weight': f'{prefix}.linear_attn.in_proj_a.weight',
            'ssm_beta.weight': f'{prefix}.linear_attn.in_proj_b.weight',
            'attn_qkv.weight': f'{prefix}.linear_attn.in_proj_qkv.weight',
            'attn_gate.weight': f'{prefix}.linear_attn.in_proj_z.weight',
            'ssm_norm.weight': f'{prefix}.linear_attn.norm.weight',
            'ssm_out.weight': f'{prefix}.linear_attn.out_proj.weight',
        }
        if rest in linear_mapping:
            return linear_mapping[rest]

    # Full attention (every 4th layer, or MTP)
    if is_full_attn or is_mtp:
        attn_mapping = {
            'attn_q.weight': f'{prefix}.self_attn.q_proj.weight',
            'attn_k.weight': f'{prefix}.self_attn.k_proj.weight',
            'attn_v.weight': f'{prefix}.self_attn.v_proj.weight',
            'attn_output.weight': f'{prefix}.self_attn.o_proj.weight',
            'attn_q_norm.weight': f'{prefix}.self_attn.q_norm.weight',
            'attn_k_norm.weight': f'{prefix}.self_attn.k_norm.weight',
        }
        if rest in attn_mapping:
            return attn_mapping[rest]

    # MoE components
    moe_mapping = {
        'ffn_gate_inp.weight': f'{prefix}.mlp.gate.weight',
        'ffn_gate_inp_shexp.weight': f'{prefix}.mlp.shared_expert_gate.weight',
        'ffn_gate_shexp.weight': f'{prefix}.mlp.shared_expert.gate_proj.weight',
        'ffn_up_shexp.weight': f'{prefix}.mlp.shared_expert.up_proj.weight',
        'ffn_down_shexp.weight': f'{prefix}.mlp.shared_expert.down_proj.weight',
        'ffn_down_exps.weight': f'{prefix}.mlp.experts.down_proj',
        # Fusion markers
        'ffn_gate_exps.weight': f'__GATE_EXPS__{prefix}',
        'ffn_up_exps.weight': f'__UP_EXPS__{prefix}',
    }
    if rest in moe_mapping:
        return moe_mapping[rest]

    # Per-expert tensors (MTP)
    expert_m = re.match(r'ffn_(gate|up|down)\.(\d+)\.weight', rest)
    if expert_m:
        proj_type = expert_m.group(1)
        expert_idx = expert_m.group(2)
        proj_map = {'gate': 'gate_proj', 'up': 'up_proj', 'down': 'down_proj'}
        return f'{prefix}.mlp.experts.{expert_idx}.{proj_map[proj_type]}.weight'

    return None


def convert_gguf_to_safetensors(gguf_dir, config_dir, output_dir, shard_size_gb=5.0):
    """Main conversion: stream GGUF -> sharded safetensors."""
    os.makedirs(output_dir, exist_ok=True)
    shard_size_bytes = int(shard_size_gb * 1024**3)

    gguf_files = sorted(glob.glob(os.path.join(gguf_dir, '*BF16*.gguf')))
    print(f"Found {len(gguf_files)} GGUF files:")
    for f in gguf_files:
        print(f"  {os.path.basename(f)}: {os.path.getsize(f) / (1024**3):.1f}GB")

    # Phase 1: Build a tensor manifest (name -> file, index) without loading data
    print("\n=== Phase 1: Building tensor manifest ===")
    manifest = []  # list of (gguf_path, tensor_index, gguf_name, hf_name, shape, dtype_val)
    gate_exps_info = {}  # layer_prefix -> (gguf_path, tensor_index)
    up_exps_info = {}

    for gguf_path in gguf_files:
        print(f"  Scanning: {os.path.basename(gguf_path)}")
        reader = GGUFReader(gguf_path)
        for idx, tensor in enumerate(reader.tensors):
            hf_name = map_gguf_to_hf(tensor.name)
            if hf_name is None:
                print(f"    UNMAPPED: {tensor.name}")
                continue
            shape = tuple(int(s) for s in tensor.shape)
            dtype_val = tensor.tensor_type.value if hasattr(tensor.tensor_type, 'value') else int(tensor.tensor_type)

            if hf_name.startswith('__GATE_EXPS__'):
                prefix = hf_name[len('__GATE_EXPS__'):]
                gate_exps_info[prefix] = (gguf_path, idx, shape, dtype_val)
            elif hf_name.startswith('__UP_EXPS__'):
                prefix = hf_name[len('__UP_EXPS__'):]
                up_exps_info[prefix] = (gguf_path, idx, shape, dtype_val)
            else:
                manifest.append((gguf_path, idx, tensor.name, hf_name, shape, dtype_val))
        del reader

    # Add fused gate_up entries to manifest
    for prefix in sorted(gate_exps_info.keys()):
        hf_name = f'{prefix}.mlp.experts.gate_up_proj'
        gate_shape = gate_exps_info[prefix][2]
        # Fused shape: dim 1 doubles
        fused_shape = (gate_shape[0], gate_shape[1] * 2, gate_shape[2])
        manifest.append(('__FUSED__', -1, f'fused_gate_up_{prefix}', hf_name, fused_shape, 30))

    # Sort by HF name for consistent sharding
    manifest.sort(key=lambda x: x[3])
    print(f"  Total tensors: {len(manifest)} (incl. {len(gate_exps_info)} fused gate_up)")

    # Phase 2: Plan shards
    print("\n=== Phase 2: Planning shards ===")
    dtype_sizes = {30: 2, 0: 4, 32: 4, 1: 2}
    shards_plan = []  # list of lists of manifest entries
    current_shard = []
    current_size = 0

    for entry in manifest:
        _, _, _, hf_name, shape, dtype_val = entry
        elem_size = dtype_sizes.get(dtype_val, 4)
        tensor_bytes = 1
        for s in shape:
            tensor_bytes *= s
        tensor_bytes *= elem_size

        if current_size + tensor_bytes > shard_size_bytes and current_shard:
            shards_plan.append(current_shard)
            current_shard = []
            current_size = 0
        current_shard.append(entry)
        current_size += tensor_bytes

    if current_shard:
        shards_plan.append(current_shard)

    print(f"  {len(manifest)} tensors -> {len(shards_plan)} shards")

    # Phase 3: Write shards one at a time (memory efficient)
    print(f"\n=== Phase 3: Writing shards ===")
    # Cache GGUF readers to avoid re-opening
    readers_cache = {}
    weight_map = {}
    total_size = 0

    for shard_idx, shard_entries in enumerate(shards_plan, 1):
        filename = f"model-{shard_idx:05d}-of-{len(shards_plan):05d}.safetensors"
        filepath = os.path.join(output_dir, filename)

        shard_tensors = {}
        for gguf_path, tensor_idx, gguf_name, hf_name, shape, dtype_val in shard_entries:
            if gguf_path == '__FUSED__':
                # Load and fuse gate + up expert tensors
                prefix = hf_name.rsplit('.mlp.experts.gate_up_proj', 1)[0]
                g_path, g_idx, g_shape, g_dtype = gate_exps_info[prefix]
                u_path, u_idx, u_shape, u_dtype = up_exps_info[prefix]

                if g_path not in readers_cache:
                    readers_cache[g_path] = GGUFReader(g_path)
                if u_path not in readers_cache:
                    readers_cache[u_path] = GGUFReader(u_path)

                gate_t = gguf_tensor_to_torch(readers_cache[g_path].tensors[g_idx])
                up_t = gguf_tensor_to_torch(readers_cache[u_path].tensors[u_idx])
                # Fuse: concatenate along dim 1
                fused = torch.cat([gate_t, up_t], dim=1)
                shard_tensors[hf_name] = fused
                del gate_t, up_t
            else:
                if gguf_path not in readers_cache:
                    readers_cache[gguf_path] = GGUFReader(gguf_path)
                t = gguf_tensor_to_torch(readers_cache[gguf_path].tensors[tensor_idx])
                shard_tensors[hf_name] = t

            weight_map[hf_name] = filename

        sys.stdout.write(f"  [{shard_idx}/{len(shards_plan)}] {filename} ({len(shard_tensors)} tensors)...")
        sys.stdout.flush()
        start = time.time()
        save_file(shard_tensors, filepath)
        elapsed = time.time() - start
        shard_bytes = os.path.getsize(filepath)
        total_size += shard_bytes
        print(f" {shard_bytes / 1024**3:.1f}GB in {elapsed:.1f}s")

        del shard_tensors
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Close readers
    del readers_cache

    # Write index
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(os.path.join(output_dir, "model.safetensors.index.json"), 'w') as f:
        json.dump(index, f, indent=2)

    # Copy config and tokenizer files
    print(f"\n=== Copying config and tokenizer files ===")
    for fname in os.listdir(config_dir):
        if fname.startswith('.'):
            continue
        src = os.path.join(config_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(output_dir, fname))
            print(f"  Copied {fname}")

    print(f"\n=== DONE === Total: {total_size / 1024**3:.1f}GB across {len(shards_plan)} shards")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gguf-dir', required=True)
    parser.add_argument('--config-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--shard-size', type=float, default=5.0, help='Shard size in GB')
    args = parser.parse_args()
    convert_gguf_to_safetensors(args.gguf_dir, args.config_dir, args.output_dir, args.shard_size)
