import sys


sys.path.insert(0, "/rwthfs/rz/cluster/home/ra717140/setups/exp2025-08-22/projects/mdlm")
sys.path.insert(0, "/rwthfs/rz/cluster/home/ra717140/setups/exp2025-08-22/tools")

import torch
import torch.nn.functional as F
import numpy as np
import models.dit
import omegaconf
from returnn.frontend import RotaryPosSelfAttention, sinusoidal_encoding

import returnn.frontend as rf
from returnn.tensor import Dim, Tensor
from returnn.frontend.decoder.transformer import FeedForward
from typing import Union, Dict, Any, Callable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

pytorch_config = omegaconf.OmegaConf.create(
    {
        "model": {
            "hidden_size": 1024,
            "n_heads": 16,
            "n_blocks": 24,
            "cond_dim": 128,
            "dropout": 0.0,
            "scale_by_sigma": True,
        }
    }
)

vocab_size = 10241

pytorch_model = models.dit.DIT(pytorch_config, vocab_size=vocab_size)
pytorch_model.eval()
pytorch_model = pytorch_model.to(device)

ckpt_path = "/home/ra717140/setups/exp2025-08-22/alias/diffusion-lm/trained_models/mdlm-returnn-librispeech-medium-128-1m2-laplace-20k-time-lrPL-lr1-wd-001-b2-099/output/checkpoints/best.ckpt"
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

state_dict_fixed = {}
for k, v in ckpt["state_dict"].items():
    new_k = k
    if new_k.startswith("model."):
        new_k = new_k[6:]
    if new_k.startswith("backbone."):
        new_k = new_k[9:]
    state_dict_fixed[new_k] = v

missing, unexpected = pytorch_model.load_state_dict(state_dict_fixed, strict=False)
print(f"PyTorch: Missing={len(missing)}, Unexpected={len(unexpected)}")
print("PyTorch model loaded")

rf.select_backend_torch()


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


class CustomFeedForward(rf.Module):
    def __init__(
        self,
        out_dim: Dim,
        *,
        input_dim: Dim,
        ff_dim: Dim,
        dropout: float = 0.1,
        activation: Union[Callable[[Tensor], Tensor], Dict[str, Any], rf.Module] = rf.relu,
        with_bias: bool = True,
    ):
        super().__init__()

        self.out_dim = out_dim
        self.dropout = dropout
        self.dropout_broadcast = rf.dropout_broadcast_default()
        self.activation = activation

        self.linear_ff = rf.Linear(input_dim, ff_dim, with_bias=with_bias)
        self.linear_out = rf.Linear(ff_dim, out_dim, with_bias=with_bias)

    def __call__(self, inp: Tensor) -> Tensor:
        x_ff1 = self.linear_ff(inp)
        x_act = self.activation(x_ff1)
        x_drop = rf.dropout(x_act, self.dropout, axis=self.dropout_broadcast and self.linear_ff.out_dim)
        x_ff2 = self.linear_out(x_drop)
        return x_ff2


class TimeEmbedder(rf.Module):
    def __init__(self, hidden_dim, frequency_embedding_size=256):
        super().__init__()
        self.hidden_size = hidden_dim
        self.frequency_embedding_size = Dim(dimension=frequency_embedding_size, name="frequency_embedding_dim")
        self.mlp = CustomFeedForward(out_dim=hidden_dim, input_dim=self.frequency_embedding_size, ff_dim=hidden_dim)

    def __call__(self, t: Tensor) -> Tensor:
        t_freq = sinusoidal_encoding(t, feat_dim=self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DITLayer(rf.Module):
    def __init__(
        self,
        out_dim: Dim = Dim(512, name="dit-default-out-dim"),
        *,
        cond_dim: Dim = Dim(128, name="dit-default-cond-dim"),
        dropout: float = 0.1,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        att_dropout: float = 0.1,
        norm: Union[type, Dict[str, Any], rf.Module, Callable] = rf.LayerNorm,
    ):
        super().__init__()
        self.dropout = dropout
        self.dropout_broadcast = rf.dropout_broadcast_default()
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.out_dim = out_dim
        self.cond_dim = cond_dim
        self.att_dropout = att_dropout

        self.mlp = FeedForward(
            out_dim=out_dim,
            ff_dim=Dim(mlp_ratio * out_dim.dimension, name="ff_dim"),
            activation=rf.gelu,
        )

        self.self_attn = RotaryPosSelfAttention(
            in_dim=out_dim,
            proj_dim=out_dim,
            key_dim_total=out_dim,
            value_dim_total=out_dim,
            num_heads=num_heads,
            with_bias=False,
        )

        self.norm1 = rf.LayerNorm(out_dim, with_bias=False)
        self.norm2 = rf.LayerNorm(out_dim, with_bias=False)

        self.adaLN_modulation = rf.Linear(
            in_dim=cond_dim,
            out_dim=Dim(6 * out_dim.dimension, name="adaLN_params"),
            with_bias=True,
        )
        self.adaLN_modulation.weight.initial = 0.0

    def __call__(self, x: Tensor, c: Tensor, *, spatial_dim: Dim) -> Tensor:
        params = self.adaLN_modulation(c)

        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = rf.split(
            params, axis=self.adaLN_modulation.out_dim, out_dims=[self.out_dim] * 6
        )

        x_skip = x
        x_sa_ln = self.norm1(x)
        x_sa_ln = modulate(x_sa_ln, shift_msa, scale_msa)
        x_sa = self.self_attn(x_sa_ln, axis=spatial_dim)
        x = rf.dropout(x_sa, drop_prob=self.att_dropout, axis=self.dropout_broadcast and self.out_dim)
        x = x * gate_msa
        x = x_skip + x

        x_skip = x
        x_ff_ln = self.norm2(x)
        x_ff_ln = modulate(x_ff_ln, shift_mlp, scale_mlp)
        x_ff = self.mlp(x_ff_ln)
        x = rf.dropout(x_ff, drop_prob=self.dropout, axis=self.dropout_broadcast and self.out_dim)
        x = x * gate_mlp
        x = x_skip + x

        return x


class DITFinalLayer(rf.Module):
    def __init__(
        self,
        hidden_dim: Dim = Dim(1024, name="dit-default-hidden-dim"),
        out_channels: Dim = Dim(1024, name="dit-default-out-channels"),
        *,
        cond_dim: Dim = Dim(128, name="dit-default-cond-dim"),
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.norm_final = rf.LayerNorm(hidden_dim, with_bias=False)
        self.linear = rf.Linear(hidden_dim, out_channels, with_bias=True)
        self.adaLN_modulation = rf.Linear(
            in_dim=cond_dim,
            out_dim=Dim(2 * hidden_dim.dimension, name="adaLN_params"),
            with_bias=True,
        )

        self.linear.weight.initial = 0.0
        self.adaLN_modulation.weight.initial = 0.0

    def __call__(self, x: Tensor, *, c: Tensor) -> Tensor:
        params = self.adaLN_modulation(c)
        (shift, scale) = rf.split(
            params, axis=self.adaLN_modulation.out_dim, out_dims=[self.hidden_dim, self.hidden_dim]
        )

        x_nm = self.norm_final(x)
        x = modulate(x_nm, shift, scale)
        x = self.linear(x)

        return x


class DIT(rf.Module):
    def __init__(
        self,
        vocab_dim: Dim,
        model_dim: Union[Dim, int] = 512,
        *,
        num_layers: int = 12,
        cond_dim: Union[Dim, int] = 128,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        att_dropout: float = 0.1,
    ):
        super().__init__()

        if isinstance(model_dim, int):
            model_dim = Dim(model_dim, name="model_dim")
        if isinstance(cond_dim, int):
            cond_dim = Dim(cond_dim, name="cond_dim")

        self.vocab_dim = vocab_dim
        self.model_dim = model_dim
        self.cond_dim = cond_dim
        self.out_dim = self.model_dim

        self.input_embedding = rf.Embedding(in_dim=vocab_dim, out_dim=model_dim)
        self.sigma_map = TimeEmbedder(hidden_dim=cond_dim)

        self.blocks = rf.Sequential(
            DITLayer(
                out_dim=model_dim,
                cond_dim=cond_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                att_dropout=att_dropout,
            )
            for _ in range(num_layers)
        )

        self.output_layer = DITFinalLayer(
            hidden_dim=model_dim,
            out_channels=vocab_dim,
            cond_dim=cond_dim,
        )

    def __call__(
        self,
        source: Tensor,
        sigma: Tensor,
        spatial_dim: Dim,
    ):
        x = self.input_embedding(source)
        c = rf.silu(self.sigma_map(sigma))
        x = self.blocks(x, c=c, spatial_dim=spatial_dim)
        x = self.output_layer(x, c=c)
        return x


vocab_dim = Dim(10241, name="vocab_with_mask")
returnn_model = DIT(
    vocab_dim=vocab_dim,
    model_dim=1024,
    num_layers=24,
    cond_dim=128,
    num_heads=16,
    dropout=0.0,
    att_dropout=0.0,
)

returnn_ckpt_path = "/home/ra717140/setups/exp2025-08-22/work/diffusion_lm_2025/jobs/extractplcheckpoint/ExtractPlCheckpointJob.50gnMVsLdg7J/output/new-checkpoint_v15.pt"
returnn_ckpt = torch.load(returnn_ckpt_path, map_location="cpu", weights_only=False)

print("Loading weights into RETURNN model...")

returnn_model.input_embedding.weight.raw_tensor.data = returnn_ckpt["input_embedding.weight"]

returnn_model.sigma_map.mlp.linear_ff.weight.raw_tensor.data = returnn_ckpt["sigma_map.mlp.linear_ff.weight"]
returnn_model.sigma_map.mlp.linear_ff.bias.raw_tensor.data = returnn_ckpt["sigma_map.mlp.linear_ff.bias"]

returnn_model.sigma_map.mlp.linear_out.weight.raw_tensor.data = returnn_ckpt["sigma_map.mlp.linear_out.weight"]
returnn_model.sigma_map.mlp.linear_out.bias.raw_tensor.data = returnn_ckpt["sigma_map.mlp.linear_out.bias"]

for i in range(24):
    prefix = f"blocks.{i}"
    returnn_model.blocks[i].norm1.scale.raw_tensor.data = returnn_ckpt[f"{prefix}.norm1.scale"]
    returnn_model.blocks[i].norm2.scale.raw_tensor.data = returnn_ckpt[f"{prefix}.norm2.scale"]
    returnn_model.blocks[i].self_attn.qkv.weight.raw_tensor.data = returnn_ckpt[f"{prefix}.self_attn.qkv.weight"]
    returnn_model.blocks[i].self_attn.proj.weight.raw_tensor.data = returnn_ckpt[f"{prefix}.self_attn.proj.weight"]
    returnn_model.blocks[i].mlp.linear_ff.weight.raw_tensor.data = returnn_ckpt[f"{prefix}.mlp.linear_ff.weight"]
    returnn_model.blocks[i].mlp.linear_ff.bias.raw_tensor.data = returnn_ckpt[f"{prefix}.mlp.linear_ff.bias"]
    returnn_model.blocks[i].mlp.linear_out.weight.raw_tensor.data = returnn_ckpt[f"{prefix}.mlp.linear_out.weight"]
    returnn_model.blocks[i].mlp.linear_out.bias.raw_tensor.data = returnn_ckpt[f"{prefix}.mlp.linear_out.bias"]
    returnn_model.blocks[i].adaLN_modulation.weight.raw_tensor.data = returnn_ckpt[f"{prefix}.adaLN_modulation.weight"]
    returnn_model.blocks[i].adaLN_modulation.bias.raw_tensor.data = returnn_ckpt[f"{prefix}.adaLN_modulation.bias"]

returnn_model.output_layer.norm_final.scale.raw_tensor.data = returnn_ckpt["output_layer.norm_final.scale"]
returnn_model.output_layer.linear.weight.raw_tensor.data = returnn_ckpt["output_layer.linear.weight"]
returnn_model.output_layer.linear.bias.raw_tensor.data = returnn_ckpt["output_layer.linear.bias"]
returnn_model.output_layer.adaLN_modulation.weight.raw_tensor.data = returnn_ckpt[
    "output_layer.adaLN_modulation.weight"
]
returnn_model.output_layer.adaLN_modulation.bias.raw_tensor.data = returnn_ckpt["output_layer.adaLN_modulation.bias"]

print("All weights loaded")
print("RETURNN model loaded")
print("=" * 80)

batch_size = 2
seq_len = 20

torch.manual_seed(42)
test_input = torch.randint(0, 10240, (batch_size, seq_len))
sigma_input = torch.zeros(batch_size)

print(f"Test input: {test_input[0, :10]}")

with torch.no_grad():
    test_input_gpu = test_input.to(device)
    sigma_input_gpu = sigma_input.to(device)

    x_pt = pytorch_model.vocab_embed(test_input_gpu)
    print("\n1. Embedding (PyTorch):")
    print(f"   norm={torch.norm(x_pt).item():.4f}, mean={x_pt.mean().item():.6f}")
    print(f"   sample: {x_pt[0, 0, :5].cpu()}")

    c_pt = F.silu(pytorch_model.sigma_map(sigma_input_gpu))
    print("\n2. Conditioning (PyTorch):")
    print(f"   norm={torch.norm(c_pt).item():.4f}, mean={c_pt.mean().item():.6f}")
    print(f"   sample: {c_pt[0, :5].cpu()}")

with torch.no_grad():
    batch_dim = Dim(batch_size, name="batch")
    spatial_dim = Dim(seq_len, name="spatial")

    test_input_rf = rf.convert_to_tensor(
        test_input.numpy(), dims=[batch_dim, spatial_dim], sparse_dim=vocab_dim, dtype="int32"
    )

    x_rf = returnn_model.input_embedding(test_input_rf)
    x_rf_np = x_rf.raw_tensor.detach().cpu().numpy()
    print("\n1. Embedding (RETURNN):")
    print(f"   norm={np.linalg.norm(x_rf_np):.4f}, mean={x_rf_np.mean():.6f}")
    print(f"   sample: {x_rf_np[0, 0, :5]}")

    sigma_rf = rf.convert_to_tensor(sigma_input.numpy(), dims=[batch_dim], dtype="float32")
    c_rf = rf.silu(returnn_model.sigma_map(sigma_rf))
    c_rf_np = c_rf.raw_tensor.detach().cpu().numpy()
    print("\n2. Conditioning (RETURNN):")
    print(f"   norm={np.linalg.norm(c_rf_np):.4f}, mean={c_rf_np.mean():.6f}")
    print(f"   sample: {c_rf_np[:5]}")

    diff_emb = np.abs(x_pt.cpu().numpy() - x_rf_np).max()
    print(f"\n   Embedding diff: {diff_emb:.6e}")

    diff_c = np.abs(c_pt.cpu().numpy()[0] - c_rf_np).max()
    print(f"   Conditioning diff: {diff_c:.6e}")

print("=" * 80)

print("\n" + "=" * 80)
print("DETAILED BLOCK 0 COMPARISON")
print("=" * 80)

with torch.no_grad():
    batch_dim = Dim(batch_size, name="batch")
    spatial_dim = Dim(seq_len, name="spatial")
    test_input_rf = rf.convert_to_tensor(
        test_input.numpy(), dims=[batch_dim, spatial_dim], sparse_dim=vocab_dim, dtype="int32"
    )
    sigma_rf = rf.convert_to_tensor(sigma_input.numpy(), dims=[batch_dim], dtype="float32")

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        x_pt = pytorch_model.vocab_embed(test_input_gpu)
        c_pt = F.silu(pytorch_model.sigma_map(sigma_input_gpu))

    x_rf = returnn_model.input_embedding(test_input_rf)
    c_rf = rf.silu(returnn_model.sigma_map(sigma_rf))

    print("\n1. adaLN modulation outputs:")
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        params_pt = pytorch_model.blocks[0].adaLN_modulation(c_pt)
    params_rf = returnn_model.blocks[0].adaLN_modulation(c_rf)

    params_pt_np = params_pt.cpu().float().numpy()
    params_rf_np = params_rf.raw_tensor.cpu().numpy()

    print(f"PyTorch: shape={params_pt_np.shape}, sample={params_pt_np[0, :5]}")
    print(f"RETURNN:  shape={params_rf_np.shape}, sample={params_rf_np[:5]}")
    print(f"Diff: {np.abs(params_pt_np[0] - params_rf_np).max():.6e}")

    print("\n2. After norm1:")
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        x_norm1_pt = pytorch_model.blocks[0].norm1(x_pt)
    x_norm1_rf = returnn_model.blocks[0].norm1(x_rf)

    x_norm1_pt_np = x_norm1_pt.cpu().float().numpy()
    x_norm1_rf_np = x_norm1_rf.raw_tensor.cpu().numpy()

    print(f"  PyTorch: norm={np.linalg.norm(x_norm1_pt_np):.4f}, sample={x_norm1_pt_np[0, 0, :5]}")
    print(f"  RETURNN:  norm={np.linalg.norm(x_norm1_rf_np):.4f}, sample={x_norm1_rf_np[0, 0, :5]}")
    print(f"  Diff: {np.abs(x_norm1_pt_np - x_norm1_rf_np).max():.6e}")

    print("\n3. QKV weight comparison:")
    qkv_pt = pytorch_model.blocks[0].attn_qkv.weight.cpu().float().numpy()
    qkv_rf = returnn_model.blocks[0].self_attn.qkv.weight.raw_tensor.cpu().numpy()

    print(f"  PyTorch QKV: shape={qkv_pt.shape}, norm={np.linalg.norm(qkv_pt):.4f}")
    print(f"  RETURNN QKV:  shape={qkv_rf.shape}, norm={np.linalg.norm(qkv_rf):.4f}")

    qkv_pt_T = qkv_pt.T
    print(f"  Diff (with transpose): {np.abs(qkv_pt_T - qkv_rf).max():.6e}")

    print("\n4. After QKV projection:")
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = params_pt.chunk(6, dim=-1)
        x_modulated_pt = x_norm1_pt * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        qkv_pt_out = pytorch_model.blocks[0].attn_qkv(x_modulated_pt)

    (shift_msa_rf, scale_msa_rf, gate_msa_rf, shift_mlp_rf, scale_mlp_rf, gate_mlp_rf) = rf.split(
        params_rf, axis=returnn_model.blocks[0].adaLN_modulation.out_dim, out_dims=[returnn_model.blocks[0].out_dim] * 6
    )

    shift_msa_rf = rf.expand_dim(shift_msa_rf, dim=spatial_dim)
    scale_msa_rf = rf.expand_dim(scale_msa_rf, dim=spatial_dim)

    x_modulated_rf = x_norm1_rf * (1 + scale_msa_rf) + shift_msa_rf
    qkv_rf_out = returnn_model.blocks[0].self_attn.qkv(x_modulated_rf)

    qkv_pt_out_np = qkv_pt_out.cpu().float().numpy()
    qkv_rf_out_np = qkv_rf_out.raw_tensor.cpu().numpy()

    print(f"  PyTorch QKV output: shape={qkv_pt_out_np.shape}, norm={np.linalg.norm(qkv_pt_out_np):.4f}")
    print(f"                      sample={qkv_pt_out_np[0, 0, :5]}")
    print(f"  RETURNN QKV output:  shape={qkv_rf_out_np.shape}, norm={np.linalg.norm(qkv_rf_out_np):.4f}")
    print(f"                      sample={qkv_rf_out_np[0, 0, :5]}")
    print(f"  Diff: {np.abs(qkv_pt_out_np - qkv_rf_out_np).max():.6e}")

print("=" * 80)

print("\n" + "=" * 80)
print("FULL FORWARD PASS")
print("=" * 80)

with torch.no_grad():
    pytorch_output = pytorch_model(test_input_gpu, sigma_input_gpu).cpu().float()
    pytorch_log_probs = F.log_softmax(pytorch_output, dim=-1)
    pytorch_nll = -pytorch_log_probs[0, 0, test_input[0, 0]]

    print("\nPyTorch:")
    print(f"  Output sample: {pytorch_output[0, 0, :5].numpy()}")
    print(f"  NLL: {pytorch_nll.item():.4f}")

with torch.no_grad():
    returnn_output = returnn_model(test_input_rf, sigma=sigma_rf, spatial_dim=spatial_dim)
    returnn_output_np = returnn_output.raw_tensor.detach().cpu().numpy()
    returnn_log_probs = F.log_softmax(torch.from_numpy(returnn_output_np), dim=-1)
    returnn_nll = -returnn_log_probs[0, 0, test_input[0, 0]]

    print("\nRETURNN:")
    print(f"  Output sample: {returnn_output_np[0, 0, :5]}")
    print(f"  NLL: {returnn_nll.item():.4f}")

diff = np.abs(pytorch_output.numpy() - returnn_output_np).max()
print(f"\nMax logit difference: {diff:.6e}")

if diff < 1e-2:
    print("\n✓✓✓ MODELS MATCH! ✓✓✓")
else:
    print("\n✗✗✗ MODELS DIFFER! ✗✗✗")

print("=" * 80)
# ============================================================================
# TEST ON REAL TEXT
# ============================================================================
print("\n" + "=" * 80)
print("TEST ON REAL TEXT")
print("=" * 80)

# Load SentencePiece tokenizer
import sentencepiece as spm

spm_model_path = "/home/ra717140/setups/exp2025-08-22/output/datasets/LibriSpeech/vocab/spm_unigram_10k_train.model"
sp = spm.SentencePieceProcessor()
sp.Load(spm_model_path)

# Test text (без ▁ символов - SentencePiece сам их добавит)
test_text = "_AND _AFTER _THIS _THERE _WERE _NO _MORE _QUIET _DAYS _FOR _CHARLOTTE _CHATTER TON"

print(f"\nTest text: {test_text}")

# Tokenize
test_token_ids = sp.EncodeAsIds(test_text)
test_tokens_str = sp.EncodeAsPieces(test_text)

print(f"Tokens: {test_tokens_str}")
print(f"Token IDs: {test_token_ids}")

# BOS/EOS
bos_id = sp.bos_id()  # Обычно 1
eos_id = sp.eos_id()  # Обычно 2

test_token_ids_full = [bos_id] + test_token_ids + [eos_id]

print(f"BOS ID: {bos_id}, EOS ID: {eos_id}")
print(f"With BOS/EOS: {test_token_ids_full}")
print(f"Length: {len(test_token_ids_full)}")

# Create tensors
test_input_real = torch.tensor([test_token_ids_full], dtype=torch.long)  # [1, T]
sigma_input_real = torch.zeros(1)

# ============================================================================
# PYTORCH INFERENCE
# ============================================================================
print("\n" + "-" * 80)
print("PyTorch Model")
print("-" * 80)

with torch.no_grad():
    test_input_gpu = test_input_real.to(device)
    sigma_input_gpu = sigma_input_real.to(device)

    # Forward pass
    pytorch_output = pytorch_model(test_input_gpu, sigma_input_gpu).cpu().float()
    pytorch_log_probs = F.log_softmax(pytorch_output, dim=-1)

    # Compute NLL for each token
    pytorch_token_nlls = []
    for i, token_id in enumerate(test_token_ids_full):
        nll = -pytorch_log_probs[0, i, token_id]
        pytorch_token_nlls.append(nll.item())
        if i == 0:
            token_str = "<BOS>"
        elif i == len(test_token_ids_full) - 1:
            token_str = "<EOS>"
        else:
            token_str = test_tokens_str[i - 1]  # -1 because BOS shifts indices
        print(f"  Token {i:2d} [{token_str:15s}] (ID:{token_id:4d}) NLL: {nll.item():.4f}")

    avg_nll = np.mean(pytorch_token_nlls)
    perplexity = np.exp(avg_nll)

    print(f"\n  Average NLL: {avg_nll:.4f}")
    print(f"  Perplexity: {perplexity:.2f}")

# ============================================================================
# RETURNN INFERENCE
# ============================================================================
print("\n" + "-" * 80)
print("RETURNN Model")
print("-" * 80)

with torch.no_grad():
    batch_dim_real = Dim(1, name="batch")
    spatial_dim_real = Dim(len(test_token_ids_full), name="spatial")

    test_input_rf = rf.convert_to_tensor(
        test_input_real.numpy(),
        dims=[batch_dim_real, spatial_dim_real],
        sparse_dim=vocab_dim,
        dtype="int32",
    )

    # Forward pass
    sigma_rf_real = rf.convert_to_tensor(sigma_input_real.numpy(), dims=[batch_dim_real], dtype="float32")
    returnn_output = returnn_model(test_input_rf, sigma=sigma_rf_real, spatial_dim=spatial_dim_real)
    returnn_output_np = returnn_output.raw_tensor.detach().cpu().numpy()
    returnn_log_probs = F.log_softmax(torch.from_numpy(returnn_output_np), dim=-1)

    # Compute NLL for each token
    returnn_token_nlls = []
    for i, token_id in enumerate(test_token_ids_full):
        nll = -returnn_log_probs[0, i, token_id]
        returnn_token_nlls.append(nll.item())
        if i == 0:
            token_str = "<BOS>"
        elif i == len(test_token_ids_full) - 1:
            token_str = "<EOS>"
        else:
            token_str = test_tokens_str[i - 1]
        print(f"  Token {i:2d} [{token_str:15s}] (ID:{token_id:4d}) NLL: {nll.item():.4f}")

    avg_nll = np.mean(returnn_token_nlls)
    perplexity = np.exp(avg_nll)

    print(f"\n  Average NLL: {avg_nll:.4f}")
    print(f"  Perplexity: {perplexity:.2f}")

# ============================================================================
# COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

print("\nToken-by-token NLL difference:")
for i, (pt_nll, rf_nll) in enumerate(zip(pytorch_token_nlls, returnn_token_nlls)):
    if i == 0:
        token_str = "<BOS>"
    elif i == len(test_token_ids_full) - 1:
        token_str = "<EOS>"
    else:
        token_str = test_tokens_str[i - 1]
    diff = abs(pt_nll - rf_nll)
    print(f"  Token {i:2d} [{token_str:15s}] PyTorch: {pt_nll:.4f}, RETURNN: {rf_nll:.4f}, Diff: {diff:.4f}")

avg_diff = np.mean([abs(a - b) for a, b in zip(pytorch_token_nlls, returnn_token_nlls)])
max_diff = max([abs(a - b) for a, b in zip(pytorch_token_nlls, returnn_token_nlls)])

print(f"\nAverage NLL difference: {avg_diff:.4f}")
print(f"Max NLL difference: {max_diff:.4f}")

if avg_diff < 0.1:
    print("\n✓✓✓ MODELS MATCH ON REAL TEXT! ✓✓✓")
else:
    print("\n✗✗✗ MODELS DIFFER ON REAL TEXT! ✗✗✗")

print("=" * 80)
