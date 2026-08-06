# Models 目录指南

本目录存放 TRELLIS.2 image-to-3D 全流程推理所需的 GGUF 模型文件。所有文件
均由上游 safetensors 权重转换而来（`scripts/` 下的 convert/download 脚本），
是本仓库运行时的**本地生成产物**，已被 `.gitignore`（`/models/`、`*.gguf`）
排除，不进入 git 历史。

> **重要**：`models/` 下的文件应当是本地的**真实文件**，不要使用指向仓库外部
> 的软链接（symlink）。外部路径一旦被移动/删除，软链会断裂导致推理失败。
> 如果是从其他仓库直接复用模型，请用 `cp` 拷贝进来，而不是 `ln -s`。

---

## 1. 一图看懂：模型在 pipeline 中的位置

```
image (RGB/RGBA)
  → [RMBG-2.0]          rmbg_f16/f32/q8.gguf        背景去除（可选）
  → [预处理]            （无模型，纯 C++）
  → [DINOv3]            dino_f16/q8.gguf            图像 → 1029×1024 条件 token
  → [SS-flow DiT]       ss_flow_f16/q8.gguf         稀疏结构流 → z_s
  → [SS decoder]        ss_dec_f16/q8.gguf          3D-conv → 64³ occupancy
  → [shape-SLAT DiT]    slat_flow[_1024]_f16/q8.gguf 稀疏形状流（512/1024）
  → [shape VAE decoder] shape_dec_f16.gguf           稀疏 ConvNeXt U-Net → dual grid
  ├→ [网格化]           （无模型，C++）
  └→ [shape VAE enc]    shape_enc_f16.gguf          重建的 dual grid → shape SLat
     → [tex-SLAT DiT]   tex_slat_flow_512/1024_*.gguf 纹理流
     → [texture decoder]tex_dec_f16.gguf            稀疏 6 通道 PBR volume
  → [材质采样]          （无模型，C++）
```

---

## 2. 模型清单总表

| 模型文件 | 大小 | 参数量 | GGUF 架构 | 精度变体 |
|---------|------|--------|-----------|----------|
| `dino_f16.gguf` | 579 MB | 303.1M | trellis2-dino | f16 |
| `dino_q8.gguf` | 309 MB | 303.1M | trellis2-dino | q8 |
| `rmbg_f16.gguf` | 421 MB | 220.7M | rmbg (swin_v1_l) | f16 |
| `rmbg_f32.gguf` | 842 MB | 220.7M | rmbg (swin_v1_l) | f32 |
| `rmbg_q8.gguf` | 247 MB | 220.7M | rmbg (swin_v1_l) | q8 |
| `ss_flow_f16.gguf` | 2494 MB | 1.29B | trellis2-ss-flow | f16 |
| `ss_flow_q8.gguf` | 1353 MB | 1.29B | trellis2-ss-flow | q8 |
| `ss_dec_f16.gguf` | 141 MB | 73.7M | trellis2-ss-dec | f16 |
| `ss_dec_q8.gguf` | 141 MB | 73.7M | trellis2-ss-dec | q8 |
| `slat_flow_f16.gguf` | 2494 MB | 1.29B | trellis2-slat-flow | f16 |
| `slat_flow_q8.gguf` | 1353 MB | 1.29B | trellis2-slat-flow | q8 |
| `slat_flow_1024_f16.gguf` | 2508 MB | 1.29B | trellis2-slat-flow | f16 |
| `slat_flow_1024_q8.gguf` | 1353 MB | 1.29B | trellis2-slat-flow | q8 |
| `shape_dec_f16.gguf` | 905 MB | 474.2M | trellis2-shape-dec | f16 |
| `shape_enc_f16.gguf` | 676 MB | 354.4M | trellis2-shape-enc | f16 |
| `tex_dec_f16.gguf` | 905 MB | 474.2M | trellis2-tex-dec | f16 |
| `tex_slat_flow_512_f16.gguf` | 2494 MB | 1.29B | trellis2-slat-flow | f16 |
| `tex_slat_flow_512_q8.gguf` | 1353 MB | 1.29B | trellis2-slat-flow | q8 |
| `tex_slat_flow_1024_f16.gguf` | 2494 MB | 1.29B | trellis2-slat-flow | f16 |
| `tex_slat_flow_1024_q8.gguf` | 1353 MB | 1.29B | trellis2-slat-flow | q8 |

> 参数量为从 GGUF 张量形状累加得到的近似值（`~params`）。

---

## 3. 逐模型详解

### 3.1 DINOv3 ViT-L/16 图像编码器 — `dino_f16.gguf` / `dino_q8.gguf`

- **架构**：`dinov3-vitl16-pretrain-lvd1689m`，hidden=1024，24 层，16 头，
  intermediate=4096，patch=16，4 个 register tokens，RoPE（θ=100）。
- **作用**：把预处理后的 512×512 图像编码为 `[1, 1029, 1024]` 条件 token
  （1 CLS + 4 register + 1024 patch，取自最后一层、去掉 affine LN）。
  这是后续所有 DiT 流（SS-flow / shape-SLAT / tex-SLAT）的**视觉条件**。
- **输入/输出**：图像 → 1029 个 1024 维 token。
- **Benchmark**（RTX 3060, CUDA）：约 **1s**。
- **注意事项**：
  - Q8 变体（309MB）质量损失极小，推荐默认使用。
  - 条件 token 与后续 F32 计算的张量 concat，因此 token 相关权重保留 f32。

### 3.2 RMBG-2.0 背景去除 — `rmbg_f16.gguf` / `rmbg_f32.gguf` / `rmbg_q8.gguf`

- **架构**：`rmbg`，backbone = Swin-Transformer-Large（`swin_v1_l`），
  输入 1024×1024，BiRefNet 式的 alpha 分割。
- **作用**：去除复杂背景（非纯色背景图），输出 feathered alpha，供后续
  pipeline 只对主体重建 3D。对纯色背景不是必需；`--rmbg` 显式传入才启用。
- **Benchmark**（RTX 3060, batch=1, 1024×1024, 5 次稳态均值）：
  | 模型 | CUDA | Vulkan |
  |------|------|--------|
  | `rmbg_f32.gguf` | 644.5 ms | 1293.4 ms |
  | `rmbg_f16.gguf` | 655.2 ms | 1278.5 ms |
  - max alpha abs diff = 1.122e-4（f32/f16 一致）。
- **注意事项**：
  - **f16 是部署默认**（体积减半，精度与 f32 几乎一致）。
  - **Q8 不推荐**：官方实测 Q8 相对 f16 只省 16.9 MiB，且 CUDA/Vulkan 均未
    提速（648.7 ms / 1278.5 ms）；全量 Q8 还会超出 `2e-3` alpha 精度门限。
  - 密集负载下建议用 `--rmbg-device cpu` 推理，把 VRAM 留给 TRELLIS 主模型。

### 3.3 SS-flow DiT（稀疏结构流）— `ss_flow_f16.gguf` / `ss_flow_q8.gguf`

- **架构**：`trellis2-ss-flow`，resolution=16（16³=4096 token），in/out=8，
  model_channels=1536，cond_channels=1024，30 层，12 头，mlp_ratio=5.33，
  RoPE，share_mod，qk_rms_norm。
- **作用**：稀疏结构 stage 的 flow 模型。12-step CFG flow-Euler 采样 → 稀疏
  结构潜变量 `z_s`（8 通道）。
- **Benchmark**（RTX 3060, CUDA）：**~19.8s**（与 PyTorch CUDA 19.76s 对齐，
  差异 <0.2%，证明 ggml matmul 性能等价）。
- **注意事项**：
  - Q8 变小（1353MB），速度几乎无损（内存带宽需求下降抵消 dequant 开销）。

### 3.4 SS decoder（稀疏结构解码器）— `ss_dec_f16.gguf` / `ss_dec_q8.gguf`

- **架构**：`trellis2-ss-dec`，latent_channels=8，out_channels=1，3 级
  （channels 512/128/32），密集 3D conv。
- **作用**：把 `z_s` 解码为 **64³ occupancy logits**，再得到 32³ voxel
  scaffold，作为 shape-SLAT 的稀疏体素骨架。
- **Benchmark**：该模型体积小，推理极快（毫秒级）。
- **注意事项**：Q8 变体大小与 f16 相同（140.6MB），因为 3D conv kernel
  `ne[0]=3` 不满足 ggml 对齐约束，实际保持原精度。

### 3.5 Shape-SLAT DiT（形状稀疏流）— `slat_flow_f16/q8.gguf`（512）与 `slat_flow_1024_f16/q8.gguf`（1024）

- **架构**：`trellis2-slat-flow`，in/out=32，model_channels=1536，
  cond_channels=1024，30 层，12 头，mlp_ratio=5.33，RoPE，share_mod，
  qk_rms_norm。
  - **512 版**：resolution=32，生成 512³ 网格（~1M 顶点）。
  - **1024 版**：resolution=64，1024 级联，生成 ~5M 顶点高精度网格。
- **作用**：在 32³ 稀疏体素（scaffold）上进行 512/1024 稀疏 shape 流，12-step
  CFG 采样得到 shape SLat 潜变量。
- **Benchmark**（RTX 3060, CUDA）：512 版 shape-SLAT 采样 **~7s**（PyTorch
  17.37s，ggml 快 2.5x）。
- **注意事项**：
  - 1024 版 HR token（~49k）仅能通过 flash attention（`sdpa_auto`）在 VRAM 内跑。
  - 需要 DINOv3 在 1024 分辨率下编码（1024 级联逻辑）。

### 3.6 Shape VAE 解码器 — `shape_dec_f16.gguf`

- **架构**：`trellis2-shape-dec`（next_dc），latent_channels=32，
  out_channels=7，5 级（channels 1024/512/256/128/64，blocks 4/16/8/4/0），
  稀疏 ConvNeXt U-Net，16× up。
- **作用**：把 shape SLat 解码为 7 通道 dual grid（含 occupancy + 特征），并
  输出 subdivision 引导。这是**性能关键瓶颈**——ggml CUDA 仅 **9.8s**，比
  PyTorch CPU 的 198s 快 **20x**。
- **注意事项**：
  - **只有 f16 版本，无 Q8**：稀疏 subdivision 和网格几何对 Q8 权重舍入不鲁棒，
    必须保持 f16（非 CUDA 限制——CUDA 已支持 Q8 copy）。

### 3.7 Shape VAE 编码器 — `shape_enc_f16.gguf`

- **架构**：`trellis2-shape-enc`（next_dc），in_channels=6，latent_channels=32，
  5 级（channels 64/128/256/512/1024，blocks 0/4/8/16/4），稀疏 U-Net。
- **作用**：把重建/验证过的 dual grid 编码回 shape SLat + subdivision guide，
  供纹理 stage 作为 shape 条件。
- **注意事项**：**只有 f16 版本**（理由同 shape_dec，精度敏感）。

### 3.8 Tex-SLAT DiT（纹理稀疏流）— `tex_slat_flow_512_f16/q8.gguf` 与 `tex_slat_flow_1024_f16/q8.gguf`

- **架构**：`trellis2-slat-flow`，in_channels=64，out_channels=32，
  concat_cond_channels=32（把 shape SLat 拼接进来），model_channels=1536，
  cond_channels=1024，30 层，12 头，mlp_ratio=5.33，RoPE，share_mod，
  qk_rms_norm。
  - **512 版**：resolution=32。
  - **1024 版**：resolution=64。
- **作用**：以 shape-SLat 为条件，采样纹理 SLat 潜变量。
- **注意事项**：1024 版 HR 同样依赖 flash attention。

### 3.9 纹理解码器 — `tex_dec_f16.gguf`

- **架构**：`trellis2-tex-dec`（next_dc），latent_channels=32, out_channels=6，
  5 级（channels 1024/512/256/128/64，blocks 4/16/8/4/0），稀疏 U-Net。
- **作用**：把纹理 SLat replay subdivision 后解码为稀疏 **6 通道 PBR volume**
  （base color + metallic + roughness + normal + 等）。
- **注意事项**：**只有 f16 版本**（无 Q8，精度敏感）。

---

## 4. 推理时间总览（端到端）

以下为 TRELLIS.2-4B 在 **RTX 3060 12GB / Ryzen 9 7950X / 64GB RAM** 上的实测
（详见 [BENCHMARK.md](BENCHMARK.md)）：

### 4.1 总体耗时（后端对比）

| Backend | 总耗时 | 相比 PyTorch |
|---------|--------|--------------|
| PyTorch CUDA | 317.9 s | 基准 |
| **ggml CUDA** | **142.7 s** | **快 2.2x** |
| ggml Vulkan | 176.8 s | 快 1.8x |
| ggml CPU | 2652.6 s | 慢 8.3x |

### 4.2 分阶段耗时（PyTorch vs ggml CUDA）

| 阶段 | 涉及模型 | PyTorch | ggml CUDA | 差异 |
|------|---------|---------|-----------|------|
| 模型加载 | 全部 | 81.4 s | ~2 s | ggml 快 40x |
| DINO 编码 | `dino_*` | 1.2 s | ~1 s | 相近 |
| SS-flow 采样 | `ss_flow_*` | 19.76 s | ~19.8 s | 完全一致 |
| Shape SLAT | `slat_flow_*` | 17.37 s | ~7 s | ggml 快 2.5x |
| Shape decode | `shape_dec` | 198 s (CPU) | 9.8 s (CUDA) | ggml 快 20x |
| **总计** | | **317.9 s** | **142.7 s** | **快 2.2x** |

### 4.3 Q8 量化对推理的影响

| 配置 | Backend | 耗时 | 顶点数 |
|------|---------|------|--------|
| F16 quality=512 | CUDA | 142.7 s | 1,884,862 |
| **Q8 quality=512** | **CUDA** | **129.0 s** | 1,824,032（−3.2%） |
| F16 coarse | CUDA | 46.9 s | 83,405 |
| **Q8 coarse** | **CUDA** | **44.4 s** | 82,722（−0.8%） |

Q8 比 F16 快约 10%（内存带宽需求减半），体积减 73%，几何质量损失 <3.5%。

---

## 5. 每个模型的精度验证

各模型已通过 tap-by-tap 与 PyTorch 参考对齐（详见
[VERIFICATION.md](VERIFICATION.md)）：

| 模型 | 测试 | 结果 |
|------|------|------|
| `dino_*` | `test_dino`（40 taps） | **rel-L2 ≤ 7e-7** |
| `ss_flow_*` | `test_ss_flow_forward` | rel-L2 2.4e-4 |
| `ss_flow_*` | `test_ss_sample`（12-step CFG） | rel-L2 5.7e-3 |
| `ss_dec_*` | `test_ss_dec` | rel-L2 2e-5 |
| `slat_flow_*` | `test_slat` | rel-L2 2.9e-4 (CPU) / 8e-4 (GPU) |
| `shape_dec` | `test_slat`（5 级全链路） | rel-L2 ≤ 6e-7 |
| `slat_flow_1024` | `test_cascade` | rel-L2 ~3e-4 (CPU) |
| `tex_*` | 纹理/材质回归 | 稀疏材质采样回归通过 |

---

## 6. 使用注意事项汇总

1. **f16 vs Q8**：可量化模型（dino、ss_flow、slat_flow、tex_slat_flow、ss_dec）
   推荐 Q8（体积小、速度无损、质量损失 <3.5%）；**精度敏感解码器
   （shape_dec、shape_enc、tex_dec）只有 f16**，不要量化。
2. **RMBG**：默认用 `rmbg_f16.gguf`；`rmbg_q8.gguf` 不推荐（无收益）；
   密集负载用 `--rmbg-device cpu` 省 VRAM。
3. **软链**：不要用指向仓库外部的软链，直接 `cp` 真实文件进来。
4. **VRAM 预算**：全 f16 约 16GB（12GB GPU OOM）；Q8 组合约 4.3GB 可跑。
5. **1024 级联**：HR 模型（`slat_flow_1024`、`tex_slat_flow_1024`）依赖 flash
   attention，且会增加 ~5min / ~10GB VRAM / ~14GB 主机内存尖峰。

---

## 7. 如何获取/重新生成

- 一键下载预构建 f16：`scripts/download_ggufs.sh`（落到 `ggufs/`）。
- 从 safetensors 转换：`scripts/download_models.sh` + `scripts/convert_all.sh`。
- 量化到 Q8：`python3 quantize_to_q8.py --batch models/ models/ --skip shape_dec
  --skip tex_dec --skip shape_enc`。
- RMBG 模型转换：`third_party/RMBG-2.0-GGML/scripts/convert_rmbg_to_gguf.py`。