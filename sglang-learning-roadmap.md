# SGLang 框架学习路线

> 本文档整理自对 sglang 仓库源码结构的梳理，按「会用 → 懂架构 → 会改代码」三个阶段组织。
> 所有代码入口均为仓库内的相对路径，可直接点击跳转。

## 一句话认识它

SGLang 是一个高性能 LLM 推理服务框架：**前端一个 `lang` DSL（结构化管理 prompt），后端一个 `srt` 运行时（RadixAttention 前缀缓存 + 连续批处理 + 多种 attention 后端）**。

仓库本体：

- `python/sglang/srt/` — Python 运行时（核心）
- `sgl-kernel/` — C++/CUDA 内核
- `rust/` — grpc 服务
- `docs_new/` — 新版官方文档（比 `docs/` 旧目录新）
- `benchmark/` — 基准测试脚本

---

## 阶段一：前置知识（不涉及代码）

学代码之前先补 LLM 推理的基础概念，否则看 `scheduler.py` 会一头雾水：

- Transformer 架构与 **KV Cache**：prefill（计算密集）vs decode（访存密集）
- **连续批处理（continuous batching）**：比静态 batching 吞吐高的原理
- **CUDA Graph**：为什么解码小 kernel 要 graph capture
- 并行策略：**TP / EP / PP / DP** 各解决什么问题
- 前缀缓存（prefix caching）为什么能省 prefill 计算

---

## 阶段二：会用（1 天）

先把服务跑起来，建立「请求怎么进怎么出」的直觉：

1. [quickstart](docs_new/docs/get-started/quickstart.mdx) + [install](docs_new/docs/get-started/install.mdx) — 启动 `python -m sglang.launch_server --model ...`
2. [openai_api](docs_new/docs/basic_usage/openai_api.mdx)、[native_api](docs_new/docs/basic_usage/native_api.mdx) — 三种调用方式
3. [offline_engine_api](docs_new/docs/basic_usage/offline_engine_api.mdx) — 不走 HTTP 的 `Engine` 直连，**这是读代码前最好的热身**，能直接单步调试
4. [sampling_params](docs_new/docs/basic_usage/sampling_params.mdx) — 采样参数全解
5. 跑一遍 [benchmark/](benchmark/) 下的 `bench_serving.py` / `bench_offline_throughput.py`，知道性能指标长什么样

---

## 阶段三：架构地图（核心）

这是最重要的一步。理解**一次请求的生命周期**，把所有模块串成一条线：

```
客户端 → HTTP server (entrypoints/http_server.py)
       → TokenizerManager (tokenizer_manager.py)   # 全局入口，路由到对应 worker
       → Scheduler (scheduler.py)                  # 批处理核心，攒 batch、调度
       → ModelRunner (model_runner.py)             # 真正跑 forward + CUDA Graph
       → 结果返回 → Detokenizer → 客户端
```

推荐的阅读顺序（配合上面的图）：

1. **入口**：[entrypoints/http_server.py](python/sglang/srt/entrypoints/http_server.py) — 看 `/generate` 怎么进来
2. **请求解析**：[tokenizer_manager.py](python/sglang/srt/managers/tokenizer_manager.py)（3100+ 行，看主流程即可，别逐行）
3. **调度器**：[scheduler.py](python/sglang/srt/managers/scheduler.py) + [schedule_policy.py](python/sglang/srt/managers/schedule_policy.py) + [schedule_batch.py](python/sglang/srt/managers/schedule_batch.py) — 这是 SGLang 的「大脑」
4. **模型执行**：[model_runner.py](python/sglang/srt/model_executor/model_runner.py)

---

## 阶段四：六大核心模块深挖（改代码前必读）

### 1. KV Cache 与 RadixAttention（SGLang 的招牌）

- [mem_cache/radix_cache.py](python/sglang/srt/mem_cache/radix_cache.py) — Radix Tree 前缀缓存，SGLang 起家的论文核心
- [mem_cache/unified_memory_pool.py](python/sglang/srt/mem_cache/unified_memory_pool.py) — 统一内存池，KV 与中间激活统一管理
- [mem_cache/chunk_cache.py](python/sglang/srt/mem_cache/chunk_cache.py) — 分块缓存
- 进阶：[advanced_features/hicache.mdx](docs_new/docs/advanced_features/hicache.mdx) 分层缓存
- 配套看 [advanced_features/overview.mdx](docs_new/docs/advanced_features/overview.mdx)

### 2. 调度与批处理

- `scheduler.py` 里的主循环：攒请求 → 调 `get_new_batch_prefill` / `get_new_batch_decode`
- **抢占（preemption）**：swap / recompute 两种策略，看 [mem_cache/evict_policy.py](python/sglang/srt/mem_cache/evict_policy.py) 和 [mem_cache/allocator/](python/sglang/srt/mem_cache/allocator/)
- 新版调度器子组件在 [managers/scheduler_components/](python/sglang/srt/managers/scheduler_components/)（request_receiver、output_sender、kv_events_publisher…）

### 3. Model Runner 与 CUDA Graph

- [model_runner.py](python/sglang/srt/model_executor/model_runner.py) — forward 入口，prefill/decode 分流
- [cpu_graph_runner.py](python/sglang/srt/model_executor/cpu_graph_runner.py) — CUDA Graph 捕获/重放逻辑
- [forward_batch_info.py](python/sglang/srt/model_executor/forward_batch_info.py) — 一次 forward 的 batch 元数据
- [runner_backend/](python/sglang/srt/model_executor/runner_backend/) 下的各后端

### 4. Attention Backends（多后端架构）

- 注册机制：[layers/attention/attention_registry.py](python/sglang/srt/layers/attention/attention_registry.py)
- 重点读 3 个：**FlashAttention**（[flashattention_backend.py](python/sglang/srt/layers/attention/flashattention_backend.py)）、**FlashInfer**（[flashinfer_backend.py](python/sglang/srt/layers/attention/flashinfer_backend.py)）、**Triton**（[triton_backend.py](python/sglang/srt/layers/attention/triton_backend.py)）
- MLA 变体（DeepSeek 系）看 [flashinfer_mla_backend.py](python/sglang/srt/layers/attention/flashinfer_mla_backend.py)
- 选择逻辑看 [advanced_features/attention_backend.mdx](docs_new/docs/advanced_features/attention_backend.mdx)

### 5. 模型实现层

- [layers/](python/sglang/srt/layers/) 是算子的「积木箱」：linear、moe、quantization、rotary_embedding、sampler
- [models/](python/sglang/srt/models/) 下 200+ 个模型文件，挑一个熟悉的模型（如 `deepseek_v3.py` 或 `llama.py`）通读一遍，理解「模型 = 层组合 + 配置解析」

### 6. 分布式

- [distributed/parallel_state.py](python/sglang/srt/distributed/parallel_state.py) — TP/EP/PP/DP 通信组
- [tp_worker.py](python/sglang/srt/managers/tp_worker.py) — 每个 GPU 进程的 worker
- [data_parallel_controller.py](python/sglang/srt/managers/data_parallel_controller.py) — DP 负载均衡

---

## 阶段五：高级特性（按需选学）

| 特性 | 代码入口 | 文档 |
|---|---|---|
| **投机解码**（EAGLE/NGram/MTP） | [speculative/](python/sglang/srt/speculative/) | [adaptive_speculative_decoding.mdx](docs_new/docs/advanced_features/adaptive_speculative_decoding.mdx) |
| **PD 分离**（prefill/decode 拆开） | [disaggregation/](python/sglang/srt/disaggregation/) | [pd_disaggregation.mdx](docs_new/docs/advanced_features/pd_disaggregation.mdx) |
| **专家并行（EP）** | [layers/moe/](python/sglang/srt/layers/moe/)、[elastic_ep/](python/sglang/srt/elastic_ep/) | [expert_parallelism.mdx](docs_new/docs/advanced_features/expert_parallelism.mdx) |
| **LoRA** | [lora/](python/sglang/srt/lora/) | [lora.mdx](docs_new/docs/advanced_features/lora.mdx) |
| **量化** | [layers/quantization/](python/sglang/srt/layers/quantization/) | [quantization.mdx](docs_new/docs/advanced_features/quantization.mdx) |
| **多模态** | [multimodal/](python/sglang/srt/multimodal/) | [sglang-diffusion/](docs_new/docs/sglang-diffusion/) |
| **采样** | [sampling/](python/sglang/srt/sampling/) | [sampling_params.mdx](docs_new/docs/basic_usage/sampling_params.mdx) |

---

## 阶段六：性能工程 + 调试（进阶）

- **基准**：[bench_serving.py](python/sglang/bench_serving.py)（吞吐/延迟）、`bench_one_batch.py`（单 batch 分析）、[developer_guide/benchmark_and_profiling.mdx](docs_new/docs/developer_guide/benchmark_and_profiling.mdx)
- **可观测性**：[observability/](python/sglang/srt/observability/)、[observability.mdx](docs_new/docs/advanced_features/observability.mdx)、[production_metrics.mdx](docs_new/docs/references/production_metrics.mdx)
- **Profiler**：[profiler.py](python/sglang/profiler.py)
- **CUDA 内核层**（最后再看）：[sgl-kernel/](sgl-kernel/) C++/CUDA，配 [development_jit_kernel_guide.mdx](docs_new/docs/developer_guide/development_jit_kernel_guide.mdx)
- 仓库内调试技能（`.claude/` 里现成）：`debug-cuda-crash`、`debug-distributed-hang`、`sglang-bisect-ci-regression`

---

## 学习资源补充

- 官方文档主线：`docs_new/docs/`（比 `docs/` 旧目录新）
- 官方博客：lmsys.org/blog（DeepSeek 部署系列、GB200 系列是很好的实战案例）
- 官方 Slides：[sgl-learning-materials](https://github.com/sgl-project/sgl-learning-materials)
- Roadmap：[roadmap.sglang.io](https://roadmap.sglang.io/)

---

## 最短可行路径

如果只想抓重点：**阶段二（会用）→ 阶段三的请求生命周期 → 阶段四的 ① KV Cache + ② 调度器 + ③ Model Runner** 三块，基本就能读懂 SGLang 的骨架了；其余模块都是在这根骨架上挂肉。
