# Representative workload durations + resource counts (for the C assumptions)

Literature / MLPerf / vendor-benchmark search backing the per-workload compute time C (hours) and
resource counts used in the carbon-scheduling analysis (`scripts/carbon_temporal.py`). Where a
workload has no inherent single duration (serving, a query, a gem5 unit-of-work), it is modeled as
the deferrable aggregate (batch / suite / campaign). C is in WALL-CLOCK hours at the cited scale.

| WL | workload | **C used** | representative duration | resources | range |
|---|---|---|---|---|---|
| A1 | Llama-3.1-8B full FT (FSDP) | **4 h** | 1-4 h (Alpaca ~52k, 2-3 ep) | **8 GPU** (we ran 4xA100-40GB) | 1-4 h/4-8 GPU -> 6-24 h/16-64 GPU multi-node |
| A2 | vLLM Llama-8B batch inference | **2 h** | ~1.5 h / ~50k prompts | **1 GPU** (TP=1) | 2-20 min (1k-10k prompts) -> ~30 h/1 GPU or ~4 h/8 GPU (1M) |
| A3 | ViT-H/14 (~632M) ImageNet train | **12 h** | days (full); 12 h = segment/FT | **many GPU** | ViT-H/14 = 2,500 TPUv3-core-days; ~52 h on 40 A100 (ViT-class); 1-GPU ViT-B ~3 days |
| A4 | DLRM-DCNv2 (Criteo) train | **1 h** | ~14-15 min (single node) | **8 GPU** | 14-15 min/8 GPU -> 1.61 min at-scale (1000s GPU); Meta prod 128 A100/supernode |
| A5 | PageRank (GAPBS Kronecker) | **1 h** | ~3 min (17.7 s/iter x 5-20) | **64 cores** | 10 s-5 min/16-128 core -> minutes-hours distributed (Spark/GraphX, billion-edge) |
| A6 | gem5 detailed (O3) simulation | **8 h** | hours-days per detailed sim | **1 core/sim** | atomic ~min-1 h; O3 hours-days (billions of instr); campaigns fan out many 1-core sims |
| A7/A8 | DuckDB OLAP (GROUP BY/scan ~100 GB) | **1 h** | single query sec; suite ~3 min | **64 cores** | ClickBench cold ~163 s; TPC-H SF100 ~10-13 s -> hours at multi-TB (SF100,000 ~1.2 h/query) |

## Sources (verified)
- **A1 FSDP 8B FT**: meta-llama/llama-cookbook FSDP recipe; Oracle Llama-2-13B full FT 3 ep ~2.5 h
  (https://blogs.oracle.com/ai-and-datascience/multi-gpu-multi-node-finetuning-llama2-oci);
  MLPerf v4.0 Llama-2-70B **LoRA** = 28 min on 8xH100 (RELATED anchor, LoRA not full FT)
  (https://www.redhat.com/en/blog/generative-ai-fine-tuning-llms-red-hat-and-supermicro-showcase).
- **A2 vLLM batch inference**: vLLM v0.6 perf blog (https://blog.vllm.ai/2024/09/05/perf-update.html);
  DatabaseMart single-H100 ~9.3 req/s ~5,500 out tok/s (https://www.databasemart.com/blog/vllm-gpu-benchmark-h100).
  NOTE: live serving has no inherent duration -> model as offline batch over N prompts (deferrable).
- **A3 ViT**: ViT paper Table 2, ViT-H/14 = 2,500 TPUv3-core-days, ViT-L/16 ImageNet-21k ~30 days on
  TPUv3-8 (https://arxiv.org/abs/2010.11929); NVIDIA DGX SuperPOD ~52 h on 40 A100 for a ViT-class
  (VOLO-D5) model (https://developer.nvidia.com/blog/training-a-state-of-the-art-imagenet-1k-visual-transformer-model-using-nvidia-dgx-superpod/).
  Cite ViT-H in TPU-core-days, not GPU-h.
- **A4 DLRM**: NVIDIA MLPerf H100 record blog (https://developer.nvidia.com/blog/breaking-mlperf-training-records-with-nvidia-h100-gpus/);
  Intel Gaudi2 ~14.1-14.8 min single-node 8-accel; Meta ZionEX 128 A100/supernode (https://arxiv.org/abs/2104.05158).
  EXCLUDE the spurious "18.79 min" figure (misattributed LLM number).
- **A5 PageRank**: GAP benchmark suite paper (https://arxiv.org/pdf/1508.03619); Beamer thesis Table 7.3,
  17.7 s/iter single-thread on kron, 16c/32t (https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-153.pdf).
  NOTE: official GAPBS kron is **scale-27** (134M V/2.1B E); our scale-29 (537M V/8.5B E) is a larger
  variant; the 64-core full-PR time is derived from single-thread (GAPBS scales ~10-15x/node).
- **A6 gem5**: gem5-20 paper, sim rates (atomic ~1 MIPS, O3 ~0.01-0.3 MIPS) (https://arxiv.org/pdf/2007.03152);
  Binkert 2011 gem5 (https://bpb-us-e1.wpmucdn.com/sites.gatech.edu/dist/c/332/files/2020/07/gem5_can2011.pdf).
  Single-threaded; campaigns = many independent 1-core sims (SimPoint/checkpoint sampling).
- **A7/A8 DuckDB**: ClickBench (https://github.com/ClickHouse/ClickBench); DuckDB TPC-H SF100 ~10-13 s
  (https://arxiv.org/pdf/2506.09226); DuckDB v1.4 LTS benchmark (https://duckdb.org/2025/10/09/benchmark-results-14-lts).
  Single query is sub-second-to-seconds -> model as a query suite / ETL pipeline (deferrable).

## Modeling notes for the paper
1. **A2, A6, A7/A8 have no single inherent duration** -> model as finite batch (N prompts) / campaign
   (many 1-core sims) / query-suite-or-ETL. This is the defensible deferrable unit for a scheduler.
2. **Resource count != our testbed.** Production configs are larger (A1: 8 GPU vs our 4; A4: 8 GPU vs
   our 1; A3: many). Our measured P and E_mech are for OUR config, so C is set at our-testbed scale and
   the claim is "as measured on 4xA100 / 1-GPU / 64-core ford."
3. **Consequence in the analysis**: short jobs (A4/A5/A7/A8, C=1 h) fit one hour -> shift to the single
   cleanest hour with K=0 suspends (zero mechanism overhead); only multi-hour jobs (A1, A2, A3, A6)
   ever suspend and pay overhead (largest for A6 gem5: low power x long x SATA).
