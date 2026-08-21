# HBC vs. Groq LPU vs. GPU — Decode Architecture Study

Findings from the `groq-lpu-study` branch. Covers the analytical proxy systems,
the modeling results, the provenance of every assumption, and the open questions.

**Read the provenance section before quoting any number from this document.**

---

## 1. Question

At decode, attention and FFN have opposite hardware appetites. Attention streams the
whole KV cache to emit one vector per token (enormous read:write ratio, capacity-hungry);
dense FFN GEMM streams weights and is bandwidth-bound at small batch. Does splitting them
across two device types beat running everything on a GPU?

Three devices were modeled:

- **B200 SXM** — real silicon, real system YAML
- **HBC proxy** — Qualcomm "High Bandwidth Compute", near-memory / processing-in-memory
- **Groq 3 LPU** — SRAM-resident inference accelerator

Workload: Llama-class dense model, decode, TP=8, batch 32, ISL 4000, OSL 1000,
`--database-mode SOL`.

---

## 2. System configurations

| | B200 SXM | HBC proxy | Groq 3 LPU |
|---|---|---|---|
| `mem_bw` | 8 TB/s (HBM3e) | 48 TB/s internal `[H1]` | 150 TB/s (SRAM) |
| `mem_bw_external` | — | 6 TB/s `[H2]` | — |
| `mem_capacity` | 180 GiB | 128 GB `[H3]` | 500 MB |
| `mem_bw_empirical_scaling_factor` | 0.80 | 0.85 `[H1]` | 0.95 |
| `mem_empirical_constant_latency` | 3 us | 2 us | 1 us |
| `near_memory_ops` | — | GenerationAttention, ContextAttention, ElementWise, Embedding | — |
| `fp8_tc_flops` | 4500 TFLOPS | 500 TFLOPS `[H4]` | 1200 TFLOPS |
| `bfloat16_tc_flops` | 2250 TFLOPS | 250 TFLOPS `[H4]` | 600 TFLOPS `[A2]` |
| `int8_tc_flops` | 4500 TFLOPS | 500 TFLOPS `[H4]` | 1200 TFLOPS |
| `fp4_tc_flops` | 9000 TFLOPS | 1000 TFLOPS `[H4]` | 2400 TFLOPS `[A2]` |
| `power` | 1000 W | 700 W `[H4/H5]` | 500 W `[A5]` |
| `num_gpus_per_node` | 8 | 8 `[H5]` | 256 (LPX rack) |
| `intra_node_bw` | 900 GB/s | 900 GB/s `[H5]` | 2.5 TB/s `[A1]` |
| `inter_node_bw` | 100 GB/s | 100 GB/s `[H5]` | 100 GB/s `[A3]` |
| `p2p_latency` | 10 us | 10 us `[H5]` | 1 us `[A4]` |
| `nccl_mem` (8-way) | 392 MB | 392 MB | 0 `[A6]` |
| `other_mem` | 3.5 GB | 3.5 GB | 16 MB `[A6]` |

Bandwidths are per-device, single direction, matching the convention of every other
system YAML in `src/aiconfigurator/systems/`.

---

## 3. Results

### 3.1 Per-operator decode breakdown (ms)

| decode op | B200 | HBC | Groq | HBC/B200 |
|---|---|---|---|---|
| `generation_attention` | 294.59 | 75.15 | 31.31 | **0.26x** |
| `gate_ffn1_gemm` | 264.83 | 353.10 | 111.74 | 1.33x |
| `ffn2_gemm` | 133.07 | 177.43 | 55.87 | 1.33x |
| `qkv_gemm` | 54.01 | 72.02 | 22.35 | 1.33x |
| `proj_gemm` | 43.47 | 57.96 | 17.88 | 1.33x |
| `ar_1` / `ar_2` | 40.74 ea | 40.74 ea | 14.67 ea | 1.00x |
| `logits_gemm` | 24.48 | 32.64 | 10.36 | 1.33x |
| `add_norm_1/2` | 10.48 ea | 1.75 ea | 0.56 ea | **0.17x** |
| `act_gate` | 4.91 | 0.82 | 0.26 | **0.17x** |
| **TOTAL** | **922.46** | **854.72** | **280.44** | **0.93x** |

Every ratio is arithmetic, not emergent:

- `0.17x = 1/6.00` — ops in `near_memory_ops` run at HBC's internal 48 TB/s vs B200's 8.
- `1.33x = 8/6` — GEMM is *not* near-memory eligible, so it pays the 6 TB/s external
  interface. HBC is **worse** than a GPU at GEMM.
- `1.00x` — allreduce is network-bound, untouched by memory bandwidth.
- `0.26x` — attention is eligible but gets 3.9x, not 6x: `[H4]`'s weak ALU eats a third
  of the bandwidth win (pure memory would predict 49 ms; the model gives 75 ms).

### 3.2 Where B200 decode time goes

| category | ms | share |
|---|---|---|
| GEMM | 519.86 | 56% |
| attention | 294.59 | 32% |
| allreduce | 82.11 | 9% |
| elementwise | 25.90 | 3% |

HBC saves 219 ms on attention and gives back 173 ms on GEMM. Net 7%.

### 3.3 End-to-end tpot (ms)

| case | Phase 1 | Phase 2 | Phase 1 overstated by |
|---|---|---|---|
| baseline all B200 | 0.9230 | 0.9230 | 1.00x |
| all Groq LPU | 0.2810 | 0.2810 | 1.00x |
| all HBC | 0.6860 | 0.8560 | 1.25x |
| **AFD: HBC attn + Groq FFN** | 0.2100 | **0.3350** | **1.60x** |

Phase 1 used a single bandwidth scalar; Phase 2 added the dual-bandwidth model.
Non-HBC rows are byte-identical, which is the backward-compatibility guarantee
showing up end to end.

### 3.4 The `[H2]` internal:external sweep

| ratio | external TB/s | tpot ms | |
|---|---|---|---|
| 1:1 – 4:1 | 48 – 12 | 0.2100 | saturated; interface irrelevant |
| 8:1 | 6.0 | 0.3350 | `[H2]` default |
| 16:1 | 3.0 | 0.5910 | |
| 32:1 | 1.5 | 1.1030 | **worse than the GPU baseline** |
| 48:1 | 1.0 | 1.6150 | |

Crossover at ~25:1 against the 0.9230 baseline. Useful band roughly 4:1 to 16:1.

### 3.5 Backward-compatibility verification

| system | ttft | kv_ms | vs before |
|---|---|---|---|
| b200_sxm | 49.522 | 0.291266 | identical |
| h200_sxm | 104.027 | 0.562534 | identical |
| gb200 | 46.858 | 0.698165 | identical |

Tests: 448 passed, 1 failed (`test_parse_qwen35_dense_config`, fails identically on
unmodified main — pre-existing).

---

## 4. Conclusion, and how far it can be trusted

**Structural finding (robust):** attention and FFN want opposite hardware. HBC-attention
+ Groq-FFN at 0.335 ms is 2.75x the all-B200 baseline and beats both homogeneous
alternatives. This follows from the dual-bandwidth architecture regardless of the
absolute constants.

**Numerical findings (fragile):** every specific millisecond figure rides on `[H1]`,
`[H2]` and `[H4]`, none of which is a published number.

A caution on framing: HBC is *not* the bandwidth champion — Groq is, by 3x, and Groq is
faster at every single operator including GEMM. Groq loses only on capacity (500 MB vs
128 GB), which is why it cannot hold the KV cache. HBC's actual claim is SRAM-class
bandwidth at HBM-class capacity: it occupies the gap the other two leave open.

---

## 5. Assumption provenance

Every derived number carries a tag in its system YAML. This is the epistemic status of
each, and it varies more than the uniform tagging suggests.

### HBC — `[H1]`–`[H5]`

Qualcomm published **only per-watt ratios**, verified against the source article:

| published claim | exact wording | baseline |
|---|---|---|
| bandwidth | "6x the bandwidth **per watt** versus HBM **for large batch sizes**" | HBM |
| capacity | "200x the capacity **per watt** versus SRAM" | SRAM |
| internal:external | "an internal-to-external ratio measured in **multiples**" | — |

All are footnoted as Qualcomm estimates against competitor published specs. There are
**no absolute TB/s, GB, TFLOPS or watt figures anywhere in the source.**

| tag | value | derivation | status |
|---|---|---|---|
| `[H1]` | 48 TB/s internal | 8 TB/s (B200 HBM3e) x 6 | ratio x real baseline; valid only if power is comparable. If HBC runs cooler, too high |
| `[H2]` | 6 TB/s external (8:1) | none — source says only "multiples" | **choice within an unquantified range.** Decisive: see 3.4 |
| `[H3]` | 128 GB | Groq 500 MB at ~500 W -> 1 MB/W, x200 -> ~100 GB | **weakest link:** the ~500 W input is `[A5]`, itself a placeholder |
| `[H4]` | 500 TFLOPS FP8 | none — chosen as ~11% of B200 | **invented.** See section 6 |
| `[H5]` | network knobs | none — reuses the HBM-class node template | placeholder |

### Groq — `[A1]`–`[A6]`

Core Groq specs **are** published and independently verified: 500 MB SRAM, 150 TB/s,
1.2 PFLOPS FP8, 2.5 TB/s scale-up per chip; rack of 256 LPUs, 128 GB, 40 PB/s,
315 PFLOPS. Two cross-checks hold: 256 x 500 MB = 128 GB, and 315/256 ~ 1.23 PFLOPS.
The `[A]` tags cover only the gaps.

| tag | covers | status |
|---|---|---|
| `[A1]` | 2.5 TB/s intra-node | derived as 640/256, but **matches the published per-chip scale-up figure** — promote to citation. Single- vs bidirectional remains open |
| `[A2]` | bf16 = 0.5x FP8, FP4 = 2x FP8 | only FP8 is published |
| `[A3]` | 100 GB/s inter-node | Spectrum-X / CX9 class; NVLink not yet integrated |
| `[A4]` | 1 us p2p | deterministic pre-compiled dataflow, no runtime scheduler |
| `[A5]` | 500 W | **not published.** `[H3]` depends on this |
| `[A6]` | nccl_mem 0, other_mem 16 MB | GPU template values (~640 MB) exceed the LPU's entire 500 MB SRAM, forcing spurious OOM |

**Three grades of evidence sit in the same results table:** B200 is measured silicon,
Groq is published spec plus small gap-fills, HBC is constructed entirely from two ratios.

---

## 6. HBC compute: sourcing investigation

**Question:** does any public source state HBC's compute throughput?

**Answer: no — and the omission is deliberate.**

- Qualcomm refused on the record: "Peak FLOPS are notably missing from Qualcomm's AI250
  disclosures — the company declined to share specifics upon our request."
  (The Register, 2026-06-30) — verified directly.
- Qualcomm publishes TOPS freely elsewhere (870 TOPS INT8 for Cloud AI 100 Ultra;
  "2,000 TOPS" automotive in the same Investor Day deck set that gave HBC none).
- Qualcomm argues against the metric: "Tokens per watt replaces FLOPS." The AI250 spec
  table has rows for Memory, Scale-Up, Scale-Out and Thermal, and no compute row.

**AI250 is HBC Gen 1**, so it is the right place to look — but it publishes no compute
either. Published AI250 figures are all memory: 768 GB, 133 TB/s effective per card.

The only absolute FLOPS figure in circulation (NextPlatform, 983 PFLOPS FP4/rack) is
self-labeled speculation, concerns the **AI200** (the non-HBC SKU), and predates the HBC
reveal. Not usable.

### 6.1 The `[H4]` rationale is contradicted by the architecture

The YAML comment justifies 500 TFLOPS with "Near-memory logic is not a tensor core...
HBC wins on memory-bound ops and LOSES on dense GEMM." The evidence does not support
that premise, because it conflates two different architecture classes:

| class | where the logic sits | dense GEMM? |
|---|---|---|
| bank-level in-DRAM PIM (Samsung HBM-PIM, SK hynix AiM) | ALUs at bank pitch, DRAM process | effectively no — ~2 ops per element ceiling |
| logic-die PNM (Samsung CXL-PNM) | separate logic-process die | **yes** — SRAM tiles, reduction trees, broadcast paths |

The bank-level ceiling is real and measured — Samsung HBM2-PIM delivers 1.229 TFLOPS
FP16 against 1.23 TB/s, exactly 1 FLOP/byte, reconstructing from the hardware
(16 pseudo-channels x 8 units x 16 lanes x 2 FLOP x 300 MHz). SK hynix GDDR6-AiM lands
on the identical invariant.

**But HBC is the logic-die case.** The Register describes DRAM stacked on top of logic
connected by TSVs; ServeTheHome reports a base die containing compute acting as the
accelerator; Qualcomm's patent family describes an array of PUs on the base die with MAC
compute blocks. And Samsung ran exactly this experiment: when they wanted real GEMM they
moved it off the DRAM — CXL-PNM puts an adder-tree for GEMV *and a separate PE array for
GEMM* on the controller logic die.

So "HBC cannot do dense GEMM" is unsupported. What the evidence supports is the narrower
"logic inside DRAM banks cannot do dense GEMM profitably."

### 6.2 Is 500 TFLOPS defensible?

As a central value, yes — as the midpoint of a ~20x bracket:

| anchor | implied FP8 | vs 500 TF |
|---|---|---|
| PIM invariant applied to `[H1]`'s 48 TB/s | ~96 TFLOPS | 5.2x high |
| PIM invariant applied to Qualcomm's published 133 TB/s | ~266 TFLOPS | 1.9x high |
| Qualcomm accelerator lineage (Cloud AI 100 Ultra -> 3nm, ~2.5 kW/card) | ~1–5 PFLOPS | 2–10x low |

500 TF is roughly the geometric mean. The number survives; the reasoning does not.
Note also that "~11% of B200" is not a citation — it is 500/4500 restated.

### 6.3 The unexamined quantity

`[H1]` / `[H4]` gives HBC **0.096 bytes/FLOP** against B200's 0.00178 — a **54x
memory-richness advantage** that appears nowhere in the file and is the emergent product
of two independently invented numbers. That ratio, not either number alone, decides
whether each operator lands memory-bound or compute-bound.

Concretely, `max(sol_math, sol_mem)` resolves to `sol_mem` for GEMM on both devices at
batch 32, which is why the GEMM penalty is exactly 8/6 and `[H4]` never enters. Crossover
to compute-bound is near batch ~42 for HBC but ~281 for B200, so **the 1.33x figure is
not a stable architectural property** — it holds only at small batch. Qualcomm's own
bandwidth claim is scoped to "large batch sizes", precisely the regime where this proxy's
GEMM penalty would be worst.

---

## 7. Known limitations

1. **AFD `oom` flags are unreliable.** `_get_afd_memory_usage(model, database, ...)`
   takes a *single* database, so heterogeneous AFD models compute per-group but memory
   homogeneously. It cannot attribute attention memory to HBC (128 GB) vs FFN memory to
   Groq (500 MB). Compute rankings stand; feasibility rankings do not. This matters most
   for Groq's 0.281 ms — the fastest figure in the study, ruled out only by an SRAM
   ceiling the model cannot currently check.

2. **A probable SDK bug was worked around, not fixed.** `base_backend.py:567` gates the
   heterogeneous path on `enable_afd and afd_num_microbatches > 1`, but that parameter
   defaults to 1 — so per-group `attn_database`/`ffn_database` are silently ignored and
   everything falls back to a single database. This produced identical tpot for three
   different hardware combos. The gated function contains an `if M == 1:` branch built
   for exactly the non-pipelined heterogeneous case, currently unreachable. The study set
   `afd_num_microbatches=2` to route around it rather than change SDK semantics uninvited.

3. **`near_memory_ops` membership is a study invention.** The source never states which
   operations the in-stack logic can perform. The list came from a read:write-ratio
   argument, not vendor guidance — and per 6.1 its exclusion of GEMM rests on a premise
   the architecture evidence contradicts.

4. **The "all HBC" row may model a configuration that does not exist.** If HBC is
   inherently a memory device paired with a host, GEMMs would run on the host, not on
   HBC's external interface. Under that reading the split is not one option among three
   — it is the only real one.

---

## 8. Open items

| # | item | why it matters |
|---|---|---|
| 1 | Sweep `[H4]` 100 TF – 2 PF log-uniformly | `[H2]` carries a decisive-knob warning and `[H4]` does not, though both are invented and load-bearing |
| 2 | Reconsider excluding GEMM from `near_memory_ops` | **Conclusion-flipping.** If GEMM runs in-stack at 48 TB/s, the 1.33x penalty becomes a 6x advantage and the case for offloading FFN to Groq may collapse |
| 3 | Batch sweep of the GEMM ratio | tests whether 1.33x survives past batch ~42 |
| 4 | Add a paired-host variant (GEMM at B200 rates, attention on HBC) | arguably the most faithful reading of the source; tests whether Groq FFN beats a plain GPU host |
| 5 | Phase 3: per-group memory attribution in `_get_afd_memory_usage` | makes AFD `oom` flags trustworthy |
| 6 | Fix the AFD gate rather than work around it | the dead `M == 1` branch suggests it was meant to work |
| 7 | Re-tag `[H3]` to note its dependence on `[A5]`; promote `[A1]` to a citation | current tagging implies uniform confidence that does not exist |

---

## 9. Sources

- Qualcomm, "HBC vs. HBM vs. SRAM: AI inference memory" (2026-07) —
  https://www.qualcomm.com/news/onq/2026/07/hbc-vs-hbm-vs-sram-ai-inference-memory
- The Register, "Qualcomm's proposed solution to catch up in AI infra: bury the compute
  under the DRAM" (2026-06-30) —
  https://www.theregister.com/systems/2026/06/30/qualcomms-proposed-solution-to-catch-up-in-ai-infra-bury-the-compute-under-the-dram/5264071
- NVIDIA Groq 3 LPX product page — https://www.nvidia.com/en-us/data-center/lpx/
- NVIDIA developer blog, "Inside NVIDIA Groq 3 LPX"
- Samsung HBM-PIM / Aquabolt-XL, ISSCC 2021 25.4 and Hot Chips 33
- Samsung LPDDR-PIM and CXL-PNM, Hot Chips 35
- SK hynix GDDR6-AiM, Hot Chips 35

Confidence note: the Register quote and the Qualcomm article content were verified
directly. Samsung and SK hynix figures are from ISSCC/Hot Chips presentations and are
well established. Secondary analyst readings of the AI250 base-die architecture are
consistent across outlets but are not primary vendor disclosures.
