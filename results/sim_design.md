# NoFreeCO2 simulator design (discussion draft)

Goal: ONE simulator that answers every N2 question under the MEASURED mechanism cost,
built on the EuroSys'24 foundation (their data, their policy semantics, their axes),
driven by real execution traces, with the N1 analytical model as the scheduler's
decision data. This document fixes the design before further code.

## 1. Questions the simulator must answer (each = one experiment family)

| Q | question | headline artifact |
|---|---|---|
| Q1 | aggressiveness vs net savings under measured cost | Pareto frontier per workload class, kill-point, a* |
| Q2 | load/capacity effect (user: core axis) | savings vs utilization U; green-rush characterization |
| Q3 | forecast error effect | oracle vs MAPE sweep; theta as noise filter |
| Q4 | value of cost-awareness | policy gap: cost-aware minus cost-blind (the N1-predictor payoff) |
| Q5 | decision granularity | g in {60,30,15,5} on REAL sub-hourly zones |
| Q6 | spatial vs temporal vs combined | savings split by mechanism; migration cost share |
| Q7 | (later) heterogeneity | A100 vs H100 zones, per-generation cost model |

## 2. Architecture

```
 INPUTS                          CORE                         OUTPUTS
 carbon: EuroSys'24 123-zone  ┌────────────────────────┐   per-job ledger CSV
   hourly + our real 5-min    │  slot-synchronous loop │     (exec/overhead/hold gCO2,
   zones (CAISO/NYISO/ISONE)  │  (slot = g minutes)    │      transitions, migrations,
 forecast: oracle | noise:MAPE│                        │      completion inflation,
   | CarbonCast (later)       │  Job FSM:              │      deadline misses)
 trace: PAI (primary),        │  QUEUED->RUN<->PARK    │          |
   Helios, Google via         │        <->SUSP ->DONE  │          v
   adapters -> canonical      │      \->MIGRATING      │   analysis scripts ->
   (arrive,dur,ngpu,S,power)  │                        │   Pareto/policy-gap/U-curve
 cost: N1 closed-form module  │  per-zone occupancy,   │   figures
   (THE predictor tool)       │  capacity K_z          │
 topology: zones, K_z, GCP    │  Scheduler = plugin    │
   latency matrix, WAN params └────────────────────────┘
```

Components:
- **Trace adapters** (`trace_adapters.py`): normalize each trace to the canonical job
  schema `(arrive_h, duration_h, ngpu, S_gb, power_w?)`. PAI: sensor-join for measured
  `max_gpu_wrk_mem` as S. Helios: S estimated from a PAI-calibrated ngpu-conditional
  distribution (stated assumption). Google: de-normalized CPU/mem (stated assumption),
  feeds the CPU-domain (criu) cost model.
- **Cost module** = N1 closed-form, single source of truth shared with the released
  predictor tool: GPU E_mech(S, ngpu), CPU E_mech(S, nproc, running_W profile), park
  74 W/board vs suspended, leg latencies. The simulator consumes ONLY this module --
  A1-A8 measurements never enter the simulator except through the model they validated.
- **Scheduler plugin API**: `decide(job, views, occupancy) -> RUN|PARK|SUSP|MIGRATE(dst)`.
  Policies: P0 blind FIFO, P1 ca_costblind (EuroSys'24 ideal made online), P2
  ca_costaware (marginal a* rule for pauses, stock rule for migrations).
- **Accounting**: every joule x CI at the moment it is spent; overhead billed at event
  CI; baseline = run-now-contiguous at home zone (their slack_0).

## 3. Design decisions (LOCKED after discussion)

| # | decision | choice + rationale |
|---|---|---|
| D1 | time base | slot-synchronous at g minutes (matches CI cadence; event-driven adds complexity without accuracy for slot-scale decisions). Measured leg latencies (s) tracked as completion inflation, not sub-slot events. |
| D2 | job semantics | rigid ngpu, checkpoint-resume only (no restart-from-scratch, no elasticity). Batch jobs with deadline = arrive + C x (1+slack). Interactive excluded (their finding: little benefit). |
| D3 | baseline | run-now contiguous at home zone (= their slack_0) -> all savings comparable to the artifact. |
| D4 | capacity | per-zone K = ceil(blind-peak / U). PARK holds a slot, dumped SUSP frees it. Deadline-forced runs bypass (emergency overflow, tracked). No displacement in v1 (blocked-only); displacement chains = later refinement. |
| D5 | forecast | controlled-noise model (MAPE, horizon-scaled) for the sweep; CarbonCast file mode later for realism anchor. Truth CI always used for ACCOUNTING; forecast only for DECISIONS. |
| D6 | spatial | migration allowed within GCP latency limit of HOME zone (their SLO semantics). Bill = E_mech pair + S x WAN J/GB (band 300-6000, default 3600) at src/dst mean CI; time += S / WAN_BW. |
| D7 | aggressiveness | the knob pair (theta = relative-gain gate, g = decision cadence). Analytics: theta* ~ per-slot r* = E_mech/E_slot (interior optimum; already observed empirically). |
| D8 | metrics | net gCO2 savings vs baseline (headline), gross, overhead share, transitions/job, migrations/job, completion inflation, deadline misses, blocked shifts. Distributions across zones, not just means. |
| D9 | validation anchors | (i) zero-cost theta=0 == artifact task() (sim_n2 verify PASS; sim_dc bridge ~1pp documented); (ii) N1 model accuracy from A1-A8 (median ~12%); (iii) invariants each run: occupancy <= K except forced, energy >= 0, all jobs finish. |
| D10 | scale | ford 64-thread: ProcessPool over (zone x policy x theta) cells; single cell stays single-threaded deterministic (seeded). |

## 3b. USER-LOCKED decisions (2026-07-23)

| # | decision | choice |
|---|---|---|
| D11 | trace scope v1 | ALL THREE: PAI (primary, measured S via sensor join) + Helios (4 clusters as zones; S from PAI-calibrated ngpu-conditional distribution, assumption stated) + Google 2011 (CPU-domain via criu cost model; de-normalization assumption stated) |
| D12 | workload classes for Pareto | r* QUARTILES: r* = E_mech / (P x C) per job; aligns figures directly with the analytical a*(r*) backbone |
| D13 | slack model | CONFIGURABLE: named configs, both absolute {24h, 168h} (artifact-comparable, E7 re-audit) and proportional (sensitivity); a small config set to be defined at experiment time |

## 3c. DC representation (LOCKED 2026-07-23; advances prior work, stays out of topology)

Survey of prior carbon work: Sukprasert (EuroSys'24) = no DC at all; CarbonScaler
(POMACS'23, SIGMETRICS'24 best student paper) = elastic slot pool; Lechowicz
pause-resume (POMACS'23) = single job + ABSTRACT switching cost; Carbon Explorer
(ASPLOS'23) = power envelope (renewables/battery). None model machines/racks/
topology/bin-packing. Our v1 abstraction (slot pool) is the venue norm; we advance it
along MEASURED dimensions, not spatial topology.

**One DC = a config block + four counters + a shared checkpoint-BW pool:**

```yaml
dc_frankfurt:
  ci_trace: DE.csv
  nodes: 25                          # sized from trace load (blind-peak / U)
  node_spec:                         # ford-measured profile (H100 spec after rental)
    gpus: 8 x A100-40GB              # -> 200 GPU slots; job must fit a node (ngpu <= 8)
    dram_gb: 755                     # -> DRAM staging pool (cuda-ckpt stages HBM here)
    ckpt_storage: {tier: nvme, write_gbps: 6.2, read_gbps: 12.8, w: 40, cap_tb: 3.9}
  cost_model: N1(A100)
```

Job residency ladder, each state occupying a DIFFERENT counter (all measured):
RUN (250-330 W/GPU; GPU+DRAM) | PARK (74 W/GPU; GPU) | SUSP-host (DRAM only, holds S)
| DUMP (disk only, holds S). Constraints: sum(gpu) <= slots, sum(dram) <= pool,
sum(disk) <= capacity. Suspending is a resource-MOVING operation, not a free release.

**Checkpoint-BW contention (the intra-DC face of the green rush):** concurrent dumps
fair-share the zone's write pool (nodes x measured tier BW); a 40 GB dump that takes
6.5 s alone takes ~3 min when 30 jobs dump into the same green window. Mechanism cost
gains a congestion term, parameterized entirely by measured bandwidths.

Priority of representation advances: (1) ckpt-BW contention, (2) multi-resource
residency counters, (3) storage tier as a DC design knob (NVMe vs SATA provisioning
shifts a*), (4) heterogeneous generation composition {n_A100, n_H100} (Q7, post-H100).

## 3d. Scheduler scope, information architecture, handoff, failure model (LOCKED)

**Two-level scheduler, hard scope fence below it:**
- GLOBAL (inter-DC): placement at arrival + migration (WHERE). Thin layer.
- LOCAL (per-DC): run/park/suspend timing (WHEN), admission, dump-BW arbitration --
  this IS the existing sim_dc per-zone loop.
- NOT modeled: intra-node GPU placement (inside measured coefficients), intra-cluster
  packing/fragmentation (perfect-packing assumption, node-fit constraint only),
  intra-DC clusters (escape hatch: a cluster can be represented as a zone with a
  ~0 ms latency edge -- how Q7 heterogeneity and Helios's 4 clusters will be modeled).

**Information architecture:** logically centralized global scheduler (single-operator
fleet, standard practice). CI is PUBLIC data (Electricity Maps/WattTime) -- everyone
sees all zones; the hard part is the FUTURE, which is exactly the forecast layer
(oracle vs MAPE noise). Occupancy is intra-operator telemetry. Optional staleness knob
epsilon (scheduler sees epsilon-old counters -> thundering herd/oscillation); OFF in
v1, admission control is the safeguard.

**Inter-DC handoff protocol (RESERVE -> TRANSFER -> ACTIVATE):**
1. RESERVE: global picks dst by net benefit (dCI x remaining work - [E_dump + WAN x S
   + E_restore]); dst admission checks its counters; deny -> job stays, "blocked
   shift" recorded; accept -> reservation with TTL = transfer estimate.
2. TRANSFER: job MIGRATING; src dump (src BW pool) -> WAN copy (S/wan_bw; energy
   WAN x S billed at src/dst mean CI) -> dst restore (dst BW pool + DRAM staging).
3. ACTIVATE: job handed to dst's LOCAL scheduler lifecycle.
Failure semantics: reservation expiry (stale variant) leaves the job dumped at src --
cost paid, no gain (the price of wrong aggressiveness). In the slot-synchronous v1 the
protocol collapses to an atomic reserve+activate with transfer occupying BW pools.

**Failure model / SPOF:** control plane is logically centralized, physically
replicated (consensus; standard practice -- one stated assumption, not simulated).
Graceful degradation falls out of the two-level split: if global dies, NO job stalls
(local layers autonomously run/park/suspend and enforce deadlines); only inter-DC
migration pauses. The carbon cost of an outage is therefore bounded by the
spatiotemporal-minus-temporal-only delta, which experiment E6 already measures.
Partition spectrum = staleness epsilon -> infinity (pure-local). Failure modeling
itself stays out of scope.

## 3e. Related-work anchors (positioning)
- Lechowicz et al., "Online Pause and Resume" (POMACS'23/SIGMETRICS'24): OUR problem
  shape (k-min search with switching costs) with optimal online algorithms -- but the
  switching cost is an abstract scalar beta. We: MEASURED, structured cost (nGPU/
  nProc/activity), park-vs-dump ladder, fleet capacity, heterogeneity. Plan: cite and
  BUILD ON their formulation (a* analytics anchored there, not reinvented); optional
  policy P3 = their online algorithm instantiated with measured costs.
- CarbonScaler (POMACS'23): elasticity dimension we do not model (jobs rigid);
  their cluster = slot pool confirms the representation norm; their suspend-resume
  baseline is cost-free -> re-audit candidate alongside Sukprasert.

## 4. Explicitly OUT of scope
Bin-packing/placement detail inside a zone, network congestion, monetary cost, DVFS,
renewables on-site, battery, preemption-restart training semantics, job dependencies.

## 5. Experiment matrix (paper figures)

| exp | axes | fixed | figure |
|---|---|---|---|
| E1 Pareto | theta x g, per workload class | U=0, oracle | Q1 kill-curves + a* |
| E2 load | U in {0,.7,.9,.99} x policy | theta*, oracle | Q2 U-curve |
| E3 forecast | MAPE {0,5,15,30} x theta | U=.9 | Q3 theta-as-filter |
| E4 policy gap | P1 vs P2 x (E2,E3 axes) | | Q4 value of cost-awareness |
| E5 granularity | g {60,30,15,5} on CAISO/NYISO/ISONE real | | Q5 g* per class |
| E6 spatial | latency limit {5..300ms} x WAN band | | Q6 mechanism split |
| E7 re-audit | artifact bounds vs measured-cost | | the quotable shrink table |
```
