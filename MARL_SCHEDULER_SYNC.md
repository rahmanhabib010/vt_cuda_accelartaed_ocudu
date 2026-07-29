# Multi-Cell UL Scheduler Synchronization for MARL

## Purpose

This document describes the original OCUDU uplink scheduling flow, the
multi-cell synchronization refactoring in this repository, its current
limitations, and how to build, test, and run it.

The long-term goal is an AI-enhanced multi-site scheduler in which a
multi-agent reinforcement learning (MARL) policy receives aligned observations
from all target cells and selects an action for each cell.

The code documented here implements the synchronization foundation for that
work. It does **not** yet implement Top-K feature extraction, dataset logging,
MARL training, model inference, or MARL action application.

## Baseline OCUDU Scheduler

In the original scheduler, `ue_scheduler_impl::run_slot_impl()` processed the
cells sequentially. For each cell, `run_sched_strategy(cell_index)` ran both
downlink and uplink slice scheduling.

The relevant UL new-transmission path was:

```text
run_slot_impl()
  cell A: run_sched_strategy()
    UL slice scheduling
      prepare feasible UL candidates
      allocate grant builders
      recommended_vrbs()
      finalize cell A grants

  cell B: run_sched_strategy()
    UL slice scheduling
      prepare feasible UL candidates
      allocate grant builders
      recommended_vrbs()
      finalize cell B grants
```

`intra_slice_scheduler::schedule_ul_newtx_candidates()` performed both stages
inside one call:

1. Select candidate UEs and reserve the control-plane resources needed for
   their UL grants.
2. Call each grant builder's `recommended_vrbs()` method to choose VRBs and
   complete the grants.

Consequently, cell A could reach VRB selection and finalization before cell B
had produced its feasible candidates. That does not provide a point where a
MARL policy can observe aligned inputs from all cells.

## Requirement

All target cells must reach the AI input point before any target cell calls
`recommended_vrbs()` for UL newTx:

```text
cell A: feasible builders -> future Top-K features --+
                                                     |
cell B: feasible builders -> future Top-K features --+--> synchronization
                                                           -> MARL decision
                                                           -> apply actions
```

The candidate-collection stage therefore has to be separated from the
VRB-finalization stage.

## Implemented Refactoring

### Candidate collection and finalization are separate

`intra_slice_scheduler` now exposes:

```cpp
bool collect_ul_sched(ul_ran_slice_candidate slice, scheduler_policy& policy);
void finalize_ul_sched();
bool has_pending_ul_sched() const;
```

`collect_ul_sched()` performs UL retransmission handling and UL newTx candidate
preparation. For a feasible newTx batch, it:

- prepares and prioritizes the UL candidates;
- creates the pending UL grant builders;
- reserves the required control-plane resources;
- retains the slice, policy, and RB-budget context; and
- returns before `recommended_vrbs()` is called.

`finalize_ul_sched()` resumes the pending batch. It currently invokes the
existing `recommended_vrbs()` heuristic, completes the grants, reports the
grants to the scheduler policy, and clears the pending state.

The original `ul_sched()` method remains as a compatibility wrapper. It calls
collection followed immediately by finalization when used directly.

Safety checks prevent:

- collecting another batch while one is pending;
- finalizing when no batch is pending; and
- advancing to another slot with an unfinalized batch.

### Cell scheduling now advances in synchronized rounds

`ue_scheduler_impl::run_slot_impl()` now:

1. prepares every cell for the current slot;
2. performs DL scheduling for every cell;
3. asks each cell to collect at most one feasible UL newTx batch;
4. waits until every cell has either produced a pending batch or exhausted its
   current UL work;
5. reaches the multi-cell synchronization point;
6. finalizes all pending batches; and
7. repeats the synchronized UL round until no cell has more work.

The current flow is:

```text
prepare all cells
  -> schedule DL for all cells
  -> collect one UL batch from each active cell
  -> MULTI-CELL AI INPUT SYNCHRONIZATION POINT
  -> finalize pending batches with the existing heuristic
  -> repeat until UL work is exhausted
  -> post-process all cells
```

DL remains before UL because UL DCI format 0_1 must observe the correct DAI.
Periodic UCI, SRS, fallback scheduling, and triggered UL preparation also remain
before feasible UL candidate collection.

## Files Changed

- `lib/scheduler/ue_scheduling/intra_slice_scheduler.h`
  - Declares collection, pending-state inspection, and finalization APIs.
  - Stores the context needed across the synchronization point.
- `lib/scheduler/ue_scheduling/intra_slice_scheduler.cpp`
  - Splits UL newTx candidate collection from VRB selection/finalization.
  - Adds pending-state invariants.
- `lib/scheduler/ue_scheduling/ue_scheduler_impl.h`
  - Splits DL scheduling from incremental UL batch collection.
- `lib/scheduler/ue_scheduling/ue_scheduler_impl.cpp`
  - Implements synchronized multi-cell UL rounds.
  - Contains the marked multi-cell AI input synchronization point.
- `tests/unittests/scheduler/policy/scheduler_policy_test.cpp`
  - Verifies that a UL PUSCH grant can exist with an empty RB allocation after
    collection and receives a non-empty allocation only after finalization.

## Current Status and Limitations

Implemented:

- separation of feasible UL newTx batch collection from VRB finalization;
- a scheduler-level barrier before any pending cell calls
  `recommended_vrbs()`;
- synchronized rounds across all cells managed by the same
  `ue_scheduler_impl`;
- compatibility with the existing non-MARL allocation behavior;
- a focused unit test for deferred VRB selection.

Not implemented:

- selection of a configured subset of target cells;
- Top-K candidate feature extraction;
- observation schemas and normalization;
- timestamps or identifiers for dataset alignment;
- dataset export or logging;
- a MARL environment or training pipeline;
- model loading and inference;
- validation and application of MARL-selected VRBs/actions;
- timeout or fallback handling for an external inference service;
- synchronization across separate gNB processes or separate hosts.

The present barrier is cell-level synchronization inside one scheduler
instance. It is not a distributed synchronization mechanism.

## Recommended Next Development Stages

1. Define an observation schema for each pending grant builder, including all
   state required to evaluate a feasible action.
2. Extract and deterministically order Top-K candidates at the marked
   synchronization point.
3. Log aligned multi-cell observations, selected heuristic actions, outcomes,
   and slot/cell identifiers.
4. Validate the collected dataset before starting MARL training.
5. Train and evaluate the MARL policy offline.
6. Add an inference interface at the same synchronization point.
7. Validate every returned action against current scheduler constraints and
   retain `recommended_vrbs()` as a safe fallback.
8. Add integration, load, latency, determinism, and multi-cell fairness tests.

Data collection should be added after this refactoring because the barrier
provides aligned per-slot, per-round candidate state. Training should start
only after the observation and outcome records have been validated.

## Build

Run commands from the repository root.

For a GH200 system using CUDA compute capability 9.0:

```bash
cmake -S . -B build_mss \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=90
```

Build the focused test:

```bash
cmake --build build_mss -j8 --target scheduler_policy_test
```

Build the gNB:

```bash
cmake --build build_mss -j8 --target gnb
```

Do not put a newline immediately after `--target`; the target name must be part
of the same shell command.

### Selecting UHD 4.6 for an X410

If both UHD 4.1 and UHD 4.6 are installed, select the X410-compatible UHD
installation before configuring:

```bash
export PATH=/opt/uhd-4.6/bin:$PATH
export LD_LIBRARY_PATH=/opt/uhd-4.6/lib:${LD_LIBRARY_PATH:-}
export PKG_CONFIG_PATH=/opt/uhd-4.6/lib/pkgconfig:${PKG_CONFIG_PATH:-}
```

Then configure with:

```bash
cmake -S . -B build_mss -U'UHD_*' \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DCMAKE_PREFIX_PATH=/opt/uhd-4.6 \
  -DCMAKE_INCLUDE_PATH=/opt/uhd-4.6/include \
  -DCMAKE_LIBRARY_PATH=/opt/uhd-4.6/lib
```

Verify the linked UHD library after building:

```bash
LD_LIBRARY_PATH=/opt/uhd-4.6/lib \
  ldd ./build_mss/apps/gnb/gnb | grep libuhd
```

For the tested X410 setup, this should resolve to
`/opt/uhd-4.6/lib/libuhd.so.4.6.0`.

## Test

Run the focused deferred-finalization test:

```bash
./build_mss/tests/unittests/scheduler/policy/scheduler_policy_test \
  --gtest_filter='*ul_newtx_vrb_selection_can_be_deferred_until_after_candidate_collection*'
```

The test is parameterized for the supported scheduler policies, so two passing
instances are expected.

Run the scheduler-labelled test suite:

```bash
ctest --test-dir build_mss -L sched -j8 --output-on-failure
```

On the Kasei development machine, the implementation was also validated against
the existing `build` directory:

- `ocudu_sched` and `scheduler_policy_test` built successfully;
- the two focused parameterized test instances passed; and
- all 499 scheduler-labelled tests passed.

## Run the Two-Cell X410 Configuration

The two-cell/two-PLMN configuration used during bring-up is:

```text
configs/gnb_rf_x410_tdd_n78_two_plmn.yml
```

Start the gNB with:

```bash
sudo ./build_mss/apps/gnb/gnb \
  -c ./configs/gnb_rf_x410_tdd_n78_two_plmn.yml
```

If the runtime loader does not retain the UHD selection, use:

```bash
sudo env LD_LIBRARY_PATH=/opt/uhd-4.6/lib \
  ./build_mss/apps/gnb/gnb \
  -c ./configs/gnb_rf_x410_tdd_n78_two_plmn.yml
```

A successful startup should report UHD 4.6, initialize the X410, list both
cells, connect N2 to the AMF, and print:

```text
==== gNB started ===
```

The message `Attempting to set tick rate to 0. Skipping.` was observed during
the successful X410 startup and was non-fatal.

The current configuration starts two logical cells on one X410 using separate
RF channels. It should not be described as a two-X410 configuration.

## Core and UE Requirements

The two-PLMN gNB configuration advertises:

- PLMN `00101`, TAC 7; and
- PLMN `99999`, TAC 7.

Open5GS AMF/NRF configuration must serve both PLMNs and TAC 7. Each Android UE
also needs its own matching Open5GS subscriber entry with the correct IMSI,
authentication key, OP/OPc, AMF value, and DNN.

An N2 TCP/SCTP connection alone is not sufficient. Successful NG setup is
indicated by the absence of the `unknown-PLMN-or-SNPN` failure and by the gNB
continuing to its started state.

## How to Confirm the MARL Path Is Exercised

Starting two cells proves that the configuration and scheduler can launch, but
it does not exercise meaningful UL newTx candidate synchronization by itself.
At least one registered UE must generate UL scheduling demand, such as a BSR
caused by uplink traffic.

Before adding model inference, useful runtime instrumentation should record:

- slot and synchronized-round identifiers;
- cell and UE identifiers;
- the number and ordering of feasible candidates per cell;
- extracted Top-K observations;
- the action chosen by the existing heuristic;
- the final VRB allocation; and
- relevant outcomes such as bytes scheduled, HARQ result, throughput, and
  latency.

## Git Notes

The synchronization changes were developed and tested against source commit:

```text
69d56b4a3d
```

Before pushing, review only the intended source, test, configuration, script,
and documentation files. Build directories such as `build/` and `build_mss/`
should normally remain untracked.

Example:

```bash
git status --short
git diff --check
git diff --stat
```

Then stage explicit paths rather than using a broad command that may include
build artifacts:

```bash
git add \
  MARL_SCHEDULER_SYNC.md \
  lib/scheduler/ue_scheduling/intra_slice_scheduler.cpp \
  lib/scheduler/ue_scheduling/intra_slice_scheduler.h \
  lib/scheduler/ue_scheduling/ue_scheduler_impl.cpp \
  lib/scheduler/ue_scheduling/ue_scheduler_impl.h \
  tests/unittests/scheduler/policy/scheduler_policy_test.cpp
```

Add deployment configurations or Open5GS helper scripts separately if they are
intended to be part of the same commit.
