# GitHub Issue Drafts

## 1) Baseline Reproducibility for Homogeneous NEAT
**Title**: Establish deterministic baseline runs for homogeneous NEAT

**Body**:
Create a reproducible baseline pipeline for `neat-python` training on selected MetaWorld tasks so future activation and framework changes can be measured fairly.

**Scope**:
- Set and log seeds for Python, NumPy, Gymnasium, and NEAT run setup.
- Add run metadata output (config hash, commit hash, timestamp, environment list).
- Persist per-generation fitness metrics and final evaluation summaries.

**Acceptance Criteria**:
- Running the same config and seed twice yields near-identical metrics (within defined tolerance).
- Baseline output artifacts are stored under `output/` with clear run IDs.
- README section explains how to reproduce a baseline run.

**Labels**: `research`, `infrastructure`, `baseline`, `metaworld`

## 2) Activation-Augmented NEAT in Current Framework
**Title**: Enable heterogeneous neuron activations in NEAT genomes

**Body**:
Implement activation-function diversity in evolution so neurons can use different nonlinearities (at minimum `relu`, `sigmoid`, `tanh`, `sin`, `gauss`) and mutate over generations.

**Scope**:
- Update NEAT config to allow multiple activation options.
- Enable non-zero activation mutation rate.
- Verify the evaluator and network creation path respect per-node activations.
- Add run-time reporting of activation distribution in winning genomes and populations.

**Acceptance Criteria**:
- Config supports activation library selection.
- Mutation introduces activation diversity in at least one smoke-test run.
- Logged metrics include activation counts per generation (or periodic checkpoints).

**Labels**: `research`, `core-algorithm`, `neat`

## 3) Homogeneous vs Heterogeneous Ablation Suite
**Title**: Add controlled ablation experiments for activation heterogeneity

**Body**:
Build an experiment harness to compare homogeneous NEAT (single activation) vs activation-augmented NEAT with matched seeds, budgets, and evaluation protocol.

**Scope**:
- Define experiment matrix (conditions, seeds, generations, population size).
- Add scripts to run both conditions automatically.
- Produce aggregate metrics: learning speed (AUC), best fitness, success rate.

**Acceptance Criteria**:
- One command (or documented sequence) runs the full ablation set.
- Results table and plots are generated into `results/`.
- Statistical summary (mean/std and significance test) is included.

**Labels**: `research`, `experiments`, `ablation`

## 4) MetaWorld Task Wrapper Integration
**Title**: Integrate MetaWorld benchmark tasks into environment interface

**Body**:
Extend `Environment` implementations to support selected MetaWorld tasks needed for thesis evaluation, while preserving current multi-environment evaluator behavior.

**Scope**:
- Add dependency and setup docs for MetaWorld.
- Implement one or more MetaWorld environment wrappers under `src/evaluation/environments/`.
- Define observation/action adapters compatible with shared-network interface.
- Add task registration and environment summary support.

**Acceptance Criteria**:
- At least 5 MetaWorld tasks can run end-to-end through `MultiEnvironmentEvaluator`.
- Environment summary prints correct input/output offsets for added tasks.
- Smoke test confirms no runtime shape mismatches.

**Labels**: `research`, `metaworld`, `integration`

## 5) Continuous Control Action Mapping Strategy
**Title**: Implement robust continuous action decoding for evolved policies

**Body**:
Design and implement an action decoding method from network outputs to continuous action spaces required by MetaWorld tasks.

**Scope**:
- Compare candidate mappings (`tanh` scaling, clipped linear, parameterized squashing).
- Add configurable decoder per environment.
- Validate action bounds and stability.

**Acceptance Criteria**:
- Decoder is configurable and documented.
- All MetaWorld wrappers enforce valid action ranges.
- Evaluation logs include action saturation statistics.

**Labels**: `core-algorithm`, `metaworld`, `rl`

## 6) Multi-Task Fitness Aggregation and Weighting
**Title**: Add configurable multi-task aggregation beyond simple mean

**Body**:
Current fitness is arithmetic mean of normalized scores. Add alternatives to improve robustness and reduce domination by easier tasks.

**Scope**:
- Implement aggregation options (mean, geometric mean, min-task floor, weighted mean).
- Add config parameters for weighting per task.
- Log per-task and aggregate fitness each generation.

**Acceptance Criteria**:
- Aggregation mode is configurable in run config.
- Per-task fitness trends are persisted for analysis.
- At least one non-mean mode is validated in smoke testing.

**Labels**: `research`, `evaluation`, `multi-task`

## 7) Evaluation Reliability and Statistical Testing
**Title**: Add repeated evaluation protocol and significance testing

**Body**:
Ensure reported improvements are statistically defensible by evaluating best genomes across multiple episodes and seeds.

**Scope**:
- Add N-episode evaluation mode for selected checkpoints/winners.
- Compute confidence intervals and significance tests between methods.
- Export publication-ready summary tables.

**Acceptance Criteria**:
- Final reports include confidence intervals for key metrics.
- Script produces a method-comparison table for thesis figures.
- Protocol is documented in `docs/`.

**Labels**: `research`, `statistics`, `thesis`

## 8) Integrate `tensorneat` as Alternative Evolution Engine
**Title**: Integrate EMI-Group/tensorneat and benchmark against neat-python

**Body**:
Add `tensorneat` ([EMI-Group/tensorneat](https://github.com/EMI-Group/tensorneat)) as an alternative backend and compare performance, scalability, and feature fit for activation-augmented multi-task control.

**Scope**:
- Add dependency management and installation notes for `tensorneat`.
- Build an adapter layer so existing environment/evaluator code can run with either backend.
- Reproduce homogeneous and activation-augmented MetaWorld baselines with `tensorneat`.
- Compare runtime, memory, and learning metrics against `neat-python`.

**Acceptance Criteria**:
- Backend can be switched via config/CLI flag (e.g. `--engine neat-python|tensorneat`).
- At least one successful training run completes with `tensorneat`.
- Benchmark report summarizes tradeoffs and recommendation.

**Labels**: `integration`, `performance`, `backend`, `tensorneat`, `metaworld`

## 9) Test Coverage for Environment Offsets and Normalization
**Title**: Add unit tests for evaluator offsets, padding, and normalization

**Body**:
Protect core multi-environment logic with tests to avoid regressions while refactoring toward MetaWorld and new NEAT backends.

**Scope**:
- Tests for offset assignment correctness in `MultiEnvironmentEvaluator`.
- Tests for padded input construction behavior in environment evaluators.
- Tests for normalization clamping and range handling.

**Acceptance Criteria**:
- Automated tests cover critical evaluator invariants.
- CI/local test command is documented.
- A failing case is demonstrated for at least one intentionally broken variant.

**Labels**: `testing`, `quality`, `core`

## 10) CLI + Config-Driven Experiment Runner
**Title**: Replace hard-coded `scripts/main.py` flow with experiment CLI

**Body**:
Move from a fixed script to a config-driven experiment runner that can launch baseline, ablation, and MetaWorld studies without code edits.

**Scope**:
- Add CLI entrypoint (run, evaluate, summarize).
- Externalize environment list, generations, population, aggregation, and backend choice.
- Add preset configs for MetaWorld task subsets (easy/intermediate/hard).

**Acceptance Criteria**:
- New experiments can be launched by config only.
- MetaWorld presets run without code edits.
- CLI usage is documented in README.

**Labels**: `infrastructure`, `usability`, `experiments`
