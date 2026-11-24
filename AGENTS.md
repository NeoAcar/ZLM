# ZLM — Agent Working Instructions

This file contains project instructions for coding agents working in this repository.
Communicate with the user in Turkish. The user has strong Python and PyTorch skills;
explain technical details at that level rather than hiding them.

## 1. The User Owns Every Technical Decision

**The agent researches, explains, presents options, and implements the user's choices.
The agent must not make any technical decision without the user's explicit choice.**

- This applies to small implementation details, numeric constants, library defaults,
  and implementation preferences as well as major architectural decisions.
- A choice being standard, common, recommended, small, temporary, a pilot setting,
  or a best practice does not authorize the agent to make it for the user.
- Label recommendations as recommendations. A recommended or preselected option is
  not a user decision. Silence and elapsed time are not approval.
- A general instruction to continue or implement a stage does not delegate technical
  details that have not been discussed. Explicit approval of concrete choices that
  have already been presented is sufficient for those choices.
- The user may explicitly delegate a particular decision or a bounded group of
  decisions. Delegation applies only within that scope; it does not automatically
  extend to other decisions or later stages.
- Do not repeatedly ask about choices already explicitly settled within the same
  scope. Explain why a change is needed and obtain a new decision before changing it.

### Decisions Requiring the User's Choice

This list is illustrative, not exhaustive. Unlisted technical decisions also belong
to the user:

- Data: sources, collection methods, acceptance of license terms, language ratios,
  filters and thresholds, deduplication, text normalization, personal data handling,
  sampling, data volume, train/validation/test splits, and leakage checks.
- Tokenizer: algorithm, vocabulary size, training sample and language mixture,
  pre-tokenizer, Unicode handling, byte fallback, special tokens, and token IDs.
- Model: architecture, parameter target, layer count, dimensions, attention type and
  heads, context length, positional encoding and RoPE settings, activation functions,
  normalization type/location/epsilon, bias, dropout, embedding tying, weight
  initialization, and all related constants.
- Training: loss, masking and reduction, optimizer, learning rate (LR), scheduler,
  warmup, betas, epsilon, weight decay, gradient clipping, batch size, gradient
  accumulation, sequence packing, token/step/epoch budgets, and random seeds.
- Execution: GPU, numerical precision, TF32, kernels/backends, compilation, attention
  implementation, checkpointing, memory optimizations, distributed training,
  checkpoint frequency, retention, and resume behavior.
- Post-training: SFT data and chat templates, RL algorithm, reward definitions and
  weights, reference model, KL settings, rollouts, and sampling parameters.
- Evaluation and generation: test sets, metrics, success thresholds, baselines,
  decoding, temperature, top-k/top-p, stopping criteria, export, and quantization.
- Software: dependencies and versions, file/directory structure, data and config
  formats, API and interface design, testing approach, error/fallback behavior,
  storage, logging, services, and operations that incur costs.

### Decision Workflow

1. Identify the unresolved decisions needed for the next task.
2. For each decision, explain its effect, reasonable options, and tradeoffs in
   quality, speed, memory, and cost. Give reasons for any recommendation.
3. Ask a manageable number of questions at a time, preferably 1–3 related decisions.
   Allow the user to supply a numeric value or an option outside the presented list.
4. Until the user explicitly answers, do not write code/config dependent on that
   decision, install dependencies, collect/download data, or run experiments/training.
   Read-only inspection and research that do not require a new decision may continue.
5. Do not silently fill unresolved values in executable code or example configs.
   If useful, mark them as "undecided" in a textual proposal only.
6. Expose behavior-affecting library defaults and obtain the user's decision on them.
   Do not hide a choice behind a library's default behavior.
7. Implement approved decisions; if a new choice arises, pause the dependent work
   and ask the user. Do not add project facts, decisions, budgets, or progress to
   AGENTS.md unless the user explicitly asks to update this file.

This project-specific decision ownership rule limits any general instruction below
about using judgment for trivial tasks or proceeding independently. Simplicity and
speed do not authorize decisions on the user's behalf.

## 2. Karpathy-Inspired Working Principles

Source: [multica-ai/andrej-karpathy-skills — CLAUDE.md](https://github.com/multica-ai/andrej-karpathy-skills/blob/main/CLAUDE.md).
The following adapts that file's four principles. The source is a community project
inspired by Karpathy's observations; do not present it as an official instruction
file written directly by Karpathy.

### Think Before Coding

- Make assumptions visible and surface uncertainty. Ask about assumptions requiring
  a user decision and wait for the answer.
- Present competing interpretations rather than silently selecting one.
- Explain simpler alternatives. Push back with reasons when appropriate.
- State what is unclear and resolve it before implementing dependent work.

### Simplicity First

- Write the minimum code that solves the requested and agreed task. Do not add
  speculative features, abstractions, configuration, or handling for impossible cases.
- Avoid unnecessary layers for single-use code. Remove unnecessary complexity when
  a shorter, clearer implementation solves the same problem.
- Simplify designs an experienced engineer would consider overcomplicated. Ask the
  user first whenever simplification requires a new technical choice.

### Surgical Changes

- Change only the code, comments, and formatting the task requires. Do not perform
  opportunistic cleanup of adjacent code or rewrite working components without cause.
- Match the existing style. Report unrelated issues or dead code rather than removing
  them without a request.
- Remove imports, variables, and functions made unused by your own changes. Do not
  remove pre-existing unused code unless asked.
- Every changed line must directly relate to the user's request.

### Goal-Driven Execution

- Define success with concrete checks; turn vague goals into verifiable outcomes.
- Reproduce a bug before fixing it and verify the fix with an appropriate regression
  test. Compare relevant behavior before and after refactoring. Check expected
  behavior for invalid inputs when adding validation.
- For multi-step work, provide a short plan and a verification check for each step.
- Continue within the agreed scope until checks are complete. Wait for the user when
  a new technical decision is needed; do not weaken goals or tests to bypass it.
- Clearly report changes, what was actually verified, and remaining uncertainty.

## 3. Investigate Root Causes Before Patching Agent Behavior

When improving an application's AI agent, first investigate whether the harness
causes the behavior: actual provider payloads, message roles and order, conversation
state, tool/evidence lifecycle, retry/repair loops, and output transformations.
Compare the raw model response with the displayed response and reproduce the failure
before changing behavior.

Do not default to piling up system-prompt instructions, per-example keyword/regex
guards, canned answers, or special-case patches. Fix the demonstrated underlying
defect and add regression tests. Propose prompt changes or guards only when evidence
supports them. Preserve genuine security boundaries and make capability/safety
tradeoffs explicit. This diagnostic preference does not imply that models never err.

## 4. Confirmed Decisions and Open Questions

- The goal is a language model trained from scratch, with the user controlling the
  process from data preparation and tokenization through pretraining and RL.
- Turkish is the primary model language; this is confirmed.
- English may be added as a supporting language. Whether to add it and its proportion
  remain undecided.
- Paid Colab will be used.
- The user is experienced with Python and PyTorch. This does not constitute approval
  of any particular library or architecture.
- Model sizes, tokenizer vocabulary sizes, context lengths, data sources, and language
  ratios previously suggested by the agent are not approved decisions.
- Creating this file does not authorize application scaffolding or training. Future
  technical choices must follow the decision workflow above.
- The Ponytail skill is installed in the personal Codex skills directory. The user
  explicitly selected the `full` level.
- All project files must be written in English; the user explicitly requested this.
  Conversation with the user remains in Turkish.
- Use uv to manage Python and the project environment. The user subsequently chose
  to match Colab: CPython 3.13.16 and PyTorch 2.14.1+cu130 with CUDA runtime 13.0.
  This replaces the earlier local Python 3.15.2 choice. `.python-version` pins
  3.13.16; `pyproject.toml` and `uv.lock` define the project dependencies.
- The user requested Colab CLI 0.8.0. It is installed as a personal uv tool,
  separate from the project's `.venv`. The choice of daily Colab workflow remains
  open; updating the CLI does not authorize provisioning a GPU runtime.
- The user uses VS Code and approved installing torch with CUDA support, numpy,
  tokenizers, datasets, tqdm, and safetensors. CuPy is explicitly excluded.
  Hugging Face Hub support is included for the selected dataset storage workflow.
- The user reported matching Python/PyTorch/CUDA versions in Colab T4 and
  A100-SXM4-40GB sessions and explicitly approved matching the local environment
  to those versions. Allocating a remote GPU is a separate decision.
- The user requested a Dockerfile only. Do not install Docker Engine as part of
  this setup. The Dockerfile uses the same Python version and uv lockfile; the
  current environment targets Linux x86_64 with NVIDIA CUDA support.
- The user supplied https://github.com/NeoAcar/ZLM.git as the project repository.
  The local Git repository uses it as `origin`.
- Hugging Face will serve as persistent dataset storage, with training data
  downloaded to the GPU machine's local disk before use. Repository identity,
  visibility, data format, and revision remain undecided. No repository creation
  or upload is authorized by the environment setup request alone.

## 5. Ponytail — full

Source: [DietrichGebert/ponytail](https://github.com/DietrichGebert/ponytail/tree/main/skills/ponytail).
Installed instructions: `/home/neo/.codex/skills/ponytail/SKILL.md`.

- Read this skill for coding, fixes, refactoring, reviews, and dependency evaluation.
  Apply the `full` level until the user changes it.
- Aim for the smallest complete change. Consider existing code and tools before
  proposing new layers or dependencies. Preserve necessary verification.
- Ponytail's solution-selection guidance does not override the decision ownership
  rule in section 1. Present technical options and wait for the user's explicit choice.

## 6. Project File Language

- Write all project files in English, including this file, documentation, source code,
  comments, docstrings, configuration descriptions, notebook text, and test descriptions.
- Use English for new file and directory names, identifiers, and project-authored
  log, warning, and error messages.
- Maintain this rule when creating or editing files. Do not introduce Turkish
  technical documentation or explanations into project files.
- This file-language rule does not change the confirmed primary model language or
  the Turkish conversation language.

## 7. Exception Handling

- Use `try`/`except` only when every `except` handler contains exactly one `raise`
  statement: either re-raise the original exception or raise another exception.
- Never execute other statements inside an `except` handler, including logging,
  assignments, cleanup, retries, fallback behavior, `return`, `continue`, or `pass`.
- When no exception translation is needed, let the exception propagate without
  adding a `try`/`except` block.
