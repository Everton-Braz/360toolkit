# Best Automated Testing Strategy for 360toolkit

## Overview

The best solution for 360toolkit is to model the pipeline as structured configuration, then automate testing at three levels: many unit tests for decision logic, a smaller set of integration tests for ffmpeg and SDK execution, and a very small number of end-to-end UI tests for critical user flows.[cite:16][cite:17] This follows the test pyramid, which favors a large base of cheap, fast tests and limits expensive, brittle UI automation.[cite:16][cite:17]

For a pipeline-driven application such as 360toolkit, this approach is better than manual testing because the number of possible combinations grows quickly across extractor choice, frame rate, lens mode, split behavior, masking, and quality settings.[cite:7][cite:10] Rather than manually checking every permutation, the software should generate representative test cases automatically and verify the expected outputs and invariants.[cite:1][cite:2][cite:6]

## Recommended Architecture

The core design decision is to separate the application into two parts:

- Pure pipeline planning logic, which receives a configuration and decides what should happen.
- Execution adapters, which call ffmpeg, the Insta360 SDK, file-system operations, and UI workflows.

This separation makes the most important logic testable without video files, external binaries, or GUI interaction.[cite:16][cite:17] It also aligns well with `pytest`, which is widely used in Python because it supports simple test functions, plain assertions, fixtures, and powerful parameterization with less boilerplate than `unittest`.[cite:21][cite:24]

A practical model is a single `PipelineConfig` object containing the main choices:

| Field | Examples | Why it matters |
|---|---|---|
| `extractor` | `sdk`, `ffmpeg` | Changes backend behavior and command generation. |
| `fps` | `0.5`, `1`, `2`, `5`, `10` | Changes frame count and sampling rules. |
| `lens_mode` | `both`, `left`, `right` | Changes number and type of outputs. |
| `quality` | `preview`, `medium`, `high` | Changes encoding or extraction settings. |
| `split` | `true`, `false` | Changes whether outputs are combined or separated. |
| `mask` | `true`, `false` | Changes post-processing behavior. |
| `output_format` | `jpg`, `png` | Changes files produced and expected extensions. |

Once the pipeline is represented as data, tests can validate the planner without launching the full app.[cite:1][cite:2]

## Why `pytest` is the best fit

`pytest` is the best primary test framework for 360toolkit because it makes configuration-driven tests easy to write and scale.[cite:1][cite:2][cite:21] It supports parameterized tests, fixtures, temporary directories, and readable assertions, which are all useful for pipeline software that must evaluate many input combinations.[cite:1][cite:2]

`unittest` is the built-in Python framework and is still valid, but it is typically more verbose and more class-oriented.[cite:21][cite:24] For this project, the most useful workflow is usually `pytest` as the main framework, with mocks and temporary file helpers layered on top as needed.[cite:21][cite:24]

## Best test strategy

### 1. Unit tests for pipeline decisions

Unit tests should cover all pure logic that does not need real files or external tools.[cite:16][cite:17] This includes validation, default values, command planning, output naming, frame count estimation, lens routing, split decisions, and mask rules.[cite:16][cite:17]

Examples of good unit-test targets:

- Invalid configuration combinations are rejected early.
- `lens_mode='left'` never schedules right-lens output.
- `split=False` never creates split jobs.
- The selected extractor changes the backend command builder.
- Output filenames are stable and deterministic.

These tests should be the majority of the suite because they are the fastest and most reliable to run repeatedly.[cite:16][cite:17]

### 2. Parameterized tests for known scenarios

Pytest parameterization allows one test function to run against many predefined inputs and expected results.[cite:1][cite:2] This is the best way to encode important combinations that are already known from experience, bug reports, or domain knowledge.[cite:1][cite:2]

Examples:

- Common user presets.
- Previously broken configurations.
- Boundary values such as the minimum and maximum allowed fps.
- Cases where masking, split behavior, or lens selection interact in special ways.

This gives strong regression protection because once a bug is fixed, its configuration can be added as another permanent parameterized test case.[cite:1][cite:2]

### 3. Pairwise testing for combinatorial explosion

When the configuration space becomes too large, pairwise testing is usually the best compromise between cost and coverage.[cite:7][cite:10] Instead of running every possible combination, pairwise generation creates a smaller suite that still covers every pair of parameter values across the model.[cite:7][cite:10]

This is especially useful for 360toolkit because many bugs come from interactions between two options, such as extractor plus lens mode, or mask plus split, rather than from every parameter interacting at once.[cite:7][cite:10] Pairwise testing does not replace carefully chosen edge cases, but it greatly reduces execution time while still giving broad compatibility coverage.[cite:7][cite:10]

### 4. Property-based testing for invariants

Property-based testing with Hypothesis is valuable for finding edge cases that are hard to predict manually.[cite:6][cite:9][cite:15] Instead of listing examples one by one, tests define invariants that must always hold, and Hypothesis generates many inputs automatically to try to break them.[cite:6][cite:9][cite:15]

Good invariants for 360toolkit include:

- Planned output paths are unique.
- Generated frame indexes are ordered and non-negative.
- A valid single-lens config never produces two lens outputs.
- A config marked `split=False` never schedules split artifacts.
- Validation failures happen before any real work starts.

This style is powerful for file naming, config validation, planning logic, and other pure functions where a large input space exists.[cite:6][cite:9][cite:15]

### 5. Integration tests for external tools

Integration tests should verify that the adapters for ffmpeg and the Insta360 SDK work correctly with real sample inputs.[cite:16][cite:17] These tests are slower and more fragile than unit tests, so they should be smaller in number and focused on critical paths.[cite:16][cite:17]

A strong integration-test set usually includes:

- One or two tiny INSV samples.
- Known expected outputs or metadata.
- Tests that run the real backend command.
- Assertions for exit code, output existence, frame count, naming, and metadata.

These tests confirm that the application still works against the actual external tools after refactors, environment changes, or packaging changes.[cite:16][cite:17]

### 6. Few end-to-end UI tests

UI automation should exist, but only for critical workflows.[cite:16][cite:17] End-to-end tests are useful for proving that the app can be launched, a pipeline can be configured, a job can be started, and results appear correctly in the interface.[cite:16][cite:17]

The suite should stay small, for example:

- Create a job with ffmpeg and both lenses.
- Create a job with one lens and masking.
- Load a saved preset and run it.
- Verify failure handling for an invalid input file.

This keeps UI automation valuable without allowing it to become the main testing strategy.[cite:16][cite:17]

## Practical implementation plan

### Pipeline model

The pipeline should be described in data rather than spread across UI event handlers. A good implementation is a config dataclass plus a planner that converts config into an execution plan.[cite:1][cite:2]

Example structure:

```python
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class PipelineConfig:
    extractor: Literal["sdk", "ffmpeg"]
    fps: float
    lens_mode: Literal["both", "left", "right"]
    quality: Literal["preview", "medium", "high"]
    split: bool
    mask: bool
    output_format: Literal["jpg", "png"]

@dataclass(frozen=True)
class ExecutionPlan:
    commands: list[str]
    expected_outputs: list[str]
    warnings: list[str]
```

Then a planner function such as `build_plan(config) -> ExecutionPlan` can become the main target of unit tests, parameterized tests, pairwise tests, and property-based tests.[cite:1][cite:2][cite:6]

### Suggested test layout

```text
tests/
  unit/
    test_config_validation.py
    test_pipeline_planner.py
    test_output_naming.py
    test_frame_sampling.py
    test_mask_rules.py
  integration/
    test_ffmpeg_extract.py
    test_sdk_extract.py
    test_end_to_end_pipeline.py
  e2e/
    test_gui_smoke.py
    test_gui_run_job.py
  data/
    tiny_sample.insv
    expected/
```

This layout keeps pure logic, backend integration, and full-app checks clearly separated.[cite:16][cite:17]

### Example `pytest` parameterization

```python
import pytest

@pytest.mark.parametrize(
    "config,expected_output_count",
    [
        (PipelineConfig("ffmpeg", 1, "both",  "high",   False, False, "jpg"), 2),
        (PipelineConfig("ffmpeg", 1, "left",  "high",   False, False, "jpg"), 1),
        (PipelineConfig("sdk",    2, "right", "medium", True,  True,  "png"), 1),
    ],
)
def test_expected_output_count(config, expected_output_count):
    plan = build_plan(config)
    assert len(plan.expected_outputs) == expected_output_count
```

This pattern is useful for all combinations that are known to matter.[cite:1][cite:2]

### Example property-based test idea

```python
from hypothesis import given, strategies as st

@given(st.text(min_size=1), st.booleans())
def test_output_name_is_never_empty(base_name, split):
    result = make_output_name(base_name, split=split)
    assert result
```

The specific strategies should be adapted to real config fields, but the main idea is to express system rules rather than single examples.[cite:6][cite:9][cite:15]

## Coverage priorities

Not all configurations deserve equal attention. The fastest reliable strategy is to prioritize tests in this order:

1. Pure decision logic and validators, because these are cheap and cover much of the application behavior.[cite:16][cite:17]
2. Common presets and previously broken cases, because these give high regression value.[cite:1][cite:2]
3. Pairwise coverage of the configuration model, because this controls combinatorial growth efficiently.[cite:7][cite:10]
4. Real-tool integration with tiny sample files, because this validates the adapters and packaging.[cite:16][cite:17]
5. A few UI smoke tests, because they protect against broken workflows without making the suite slow and fragile.[cite:16][cite:17]

This ordering gives faster feedback and better long-term maintainability than trying to automate every scenario through the GUI.[cite:16][cite:17]

## What other teams typically do

Most mature teams do not attempt to test every behavior through manual steps or only through end-to-end automation.[cite:16][cite:17] They push logic downward into testable units, use parameterization for known cases, use broader generative techniques when the input space is large, and keep UI tests limited to essential flows.[cite:1][cite:2][cite:6][cite:16]

In Python projects, the common practical stack is `pytest` for the runner, fixtures for setup, mocks for isolation, temporary directories for file assertions, and optionally Hypothesis for generative coverage.[cite:1][cite:2][cite:6][cite:15] This combination is particularly suitable for tooling applications and data-processing pipelines because it supports both precise regression tests and broad automatic exploration of edge cases.[cite:1][cite:2][cite:6]

## Final recommendation

The best solution for 360toolkit is:

- Use `pytest` as the main framework.[cite:1][cite:2][cite:21]
- Refactor the app around a `PipelineConfig` and a pure `build_plan()` function.[cite:1][cite:2]
- Write many unit tests for planning and validation logic.[cite:16][cite:17]
- Add parameterized tests for known important scenarios.[cite:1][cite:2]
- Add pairwise-generated configuration tests for broad coverage without explosion.[cite:7][cite:10]
- Add Hypothesis tests for invariants in naming, validation, and planning.[cite:6][cite:9][cite:15]
- Keep a small number of real integration tests for ffmpeg and SDK execution.[cite:16][cite:17]
- Keep only a few high-value UI end-to-end tests.[cite:16][cite:17]

This is the most balanced solution because it is fast enough to run often, reliable enough to catch regressions, and scalable enough to handle the large number of possible pipeline configurations in 360toolkit.[cite:16][cite:17][cite:7]
