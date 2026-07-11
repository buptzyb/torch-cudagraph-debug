# TCGD Investigation: <title>

## Question

<State the exact tensor or memory question and the success criterion.>

## Environment

| Field | Value |
|---|---|
| Workload revision | `<sha>` |
| torch-cudagraph-debug revision | `<sha or version>` |
| Python | `<version>` |
| PyTorch | `<version>` |
| CUDA / driver | `<versions>` |
| GPU | `<model and count>` |
| Output root | `<absolute path>` |

## Experimental Design

<List the controlled baseline/candidate difference and the exact semantic
measurement points or tensor observations. Include warm-up, repetition count,
variant order, and the plan for measuring or bounding instrumentation overhead.>

## Run Matrix

| Run | Variant | Order / repetition | Instrumentation | Completed | Artifact path |
|---|---|---|---|---|---|
| <id> | <baseline/candidate/no-tool> | <A1/B1/...> | <none/tool> | <yes/no> | <absolute path> |

## Commands

```bash
<exact reproducible commands>
```

## Tool Evidence

<Include unedited public API or CLI output, or link each claim to the absolute
path of the saved output. State whether allocator history was enabled.>

## Findings

| Finding | Evidence | Confidence |
|---|---|---|
| <finding> | <public output and points> | <high/medium/low> |

## Supplemental Analysis

<Document raw snapshot queries, custom scripts, or source inspection. For each
item, state why the public package output was insufficient. Write `None` when
no escalation was needed.>

## Inference

<Explain what the combined evidence supports. Distinguish correlation from a
demonstrated cause and absolute memory use from allocator reservation.>

## Tool Gaps

<List information that required leaving the public workflow. Write `None` when
the package answered the complete question.>

## Reproduction Status

- Baseline completed: `<yes/no>`
- Candidate completed: `<yes/no/not applicable>`
- Required observations present: `<yes/no>`
- Artifacts verified: `<yes/no>`
- Instrumentation observer effect assessed: `<yes/no/not applicable>`
- Remaining uncertainty: `<text or none>`
