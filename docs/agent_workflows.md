# Agent Workflows

The repository includes an optional `tcgd-case-study` skill and a
`tcgd-debugger` custom agent for evidence-backed Tensor Debug and Memory Debug
investigations. They do not change the Python package API or run automatically
inside an application.

## Choose The Entry Point

Use the skill in the current conversation when the main agent should retain
ownership of the investigation. Use the custom agent when reproduction,
collection output, and supplemental analysis would otherwise consume the main
conversation's context.

Both entry points follow the same workflow:

1. define matched semantic measurement points;
2. collect fresh evidence with public Probe or Recorder APIs;
3. analyze public reports before raw PyTorch data;
4. preserve commands, logs, bundles, and environment metadata;
5. separate tool evidence, supplemental work, inference, and tool gaps.

## Codex

Start Codex from the repository root so it discovers the repository-scoped
assets.

- Invoke the skill explicitly with `$tcgd-case-study`.
- Ask Codex to spawn the `tcgd-debugger` custom agent for an isolated case-study
  context.

Codex reads the canonical skill from
`.agents/skills/tcgd-case-study/SKILL.md` and the custom-agent definition from
`.codex/agents/tcgd-debugger.toml`.

## Claude Code

Start Claude Code from the repository root.

- Invoke `/tcgd-case-study` for the skill workflow.
- Mention `@tcgd-debugger` or start with `claude --agent tcgd-debugger` for the
  custom agent.

Claude Code discovers the same canonical skill through the
`.claude/skills/tcgd-case-study` link. Its thin custom-agent definition preloads
that skill instead of copying the workflow.

## External Workloads

The assets are repository-scoped. Start the assistant in this checkout and
provide access to the workload checkout using the assistant's normal
additional-directory mechanism. The skill does not embed cluster-specific
commands. It delegates GPU acquisition and lifecycle policy to resource tools
available in the current environment.

Case-study outputs belong outside the source checkout. Unless the user provides
another location, the skill uses a new timestamped directory under `/tmp`.

## Maintenance

Edit the workflow only in `.agents/skills/tcgd-case-study/`. The Claude skill
path must remain a link to that canonical directory. Keep the Codex and Claude
agent files limited to runtime-specific metadata and the shared role contract.
