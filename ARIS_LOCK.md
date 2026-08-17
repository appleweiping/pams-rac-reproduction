# ARIS workflow lock

This paper branch uses the official Auto-Research-In-Sleep (ARIS) repository:

- Upstream: <https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep>
- Branch: `main`
- Locked commit: `e12e07c7b85ee1a4dc07e5463089aa16836af2bf`
- Installed: 2026-08-11/12 on Windows through the official Codex installer
- Assurance requested: `submission`

The project-local installation uses the Codex-native skill mirror. The
paper-plan, paper-figure, paper-write, and auto-paper-improvement-loop entries
are overlaid by the official Claude-review package. Installed paper workflow
skills include paper-writing, paper-plan, paper-figure, figure-spec,
paper-write, paper-compile, experiment-audit, result-to-claim,
paper-claim-audit, citation-audit, proof-checker, kill-argument,
integrity-forensics, and their shared references.

The `claude-review` MCP server was installed and registered, and the Claude CLI
reported local authentication through a configured proxy. Its live reviewer
health probe did not complete, and the newly registered MCP was not exposed in
the already-running Codex session. Therefore this package must record semantic
reviews as same-family/provisional until a restarted session completes a
healthy cross-family probe. No final report may claim cross-family acceptance
or `submission-ready: yes` on the basis of this run.

Local junctions, installer manifests, proxy logs, and machine-specific absolute
paths are intentionally ignored by Git. This lock file is the portable version
record; updating ARIS requires changing the pinned commit and rerunning its
official reconcile installer before any later paper audit.
