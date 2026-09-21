# Project instructions

These instructions apply to every Codex conversation and agent working in this workspace, including subagents when delegation is authorized.

## Cloud repository and synchronization

- The project's cloud repository is [eason8319/phd-research on GitHub](https://github.com/eason8319/phd-research).
- When the user asks to synchronize project content with the cloud, consult this section and use this repository as the synchronization target.
- Before synchronizing, check the local Git setup, remote configuration, and current changes to determine the steps needed for the requested synchronization.

## Cursor rule compatibility

- Before starting project work, read the current rules in this workspace's `.cursor/rules/` directory.
- Always follow rules marked `alwaysApply: true`. Apply other rules according to their file patterns, stated scope, and the current task.
- The Cursor rule files are the source of truth for project conventions. Re-read relevant rules when they change; do not rely on remembered copies from other conversations.
- When delegating authorized work, pass along the workspace root and applicable rules, and require the agent to read this file and the relevant Cursor rules before acting.
- These project rules remain subject to higher-priority system/developer instructions and the user's explicit instructions.

## Literature survey

Monthly collection of papers and preprints uses `docs/literature-survey-monthly.md` (search constraints and the monthly log) and synchronizes `docs/literature-survey-reading-list.md` with `docs/literature-survey.bib`. Additions, removals, publication changes, and renumbering must update both files; validate one-to-one IDs, bibliographic metadata, and existing citations before completion. Rule: `.cursor/rules/literature-survey.mdc`. Trigger phrase: `做本月文献更新`. Do not add retrieval scripts to this repository.

## Workspace isolation

The following requirements mirror `.cursor/rules/workspace-isolation.mdc` so they are also visible directly in Codex's project instructions.

This repository is the only allowed working context.

- Read, search, and edit files only under the current workspace root.
- Do not open, search, or cite other local projects, other workspace folders, or paths outside this repo (except the current workspace's own `.cursor/` project files if needed for this repo).
- Do not search, read, or quote past conversations, agent transcripts, chat history, or cached cloud chats.
- Do not use the user's personal agent store, global memories, or other chats as background. If a fact is not in this repo or the current thread, ask or look in this repo.
- Do not treat previously viewed files from other projects as relevant context.
- If information seems to exist only in an old chat, say so and work from this repo instead of retrieving that chat.
