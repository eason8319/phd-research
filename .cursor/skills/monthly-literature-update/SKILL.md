---
name: monthly-literature-update
description: Monthly literature sweep for the PhD reading list and BibTeX. Collects new papers and preprints, synchronizes docs/literature-survey-reading-list.md with docs/literature-survey.bib, and checks whether previously listed preprints have been formally published. Use when the user asks for 本月文献更新, monthly literature collection, preprint publication check, or to update the literature survey reading list.
---

# Monthly literature update

Canonical procedure: `docs/literature-survey-monthly.md`. Also read `docs/literature-survey-reading-list.md` (scope, category boundaries, and section K) and `docs/literature-survey.bib` (complete bibliographic records and current cite keys). A–D are the four research categories, E is the foundations/evaluation appendix, and K separately holds preprints and technical reports.

Do not add or run project retrieval scripts. Search and preprint checks use the sources and queries in the monthly document.

Follow the monthly document's search window, sources, and query table. Check every K entry for publication regardless of its original submission date. Assign each work to one primary category; keep preprints only in K with a topic label. Once a publication is confirmed, move the existing entry to A–D or E and remove it from K; do not count that move as an addition. Distinguish main-conference, journal, workshop, preprint, technical-report, and non-paper identities.

Synchronize the reading list and BibTeX for every addition, removal, title correction, publication transition, and renumbering. Keep full titles and complete authors in BibTeX even though the reading list displays short names. Verify types, years, venues, DOI/arXiv identifiers, and available volume/issue/pages against primary sources. A preprint-to-publication transition updates the same work's record rather than creating a duplicate; preserve verified open-version identifiers. Never invent missing fields.

Before changing keys, map old keys to new keys by DOI/arXiv/full title and update existing repository citations in the same operation. Never silently retarget a citation after renumbering; resolve references to removed records before deleting their keys. Validate BibTeX syntax, unique keys and works, required fields, and a one-to-one identity match between catalog IDs and BibTeX entries. Inconsistent files mean the update is incomplete.

After synchronization, refresh the list's date, counts, and numbering and the BibTeX header; check source links. Append only one concise row to the monthly update table, including a brief BibTeX synchronization result and any removals or unresolved lookup failures. Do not append narrative sections or per-paper change lists. In chat, briefly report the result and any unresolved lookup limits.

Math in `.md` uses `$...$` / `$$...$$`.
