# Release Checklist

Use this checklist before making the repository public or cutting a tagged release.

## Repository Hygiene

- [ ] Remove private notes, interview materials, local backups, and editor settings.
- [ ] Remove generated outputs, caches, logs, and temporary artifacts.
- [ ] Confirm `.gitignore` covers local-only files.
- [ ] Add the final public repository URL to [CITATION.cff](/Users/z786/Workspace/caes_original/CITATION.cff) after publishing.
- [ ] Remove stray `.DS_Store` files.

## Documentation

- [ ] README matches the actual repository scope.
- [ ] Data schema is documented in [docs/DATA_FORMAT.md](/Users/z786/Workspace/caes_original/docs/DATA_FORMAT.md).
- [ ] Architecture is documented in [docs/KEY_MODULES.md](/Users/z786/Workspace/caes_original/docs/KEY_MODULES.md).
- [ ] Training, inference, and demo commands are runnable in principle.
- [ ] Licensing and citation information are present.

## Code Quality

- [ ] Remove debug-only prints and temporary code paths.
- [ ] Remove hard-coded local dataset or checkpoint paths.
- [ ] Confirm entry scripts still import correctly.
- [ ] Check that deleted files are not referenced by docs or scripts.

## Validation

- [ ] Run syntax checks on modified Python files.
- [ ] Run a minimal preprocess command on a toy sample if available.
- [ ] Run a minimal inference command if a checkpoint is available.
- [ ] Record any known gaps in the README or docs instead of hiding them.

## GitHub Setup

- [ ] Add repository description and topics.
- [ ] Add the paper link to the repository homepage.
- [ ] Enable Issues if you want external bug reports.
- [ ] Review issue templates and PR template.
- [ ] Create the first release note or pinned announcement if needed.
