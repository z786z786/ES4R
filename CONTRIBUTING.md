# Contributing

Thanks for contributing to `ES4R`.

## Scope

This repository is a research-code release. The main goal of contributions is to improve:

- reproducibility
- documentation clarity
- code quality of the training and inference path
- bug fixes that do not silently change the paper-aligned behavior

Large feature additions or major refactors should be discussed in an issue before implementation.

## Before You Open A PR

Please make sure your change:

- stays within the response-generation scope of this repository
- does not add private data, checkpoints, or machine-specific paths
- keeps the public release structure clean
- updates docs when behavior or interfaces change

## Recommended Workflow

1. Open an issue for bugs, reproduction gaps, or substantial changes.
2. Keep pull requests focused on one topic.
3. Explain the motivation, the change, and any behavior impact.
4. Include validation notes, even if only static checks or a minimal dry run.

## Code Guidelines

- Avoid hard-coded local paths.
- Prefer small, reviewable patches over broad rewrites.
- Preserve existing research behavior unless the PR explicitly fixes a bug.
- Remove debug-only prints, temporary files, and local artifacts before submitting.

## Validation

At minimum, contributors should report one of the following:

- syntax validation
- a minimal preprocessing run
- a minimal inference run
- a training dry run on a tiny sample

If full training is not feasible, say so explicitly in the PR.

## Documentation

Please update the relevant files when needed:

- [README.md](README.md)
- [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md)
- [docs/KEY_MODULES.md](docs/KEY_MODULES.md)
- [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md)

## Pull Request Checklist

- [ ] The change is scoped and reviewable.
- [ ] No private or machine-specific files were added.
- [ ] Documentation was updated if behavior changed.
- [ ] Validation steps and results are included.
- [ ] The PR description explains any limitations or follow-up work.
