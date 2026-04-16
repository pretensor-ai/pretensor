# Contributing

Thank you for your interest in contributing to Pretensor OSS! External contributions are welcome — fork the repo, open a PR, and a maintainer will take a look.

## Reporting issues

Please open an issue on GitHub with a clear description of the problem, steps to reproduce, and your environment (Python version, database type, OS).

- **Security issues** — do **not** file a public issue or PR. See [SECURITY.md](SECURITY.md) for private disclosure via GitHub Security Advisories.

## Before you open a PR

- **Small changes** (bug fixes, typos, docs improvements, small features under ~100 lines) — no issue required; just open a PR.
- **Larger features** or changes that touch multiple subsystems — please open an issue first so we can discuss the approach before you invest time writing code.
- **Not sure?** Open an issue and ask.

## Pull requests

1. Fork the repository and create a branch from `main`.
2. To match the maintainer workflow, prefer one of these branch name patterns:
   `feat/<slug>`, `fix/<slug>`, `chore/<slug>`, `docs/<slug>`,
   `test/<slug>`, `refactor/<slug>`, `ci/<slug>`
   (e.g. `feat/add-postgres-support`, `fix/null-column-handling`).
   The `publish/*` branch namespace is reserved for automation; do not use it for contributor work.
3. Make your changes and add tests where appropriate.
4. Set up the repository and run the full check suite before submitting:

   ```bash
   make install
   make verify
   ```

   If `uv` is not installed, create and activate `.venv` first:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   make install
   make verify
   ```

5. Open a pull request against `main` and wait for CI to pass.

External contributors do not need to follow the maintainer PR title convention — a plain Conventional Commits title (see below) is fine. PRs without a linked tracker issue are accepted.

## What a good PR looks like

- [ ] Tests added or updated for the code you touched.
- [ ] `CHANGELOG.md` updated under `[Unreleased]` (or the change is noted as not user-facing, e.g. internal refactors, test-only changes).
- [ ] `make lint`, `make typecheck`, and `make test` all green locally (equivalent to `ruff check src/ tests/`, `pyright src/`, and `pytest`).
- [ ] PR description explains the motivation — what problem this solves or what capability this adds.
- [ ] Existing project architecture invariants are respected (don't reach across layer boundaries, don't introduce circular imports, keep connector I/O behind the connector protocol).

## Developer Certificate of Origin (DCO)

Pretensor uses the [Developer Certificate of Origin](https://developercertificate.org/) rather than a Contributor License Agreement. By signing off on a commit, you certify that you wrote the contribution (or have the right to submit it) under the project's license.

All commits to a PR must be signed off. Add the `Signed-off-by:` trailer automatically with `-s`:

```bash
git commit -s -m "fix: handle null column stats"
```

This appends a line like:

```
Signed-off-by: Jane Doe <jane@example.com>
```

Your sign-off identity must be traceable to you — either your real name with a working email, or your GitHub username paired with your `@users.noreply.github.com` email. Fully anonymous sign-offs are not accepted. Configure git once:

```bash
# Real name + email
git config --global user.name "Jane Doe"
git config --global user.email "jane@example.com"

# Or GitHub username + noreply alias (find your numeric ID at https://github.com/settings/emails)
git config --global user.name "janedoe"
git config --global user.email "12345678+janedoe@users.noreply.github.com"
```

If you forgot to sign off the latest commit, amend it:

```bash
git commit --amend -s --no-edit
git push --force-with-lease
```

For older commits across a branch, rebase and re-sign in one pass:

```bash
git rebase --exec 'git commit --amend --no-edit -s' -i main
git push --force-with-lease
```

CI will block PRs that contain commits without a valid `Signed-off-by:` line.

## Review expectations

- Maintainers aim for a best-effort first response within 7 days. There is no SLA — Pretensor is an alpha project.
- Stale PRs with no contributor response may be closed after ~30 days of inactivity. A closed PR can always be reopened.
- Feedback rounds are collaborative, not adversarial. Maintainers will be explicit about what's required to merge vs. nice-to-have.

## Dedicated Branches

- Maintainers publish updates to dedicated `publish/<slug>-<shortsha>` branches.
- Contributor PRs should always target `main`, never `publish/*`.

## Repository expectations

The public repo CI focuses on the core verification suite (`ruff`, `pyright`, and `pytest`). Maintainers may also review pull requests against the expectations below.

### No secrets (R2)

Never commit API keys, tokens, passwords, or credentials. If you discover a leaked secret, rotate it immediately and report it through GitHub's private security reporting flow.

### No personal paths (R3)

Commits must not contain hard-coded personal home directory paths. Use `$HOME`, `~`, or relative paths instead.

### Noreply email (R4)

Prefer a GitHub noreply email if you do not want your personal address attached to commits:

```bash
git config user.email '<id>+<login>@users.noreply.github.com'
```

### Conventional Commits (R5)

Use [Conventional Commits](https://www.conventionalcommits.org/) for commit titles and PR titles when possible:

```
<type>(<optional-scope>): <description>
```

Allowed types (11):

| Type | Use for | Example |
|------|---------|---------|
| `feat` | New feature | `feat: add Snowflake connector` |
| `fix` | Bug fix | `fix: handle null columns in stats` |
| `docs` | Documentation only | `docs: update install instructions` |
| `test` | Adding or updating tests | `test: add connector edge cases` |
| `refactor` | Code restructuring (no behavior change) | `refactor: extract column parser` |
| `chore` | Maintenance, deps, configs | `chore: bump ruff to 0.15` |
| `ci` | CI/CD changes | `ci: add Python 3.13 to matrix` |
| `build` | Build system or dependencies | `build: switch to hatchling backend` |
| `perf` | Performance improvement | `perf: batch column introspection` |
| `style` | Formatting, whitespace | `style: fix trailing whitespace` |
| `revert` | Revert a previous commit | `revert: undo Snowflake connector` |

Rules: title max 72 chars, lowercase description, no trailing period.

### PR discipline

- One logical change per PR.
- Branch from `main`, keep your branch up to date with `main`.
- All CI checks (ruff, pyright, pytest) must pass before review.

## Code style

- We use **ruff** for linting and formatting, and **pyright** for type checking.
- All public functions should have type annotations.
- Follow existing patterns in the codebase.

## License

By contributing, you agree that your contributions will be licensed under the MIT License (see [LICENSE](LICENSE)).
