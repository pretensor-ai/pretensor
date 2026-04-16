# Security Policy

## Supported versions

Pretensor is currently in **pre-release development** and is not yet published to PyPI. Until the first packaged release, security fixes land on the latest supported state of this repository.

| Version | Supported |
|---------|-----------|
| Current `main` branch snapshot | Yes |
| Older commits, forks, or unpublished snapshots | No |

Once Pretensor begins shipping packaged releases, this policy will be updated to cover a defined support window.

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Report privately through GitHub Security Advisories:

> [Open a private security advisory](https://github.com/pretensor-ai/pretensor/security/advisories/new)

Include:

- A description of the issue and the affected component.
- Steps to reproduce, ideally with a minimal proof of concept.
- The git commit SHA or package metadata you tested against (for example `git rev-parse HEAD` from a source checkout).
- Your assessment of impact (confidentiality, integrity, availability).

## What to expect

Pretensor is maintained by a small team on a best-effort basis during pre-release development. We aim to:

- Acknowledge your report within **5 business days**.
- Provide a status update within **14 days** of acknowledgement.
- Coordinate a public disclosure after a fix is available, crediting the reporter (unless anonymity is requested).

## Scope

In scope:

- The `pretensor` Python package and CLI when installed from this repository.
- The MCP server entrypoint (`pretensor serve`).
- Default connector implementations shipped in this repository.

Examples of vulnerabilities we want to hear about:

- Credential leakage from DSN encryption or connection storage paths.
- Arbitrary code execution through the CLI, indexing flows, or MCP tool handlers.
- Bypass of the read-only guard on the `cypher` MCP tool that lets a caller mutate the local Kuzu graph.

Out of scope:

- Third-party databases, MCP clients, or drivers that Pretensor integrates with.
- Vulnerabilities that require physical access or compromise of the host machine running Pretensor.
- Denial-of-service issues that result from supplying deliberately large or malformed schemas to the indexer — operator responsibility.

## Safe harbor

We will not pursue legal action against researchers who:

- Make a good-faith effort to avoid privacy violations, destruction of data, and interruption of services.
- Report vulnerabilities through the channel above and give us a reasonable window to respond before public disclosure.
- Do not exploit a vulnerability beyond what is necessary to confirm it.
