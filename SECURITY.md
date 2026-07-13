# Security Policy

## Supported versions

Pretensor is currently in **alpha** on PyPI (`pip install pretensor`). Security fixes are released against the latest alpha on PyPI; pin to a specific version (for example `pretensor==<version>`) until `1.0.0` if you need a stable target.

| Version | Supported |
|---------|-----------|
| Latest pre-release on PyPI | Yes |
| Older pre-releases | No — upgrade to the latest alpha for fixes |
| Source checkouts ahead of PyPI | Best effort, until the next alpha rolls forward |

Once Pretensor reaches `1.0.0`, this section will be updated to describe a defined support window.

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Report privately through GitHub Security Advisories:

> [Open a private security advisory](https://github.com/pretensor-ai/pretensor/security/advisories/new)

Include:

- A description of the issue and the affected component.
- Steps to reproduce, ideally with a minimal proof of concept.
- The version or commit you tested against — for a PyPI install, `pip show pretensor`; for a source checkout, `git rev-parse HEAD`.
- Your assessment of impact (confidentiality, integrity, availability).

## What to expect

Pretensor is maintained by a small team on a best-effort basis during pre-release development. We aim to:

- Acknowledge your report within **5 business days**.
- Provide a status update within **14 days** of acknowledgement.
- Coordinate a public disclosure after a fix is available, crediting the reporter (unless anonymity is requested).

## Scope

In scope:

- The `pretensor` Python package and CLI, whether installed from PyPI or from a source checkout of this repository.
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

## Credential storage and key rotation

Connection strings (DSNs) are encrypted at rest by default. On `pretensor index`
and `pretensor reindex`, the DSN is encrypted with a per-state-dir symmetric key
before it is written to the registry, so the registry never stores a plaintext
password for connections created with current versions.

Files under the state directory (default `.pretensor/`):

- `keystore` — the symmetric key that decrypts every stored DSN. Written
  owner-only (`0o600`); its parent directory is created `0o700`. On a shared or
  multi-user host, keeping these permissions intact is what prevents other local
  users from reading your database credentials. Pretensor warns when it loads an
  existing keystore (during `index`/`reindex`/`add` or the first MCP tool call
  that decrypts a DSN) if its permissions are looser than `0o600`.
- `registry.json` — the connection registry. Written `0o600`. Entries created
  before default-on encryption may still contain a cleartext DSN; re-running
  `pretensor reindex` re-encrypts them (it never downgrades an entry to
  cleartext).

These permissions are enforced on POSIX systems. On Windows, file modes are not
applied the same way — protect the state directory with filesystem ACLs.

### Rotating the encryption key

There is a single symmetric key. To rotate it (for example, if you suspect the
keystore was exposed):

1. Note the connections you have indexed (`pretensor list`).
2. Delete the keystore: `rm .pretensor/keystore`.
3. Re-add each connection (`pretensor add …`) or re-run `pretensor index …`.
   A fresh keystore is generated and the DSNs are re-encrypted under the new key.

Because the operator is the trusted party and a manual rotation path exists,
Pretensor does not ship dedicated key-rotation tooling. Rotate the underlying
**database** credentials through your database, not through Pretensor.

## Safe harbor

We will not pursue legal action against researchers who:

- Make a good-faith effort to avoid privacy violations, destruction of data, and interruption of services.
- Report vulnerabilities through the channel above and give us a reasonable window to respond before public disclosure.
- Do not exploit a vulnerability beyond what is necessary to confirm it.
