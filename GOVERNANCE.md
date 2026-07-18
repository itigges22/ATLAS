# Governance

ATLAS is currently a **single-maintainer project** (see
[MAINTAINERS.md](MAINTAINERS.md)). This document states how decisions
are made today and what changes as maintainership grows — it does not
pretend a committee exists.

## Decision making

- Day-to-day decisions (bug fixes, refactors, docs) are made by the
  maintainer merging to `dev`.
- Significant decisions — architecture changes, new dependencies,
  security-model changes, release policy, anything altering the
  documented support matrix — are recorded as an ADR under `docs/adr/`
  before or with the change.
- Community input happens in GitHub issues. Substantive disagreement is
  resolved by the maintainer; the reasoning is written down in the
  issue or ADR.

## Change flow

All changes land on `dev`, are promoted to `staging`, then fast-forward
to `main`. Releases are tagged from `main` and published only through
the CI pipeline (immutable `sha-*` images, tag promotion gated on the
tests workflow).

Both `main` and `dev` are protected: force-pushes and deletions are
blocked, the full CI check set (21 contexts — Go/Python/lens/contract/
E2E tests, lint, compose, install matrix, CodeQL) is required, and
conversation resolution is required on PRs. Admin enforcement is
**off** by design: as a solo maintainer, the lead direct-pushes to
`dev` and fast-forwards `main` (both are ordinary non-force pushes that
protection permits), while any external contributor PR must pass every
required check before merge. Human PR review is **not** required — with
one maintainer there is no second reviewer, so requiring it would
deadlock the branch; this relaxation is revisited the moment a second
maintainer exists (see MAINTAINERS.md open seats). This is the honest
maximum protection for a single-maintainer project: it stops accidents
and enforces the test gate without pretending a review quorum exists.

## Release authority

The maintainer cuts releases. A release requires the checklist in
`docs/RELEASE.md` (all CI green, changelog/version consistency, hardware
smoke). **Bus-factor status: 1.** Growing to a second release-capable
maintainer is an explicit project goal and a precondition for calling
the project mature — until then, the honest statement is that the
maintainer disappearing stops releases (the AGPL license and public
registry artifacts allow forks to continue).

## Security

Security reports follow [SECURITY.md](SECURITY.md). Security fixes may
land ahead of normal review flow, with the ADR/notes written
retroactively.

## Becoming a maintainer

Sustained, high-quality contributions in a subsystem (see CODEOWNERS
for the subsystem map) plus demonstrated judgment on
security-relevant changes. The path: triage rights → subsystem review
ownership → release capability. The maintainer actively wants this to
happen; if you're interested, open an issue saying which subsystem.

## Inactive-maintainer / succession policy

If the sole maintainer is unresponsive for 60+ days, the project should
be considered fork-friendly: everything needed to continue (build
pipeline, registry contracts, artifact hashes, release process docs) is
in-repo by design, and the HF artifact repos are mirrored by their
hashes in the registry so a fork can re-host and re-pin.
