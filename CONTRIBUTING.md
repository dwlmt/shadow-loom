# Contributing to Shadow Loom

Thank you for your interest in contributing. Shadow Loom is
**dual-licensed** (AGPLv3 + commercial exception), and to keep that
model working, contributions are accepted under the terms below.

By submitting a pull request, issue, patch, or other contribution
("Contribution") to this repository, you agree to all of the following.

---

## 1. Developer Certificate of Origin (DCO)

Every commit must be signed off under the
[Developer Certificate of Origin v1.1](https://developercertificate.org/),
asserting that you have the right to submit the Contribution under the
project's licences.

In practice, add a `Signed-off-by` trailer to every commit:

```
Signed-off-by: Your Real Name <your.email@example.com>
```

Git will add this automatically when you commit with `-s`:

```bash
git commit -s -m "your message"
```

The DCO is a per-commit assertion of provenance. It is **necessary
but not sufficient** for this project — see section 2.

---

## 2. Contributor License Agreement (CLA)

In addition to the DCO, contributions to Shadow Loom are governed by
the **Shadow Loom Contributor License Agreement** in [`CLA.md`](CLA.md).

The CLA grants the Maintainer a perpetual, irrevocable, worldwide,
royalty-free, non-exclusive licence to relicense your Contribution
under both the AGPLv3 and the commercial licence described in
[`COMMERCIAL-LICENSE.md`](COMMERCIAL-LICENSE.md). You retain
copyright in your Contribution.

You accept the CLA in any of these equivalent ways:

1. **DCO sign-off** (`git commit -s`) on every commit in your pull
   request — see [`CLA.md` § 6.1](CLA.md#6-how-to-accept-this-cla).
2. **CLA-assistant bot.** If the repository has a CLA-assistant bot
   enabled, click the acceptance link on your first pull request.
   The bot records your acceptance for all subsequent contributions
   from the same GitHub account.
3. **Corporate / signed CLA.** For employer-mandated paper trails,
   email a signed copy of [`CLA.md`](CLA.md) to
   `david.wilmot@gmail.com` with subject `Shadow Loom — Corporate CLA`.

This dual-grant is the same model used by Plausible Analytics,
Mattermost, Sentry, Elastic, and Grafana Labs. It is what allows
Shadow Loom to continue offering the commercial exception described
in `COMMERCIAL-LICENSE.md`.

If you cannot or do not wish to grant the licences in `CLA.md`
(for example because of an employer policy), please open an issue
**before** sending a pull request so that an alternative arrangement
can be discussed.

---

## 3. Outbound licence to users

Once merged, your Contribution is distributed to downstream users
under the AGPLv3 (and, where a separate commercial agreement has been
executed, under the commercial licence).

---

## 4. Originality, third-party code, and AI-assisted contributions

You confirm that, to the best of your knowledge:

* the Contribution is your original work, **or** is derived from
  material whose licence is compatible with AGPLv3 and whose source
  and licence are clearly disclosed in the pull request; and
* you have the right to grant the licences in sections 1 and 2.

AI-assisted contributions are permitted, but you remain responsible
for ensuring the submitted code does not infringe third-party rights
and meets the standards above.

---

## 5. Code of conduct

Be kind, be specific, assume good faith. Personal attacks, harassment,
and discriminatory behaviour are not tolerated and may result in
contributions being declined and access being revoked.

---

## 6. Practical workflow

* Open an issue first for non-trivial changes so design can be
  discussed before implementation.
* Keep pull requests focused; one logical change per PR.
* Add or update tests under `tests/` for behavioural changes.
* Run the test suite locally before submitting (`pytest`).
* Update relevant docs under `docs/` when behaviour, APIs, or
  architecture changes.

---

## 7. Reporting security issues

Please do **not** open public issues for security vulnerabilities.
Email `david.wilmot@gmail.com` with subject
`Shadow Loom — Security` and a description of the issue.

---

## 8. Questions

For licensing questions (including obtaining the commercial licence),
see [`COMMERCIAL-LICENSE.md`](COMMERCIAL-LICENSE.md) or contact
`david.wilmot@gmail.com`.
