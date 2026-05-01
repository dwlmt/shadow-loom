# Shadow Loom — Commercial License (AGPLv3 Exception)

**Copyright © 2025–2026 David Rae Wilmot. All rights reserved.**

Shadow Loom is dual-licensed:

1. As **free / open-source software** under the
   [GNU Affero General Public License v3.0](LICENSE) (AGPLv3); or
2. Under a **commercial license** purchased from the copyright holder,
   which grants an exception to the AGPLv3 obligations described below.

This document explains the commercial offering. The full open-source
license is in [`LICENSE`](LICENSE); the copyright statement and the
choice between the two licenses is in [`COPYRIGHT`](COPYRIGHT).

---

## 1. Why a commercial license exists

The default license, AGPLv3, is a strong copyleft license. In summary:

* If you **modify** Shadow Loom, your modifications must also be
  released under AGPLv3.
* If you make Shadow Loom (modified or unmodified) **available to users
  over a network** — for example as part of a SaaS product, an internal
  web service, an MCP server exposed to external agents, or a hosted
  API — you must offer **the complete corresponding source code of the
  entire combined work** to those users under AGPLv3.
* "The entire combined work" includes any proprietary code that links
  to, embeds, or is intimately combined with Shadow Loom — not just the
  Shadow Loom files themselves.

For many uses (research, personal projects, fully-open products) AGPLv3
is exactly what you want. For commercial deployments where the
network-use disclosure obligation is incompatible with your business
model, the **commercial license** removes those obligations in exchange
for a fee.

This is a long-standing pattern (MongoDB, Sentry, Plausible Analytics,
Bitwarden, Mattermost, MinIO, Grafana Labs, CockroachDB, Neo4j and
others all use a variant of it). It is sometimes called "dual licensing"
or an "open-core / commercial exception".

---

## 2. What the commercial license grants

Subject to the terms of an executed commercial agreement, the
commercial license grants the licensee:

* The right to use, modify, and embed Shadow Loom in proprietary
  products **without** triggering AGPLv3 § 5 (Conveying Modified
  Source Versions) or § 13 (Remote Network Interaction).
* The right to **keep modifications proprietary** and to ship them as
  closed-source binaries, container images, or hosted services.
* The right to combine Shadow Loom with proprietary code without that
  combined work becoming a "covered work" under AGPLv3 § 0.
* The right to **offer Shadow Loom as part of a SaaS / hosted service
  without disclosing the corresponding source** to end users.
* Optional support, indemnification, and warranty terms, negotiated
  separately and stated in the agreement.

The commercial license is **non-exclusive, non-transferable** (without
written consent), and granted for the entity, deployments, and term
specified in the executed agreement.

It does **not** grant trademark rights to the "Shadow Loom" name or
logo beyond what is required to identify the unmodified product. It
does **not** grant rights to relicense Shadow Loom to third parties.

---

## 3. Who needs the commercial license

You **need** the commercial license if **any** of the following apply
and you are unable or unwilling to comply with AGPLv3 § 13:

* You are running Shadow Loom (or a derivative) as a hosted service
  available to users outside your own organisation, and cannot publish
  the corresponding source of your full stack.
* You are embedding Shadow Loom in a closed-source product distributed
  to customers.
* You are combining Shadow Loom with proprietary modules that you do
  not wish to release under AGPLv3.
* You are running Shadow Loom in a regulated environment whose
  procurement rules forbid AGPL-licensed components.

You **do not** need the commercial license if **all** of the following
apply:

* Your use of Shadow Loom is internal to your own organisation, and
  you do not expose it as a service to external users; **or**
* You are willing to license your full combined work under AGPLv3 and
  to make the corresponding source available to your users in
  compliance with AGPLv3 § 13.

When in doubt, ask. Compliance is cheaper than a dispute.

---

## 4. How to obtain the commercial license

Contact the copyright holder:

> **David Rae Wilmot**
> david.wilmot@gmail.com
> Subject line: `Shadow Loom — Commercial License Enquiry`

In your message, please include:

1. The **legal name** of the licensee entity.
2. A short **description of the intended deployment** (product, hosted
   service, internal tool, etc.) and the **expected scale** (number of
   end users, number of nodes, internal vs external).
3. Whether you want **support / SLA / indemnification** in scope.
4. The **term** you are interested in (typical: 12 months, renewable).

A standard commercial agreement and pricing will be sent in reply.
Pricing scales with deployment size and support level. Discounts are
available for early-stage startups and academic spin-outs; non-profits
and charities may qualify for a no-fee commercial exception.

---

## 5. Contributor Licensing

Contributions to Shadow Loom are governed by the **Shadow Loom
Contributor License Agreement** in [`CLA.md`](CLA.md), accepted
through any of the equivalent mechanisms listed in
[`CLA.md` § 6](CLA.md#6-how-to-accept-this-cla) (DCO sign-off,
CLA-assistant bot, or signed corporate CLA), together with the
[Developer Certificate of Origin](https://developercertificate.org/).

The CLA grants the copyright holder a perpetual, irrevocable,
worldwide, royalty-free, non-exclusive licence to redistribute every
Contribution under both the AGPLv3 **and** the commercial licence
described in this document. Contributors retain copyright in their
Contributions.

This dual-grant is what enables Shadow Loom to keep the dual-licence
model working as the project grows. It is the same model used by
Plausible Analytics, Mattermost, Sentry, Elastic, and Grafana Labs.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the practical workflow
and [`CLA.md`](CLA.md) for the full text of the CLA. If you are
unable or unwilling to grant this licence-back, please open an issue
before submitting code so that an alternative arrangement can be
discussed.

---

## 6. Commercial-tier features (open-core add-ons)

The **AGPLv3 build** of Shadow Loom is fully functional for research,
self-hosting, and AGPL-compatible commercial use; nothing critical to
the published research is gated.

In addition, the commercial licence **may** bundle (subject to the
executed agreement and the chosen tier) the following enterprise
features, which are not part of the AGPLv3 distribution:

* **White-labelling / custom branding** — replace the Shadow Loom
  name and marks in the UI, MCP server identity, and exported
  artifacts with the licensee's own.
* **Single Sign-On (SSO)** — SAML 2.0 and OpenID Connect identity
  providers (Okta, Entra ID, Auth0, Google Workspace), SCIM
  provisioning, and enterprise role-based access control (RBAC).
* **Audit logs & compliance hooks** — tamper-evident audit trail
  exports, retention policies, and integrations for SOC 2, ISO 27001,
  HIPAA, and GDPR Article 30 reporting.
* **Private / air-gapped deployment** — offline installer, signed
  container images, and configuration for environments without
  outbound internet access.
* **Dedicated support & SLA** — guaranteed response and resolution
  times, named support contact, prioritised bug fixes, and security
  advisories ahead of public disclosure.
* **Custom compliance & indemnification** — IP indemnification,
  warranty terms, data-processing addenda (DPA), and bespoke
  compliance attestations negotiated as part of the agreement.
* **Long-term support (LTS) branches** — extended security and
  bug-fix support on selected minor releases beyond the upstream
  community window.

Feature availability and pricing depend on the agreed tier. The
open-source AGPLv3 build will continue to receive the core
research-grade functionality regardless of these add-ons.

---

## 7. Scope of the licensed work (derivative works & integrations)

AGPLv3 § 0 and § 5 define a "covered work" and a work "based on" the
Program in terms of copyright law. The exact line is fact-specific,
but for clarity the maintainer's good-faith interpretation of the
project boundary is as follows. **This section is interpretive
guidance from the maintainer; it does not modify the AGPLv3 and is
not a substitute for legal advice.**

**Considered part of "Shadow Loom" (the covered work):**

* All code under the `shadow_loom/`, `shadow_loom_mcp/`, and
  `shadow_loom_ui/` packages, plus any module that imports from them
  in the same Python process.
* Forks and modified versions of any of the above, even when
  re-packaged or renamed.
* Plugins, extensions, or subclasses that import Shadow Loom symbols
  and run in the same address space.
* Prompt templates, schemas, and configuration files distributed with
  Shadow Loom.

**Considered NOT a derivative / NOT a combined work** (in the
maintainer's good-faith interpretation, and subject to the
limitations below):

* **API consumers** that interact with Shadow Loom only through its
  documented HTTP, MCP, or CLI interfaces, exchanging data over a
  network or process boundary, without linking against Shadow Loom
  code in the same process.
* **Independent services** running in separate containers or
  processes that communicate with Shadow Loom only through the
  interfaces above, where the consumer does not embed, redistribute,
  or modify Shadow Loom itself.
* **Datasets, prompts, and outputs** generated *by* running Shadow
  Loom on user-supplied inputs. Shadow Loom does not assert copyright
  over the artifacts you produce with it; AGPLv3 § 2 already affirms
  this for outputs that do not themselves include covered code.
* **User-supplied novels, plot files, and graph databases** processed
  by Shadow Loom remain the property and licensing concern of their
  owners.

**Limitations on the above interpretation:**

1. Distributing a modified Shadow Loom (even behind an API) still
   triggers AGPLv3 § 13 — running the modified version on a public
   network requires offering its corresponding source to users.
2. Re-exporting Shadow Loom's internals through a thin API shim
   purely to evade copyleft is not a good-faith integration and is
   treated as a derivative work.
3. Bundling Shadow Loom into the same distribution package, container
   image, or installer as proprietary code may create a combined work
   under AGPLv3 § 5 even when the runtime communication is over a
   socket. When in doubt, isolate Shadow Loom in its own container or
   service and obtain a commercial licence for tightly-coupled
   embeddings.
4. This section is not a licence exception. It explains how the
   maintainer reads AGPLv3 in typical integration patterns; the
   AGPLv3 itself remains the binding text.

If you are unsure whether your integration crosses the line, **ask
before shipping**. The maintainer can usually confirm in writing that
a specific integration pattern does not require a commercial licence,
or offer one if it does.

---

## 8. Trademarks

"Shadow Loom" and the Shadow Loom logo (if any) are trademarks of
David Rae Wilmot. Neither the AGPLv3 nor the commercial license
grants any trademark rights beyond those required to identify the
unmodified upstream product. Forks and derivative works must be
clearly distinguishable by name and branding.

---

## 9. Disclaimer

This document is a summary of the commercial-license offering and is
**not itself a binding contract**. Any commercial license is granted
exclusively under a separately executed written agreement signed by
the copyright holder. In the absence of such an agreement, your use
of Shadow Loom is governed solely by the AGPLv3 in [`LICENSE`](LICENSE).

Nothing in this document is legal advice; if in doubt, consult your
own counsel.
