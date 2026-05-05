# Shadow Loom — Commercial License (AGPLv3 Exception)

**Copyright © 2026 David Rae Wilmot. All rights reserved.**

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

You **do not** need the commercial license if **either** of the
following applies:

* Your use of Shadow Loom is **internal to a single legal entity** —
  the modified or unmodified version is made available only to
  employees, contractors, or members of that entity acting in that
  capacity, and is not exposed to users outside it (whether free,
  paid, public, or behind authentication). Sharing a deployment
  across separate legal entities (parent / subsidiary, contractor /
  client, multi-tenant SaaS) is **not** internal use for these
  purposes; **or**
* You are willing to license your full combined work under AGPLv3 and
  to make the corresponding source available to your users in
  compliance with AGPLv3 § 13.

Note that even strictly-internal use of an *unmodified* upstream
build does not by itself trigger § 13; the obligation arises when a
*modified* covered work is made available to users over a network.
The internal-use bullet above is a conservative summary, not a
licence amendment.

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

**Arm's-length interactions the maintainer does not, on their own,
treat as creating a "modified version" or "work based on" Shadow
Loom under AGPLv3 §§ 0 and 5** (interpretive guidance only — the
courts, not the maintainer, ultimately decide derivative-work
status, and the obligations of AGPLv3 § 13 are *not* affected by
anything in this section):

* **Out-of-process API consumers.** A program that interacts with
  Shadow Loom *solely* through its documented HTTP, MCP, or
  command-line interfaces — over a network socket, pipe, or
  separate-process boundary — and that does **not** import, link,
  embed, vendor, or otherwise load any Shadow Loom source or
  compiled artifact into its own address space, is communicating at
  arm's length. Such interaction alone is not, in the maintainer's
  view, sufficient to make the consumer a covered work.
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

1. **AGPLv3 § 13 is unaffected.** Anyone who modifies Shadow Loom
   and makes the modified version available to users over a network
   — including making it reachable through the very HTTP, MCP, or
   CLI interfaces named above — must offer those users the
   Corresponding Source of the modified Shadow Loom. The arm's-length
   reading concerns the *consumer's* code, not the operator's
   obligations as a Shadow Loom modifier.
2. **No additional permission is granted.** This section is
   interpretive guidance, not an additional permission under
   AGPLv3 § 7, and is not a license exception. The AGPLv3 itself
   remains the binding text and the maintainer cannot waive
   downstream recipients' rights under it.
3. **In-process use is a combined work.** Importing any Shadow Loom
   Python package (`shadow_loom`, `shadow_loom_mcp`, `shadow_loom_ui`,
   or any submodule), subclassing its types, vendoring its source,
   or statically/dynamically linking against it places the resulting
   program inside the covered work regardless of whether it also
   speaks to a separate Shadow Loom instance over a socket.
4. **Anti-evasion.** Re-exporting Shadow Loom's internals through a
   thin API shim, RPC wrapper, or fork-and-rename whose purpose is to
   keep proprietary code in the same address space while pretending
   the boundary is at the network is not a good-faith arm's-length
   integration and is treated as a derivative work.
5. **Bundling.** Shipping Shadow Loom in the same distribution
   package, container image, installer, or virtual environment as
   proprietary code may create a combined work under AGPLv3 § 5 even
   when runtime communication is over a socket. When in doubt,
   isolate Shadow Loom in its own container or service and obtain a
   commercial licence for tightly-coupled embeddings.
6. **Specific patterns are fact-specific.** This section does not
   opine on whether any *particular* integration is arm's length;
   that determination depends on facts the maintainer cannot verify
   in advance.

If you are unsure whether your integration crosses the line, **ask
before shipping**. The maintainer can usually confirm in writing that
a specific integration pattern does not require a commercial licence,
or offer one if it does.

### 7a. Additional permissions under AGPLv3 § 7

For the avoidance of doubt, **no additional permissions are granted
under AGPLv3 § 7** in respect of the AGPLv3-licensed distribution of
Shadow Loom. The AGPLv3 text in [`LICENSE`](LICENSE) is the complete
statement of the permissions granted to recipients of the
open-source build. In particular:

* No linking exception, classpath exception, or system-library
  exception is granted. Any program that links Shadow Loom into the
  same address space (importing `shadow_loom`, `shadow_loom_mcp`,
  `shadow_loom_ui`, or any submodule; subclassing their types;
  vendoring or statically/dynamically linking their compiled output)
  forms a combined work under AGPLv3 § 5 and must itself be licensed
  under AGPLv3 unless covered by a separately executed commercial
  agreement under § 4 above.
* No § 13 waiver is granted. Operators of modified versions of
  Shadow Loom must continue to offer Corresponding Source to remote
  users as required by AGPLv3 § 13.
* The interpretive guidance in § 7 of this document (arm's-length
  network use) is **not** an additional permission under AGPLv3 § 7.
  It is the maintainer's good-faith reading of how AGPLv3 §§ 0 and 5
  already apply to typical integration patterns; it does not modify
  the licence and confers no rights that the licence itself does not
  already confer.
* Recipients of the AGPLv3 build may not add their own further
  restrictions to the licence beyond those AGPLv3 § 7 itself
  permits, and may not represent that any such additional
  permissions originate from the maintainer.

If at some future date the maintainer wishes to grant an additional
permission (for example, an explicit linking exception for a named
compatible licence), it will be added in this section as a numbered
sub-clause and dated. Until then, treat the permissions in AGPLv3
as exhaustive.

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

---

## 10. Governing law and jurisdiction

This document, the commercial-licence offering it describes, and any
dispute, claim or matter (whether contractual or non-contractual)
arising out of or in connection with it — including pre-contractual
negotiations, the interpretation of § 7 (project boundary), and any
claim relating to trademark use under § 8 — are **governed by the
laws of England and Wales**, without regard to conflict-of-laws
principles.

The **courts of England and Wales have exclusive jurisdiction** to
settle any such dispute, claim or matter. By approaching the
copyright holder for a commercial licence, by signing an executed
commercial agreement, or by relying on the interpretive guidance in
§ 7 of this document, you submit to the personal jurisdiction of
those courts.

An executed commercial agreement may restate, narrow, or supplement
this clause; where the executed agreement and this section conflict,
the executed agreement prevails as between its signatories.

This matches the forum for the Contributor License Agreement
([`CLA.md` § 8](CLA.md#8-governing-law)) and the Content Policy
([`CONTENT-POLICY.md` § 10](CONTENT-POLICY.md#10-governing-law-and-jurisdiction)),
so every Shadow Loom legal text the maintainer controls is enforced
in a single forum.
