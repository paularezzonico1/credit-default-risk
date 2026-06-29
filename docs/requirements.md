# Business and Functional Requirements

**Document title:** Retail Credit Default Scoring System — Business and Functional Requirements
**Document ID:** CRC-BRD-001
**Version:** 1.0
**Status:** For review and sign-off
**Date:** 2026-06-17
**Commissioned by:** Credit Risk Committee
**System of record:** `credit-default-risk` (explainable retail-loan default scoring platform)

---

## 1. Purpose and scope

### 1.1 Purpose
This document specifies the business, functional, and non-functional requirements for the
Retail Credit Default Scoring System (the "System"). It is commissioned by the Credit Risk
Committee to govern the development, validation, deployment, and ongoing operation of an
automated, explainable model that estimates the probability of default (PD) for retail loan
applicants and supports the resulting approve, review, and decline decisions.

### 1.2 Scope
**In scope:** PD estimation for retail (consumer) lending, per-decision explanation, decision
recommendation against a documented threshold, model governance metadata, portfolio stress
testing under macroeconomic scenarios, and the service interfaces (scoring API and operator
dashboard) used to consume these outputs.

**Out of scope:** Final credit adjudication authority (the System recommends, a human or
downstream policy engine adjudicates), pricing and limit assignment, collections, and any
non-retail (corporate, sovereign, or counterparty) exposure.

### 1.3 Definitions
| Term | Meaning |
|---|---|
| PD | Probability of default, expressed in `[0, 1]`. |
| Decision | One of APPROVE, REVIEW, or DECLINE, derived from PD and the decision threshold. |
| Threshold | The PD cut-off applied to map a score to a decision (documented default: 0.20). |
| SHAP | Shapley-additive feature attribution used to explain an individual score. |
| Artifact | A serialized, versioned model object and its governance metadata. |
| MRM | Model Risk Management function. |
| Audit key | The unique `request_id` issued for every scored decision. |

---

## 2. Business problem statement

The bank's retail lending decisions are currently supported by tooling that the Credit Risk
Committee judges to be insufficiently accurate, inconsistently explainable, and difficult to
evidence to regulators. This creates three concrete business problems.

**2.1 Capital reserve accuracy.** Expected-loss and regulatory-capital provisioning depend
directly on the quality of PD estimates. Systematic over-estimation of default risk traps
capital that could be deployed and suppresses approvals of creditworthy borrowers; systematic
under-estimation leaves the bank under-reserved against realized losses. The Committee requires
a PD estimate whose discriminatory power is measured, validated out-of-sample, and monitored,
so that reserve calculations rest on a defensible statistical foundation.

**2.2 Regulatory compliance.** Supervisory expectations (including SR 11-7 model risk
guidance, ECOA/Regulation B adverse-action requirements, and CCAR-style stress expectations)
require that automated credit decisions be explainable, documented, reproducible, and capable
of producing a specific reason for any adverse action. A decline that cannot be explained to
the applicant and to the regulator is a compliance liability regardless of its statistical
accuracy.

**2.3 Decision consistency and auditability.** Manual and opaque decisioning produces variance
between officers and leaves gaps in the audit trail. The Committee requires that every decision
be reproducible from a recorded input, a versioned model, and a recorded explanation.

**Business objective:** deploy a governed scoring System that improves PD accuracy (measured by
out-of-sample ROC-AUC and calibration), produces a regulator-ready explanation for every
adverse decision, and stress-tests the portfolio under adverse macroeconomic scenarios, while
keeping the human adjudicator in control of final credit authority.

---

## 3. Stakeholders

| Stakeholder | Interest in the System | Primary concerns |
|---|---|---|
| **Credit Risk Committee** | Owns the business mandate; approves the risk appetite, the decision threshold, and go-live. | PD accuracy, approval/decline economics, capital impact, residual risk. |
| **Compliance** | Ensures the System meets consumer-protection and fair-lending obligations. | Adverse-action reason codes, fair-lending exposure, applicant-facing explanations, record retention. |
| **Model Risk Management (MRM)** | Independent validation and ongoing monitoring of the model. | Conceptual soundness, out-of-sample performance, reproducibility, documentation, model versioning. |
| Credit Officers / Adjudicators | Day-to-day consumers of scores and explanations. | Clarity of the recommendation, usable reason factors, response latency. |
| Technology / Platform Operations | Builds, deploys, and runs the service. | Availability, latency, configurability, observability. |
| Internal Audit | Periodic assurance over the control environment. | Traceability from requirement to control to evidence. |

---

## 4. Functional requirements

Each requirement is uniquely numbered for traceability and carries explicit acceptance
criteria. "Shall" denotes a mandatory requirement.

### FR-1: Probability-of-default scoring
**Requirement.** The System shall accept a defined set of borrower features and output a
probability of default as a number in the closed interval `[0, 1]`.
**Rationale.** PD is the primary input to provisioning and decisioning (see 2.1).
**Acceptance criteria.**
- AC-1.1: Given a valid borrower payload, the System returns a `probability_of_default` field within `[0, 1]`.
- AC-1.2: The same input scored against the same model version returns an identical PD (deterministic, reproducible).
- AC-1.3: The set of accepted features matches the documented model feature list returned by the governance endpoint.

### FR-2: Decision recommendation against a governed threshold
**Requirement.** The System shall map each PD to exactly one decision of APPROVE, REVIEW, or
DECLINE using the threshold configured and approved by the Credit Risk Committee.
**Rationale.** Translates a continuous score into an actionable, consistent recommendation.
**Acceptance criteria.**
- AC-2.1: Every scored request returns a `decision` value in {APPROVE, REVIEW, DECLINE}.
- AC-2.2: The applied `threshold` is returned alongside every decision.
- AC-2.3: The threshold is configurable at deployment without code change, and defaults to the documented value (0.20) when no override is set.
- AC-2.4: The System never issues a final adjudication; the decision is a recommendation consumed by a human or downstream policy step.

### FR-3: SHAP-based explanation for every decline decision
**Requirement.** The System shall output a SHAP-based explanation identifying the leading
feature contributions for every DECLINE decision, and shall make the same explanation available
for APPROVE and REVIEW decisions.
**Rationale.** Adverse-action and supervisory rules require a specific, defensible reason for
every decline (see 2.2).
**Acceptance criteria.**
- AC-3.1: Every DECLINE response includes a non-empty `top_risk_factors` list.
- AC-3.2: Each risk factor includes the feature name, a human-readable label, its SHAP contribution in log-odds units, and a direction ("toward default" or "away from default").
- AC-3.3: The signed sum of attributions is consistent with the model's reasoning, so the explanation reflects the actual score rather than a generic narrative.
- AC-3.4: At least the top three contributing factors are returned for each decision.

### FR-4: Per-decision audit key
**Requirement.** The System shall assign a unique, persistent identifier (`request_id`) to
every scored decision.
**Rationale.** Reproducibility and auditability (see 2.3) require that each decision be
individually addressable.
**Acceptance criteria.**
- AC-4.1: Every scoring response contains a unique `request_id`.
- AC-4.2: The `request_id`, input, PD, decision, threshold, explanation, and model version are recorded together as one auditable decision record.

### FR-5: Model governance metadata
**Requirement.** The System shall expose, on demand, the governance metadata of the deployed
model: version identifier, training date, documented hold-out ROC-AUC, active threshold, and
the feature list.
**Rationale.** MRM and Audit must be able to confirm which model produced a given decision.
**Acceptance criteria.**
- AC-5.1: A governance endpoint returns `model_version`, `training_date`, `holdout_auc`, `threshold`, `n_features`, and `features`.
- AC-5.2: The `model_version` returned in metadata matches the `model_version` stamped on every scoring response.

### FR-6: Input validation and safe rejection
**Requirement.** The System shall validate every input against documented field bounds and
reject malformed input (out-of-range, non-finite, or missing values) with a structured,
non-silent error.
**Rationale.** Invalid input must never produce a silently wrong score that feeds provisioning.
**Acceptance criteria.**
- AC-6.1: Out-of-range, missing, or non-finite (NaN/inf) values are rejected with a client-error response identifying the offending field.
- AC-6.2: A rejected request produces no decision record and no PD.
- AC-6.3: Validation bounds are documented and match the published input schema.

### FR-7: Liveness and readiness
**Requirement.** The System shall expose a health check that reports healthy only when the
model artifacts are loaded and the System is able to score.
**Rationale.** Operators and downstream consumers must not route decisions to an unready service.
**Acceptance criteria.**
- AC-7.1: The health endpoint reports "healthy" only when artifacts are loaded, otherwise "unhealthy".
- AC-7.2: When the model is not loaded, scoring and governance requests fail with a clear unavailable response rather than an incorrect score.

### FR-8: Macroeconomic stress testing
**Requirement.** The System shall support re-scoring of the portfolio under at least three
defined macroeconomic scenarios (baseline plus adverse variants) and report the resulting shift
in portfolio risk.
**Rationale.** CCAR-style stress expectations and capital adequacy review (see 2.1, 2.2).
**Acceptance criteria.**
- AC-8.1: A baseline and at least two adverse scenarios can be applied to a borrower population.
- AC-8.2: The output reports the change in aggregate predicted default rate per scenario relative to baseline.
- AC-8.3: Scenario definitions and results are reproducible and documented.

### FR-9: Decoupled service interface
**Requirement.** The System shall serve scoring over a defined API so that the presentation
layer (operator dashboard) and the model-serving layer are independently deployable.
**Rationale.** Separation supports independent change control, validation, and scaling.
**Acceptance criteria.**
- AC-9.1: The operator dashboard consumes scores exclusively through the scoring API.
- AC-9.2: When the API is unreachable, the dashboard surfaces a clear error and does not present a stale or fabricated score.

### FR-10: Configurability without code change
**Requirement.** The System shall read all operational parameters (artifact location, decision
threshold, service binding, database connection) from configuration, so the same build runs in
any environment by configuration alone.
**Rationale.** Reproducible promotion across dev, validation, and production with controlled change.
**Acceptance criteria.**
- AC-10.1: Threshold, artifact path, and service binding are set by environment configuration with documented defaults.
- AC-10.2: No environment-specific value is hardcoded in the application.

---

## 5. Non-functional requirements

### NFR-1: Explainability for regulators
The System shall produce, for any decision, an explanation expressed in per-feature
contributions suitable for inclusion in an adverse-action notice and for review by Compliance
and supervisors. Explanations shall be reproducible for the life of the decision record.
**Acceptance:** Compliance can map any sampled DECLINE to a specific, human-readable set of
contributing factors; MRM can reproduce the explanation from the recorded input and model version.

### NFR-2: Latency
A single synchronous scoring request, including its explanation, shall return within **500 ms at
the 95th percentile** under expected production load, measured at the API boundary.
**Acceptance:** A load test at expected concurrency shows p95 end-to-end scoring latency at or
below 500 ms and p99 at or below 1000 ms.

### NFR-3: Availability
The scoring service shall target **99.5% monthly availability** during defined business hours,
with health-gated routing so that an unready instance does not receive traffic.
**Acceptance:** Operational monitoring evidences availability at or above target; unhealthy
instances are demonstrably excluded from serving.

### NFR-4: Data retention and record-keeping
Decision records (audit key, input, PD, decision, threshold, explanation, and model version)
shall be retained for a **minimum of seven (7) years** to satisfy lending record-retention
obligations, and shall be retrievable by audit key.
**Acceptance:** A sampled historical `request_id` can be retrieved with its full decision
record within the retention window.

### NFR-5: Reproducibility and versioning
Every deployed model shall carry a unique version and a training date, and any decision shall be
reproducible from its recorded input and model version.
**Acceptance:** MRM re-scores a sampled historical decision against the stamped model version
and obtains an identical PD, decision, and explanation.

### NFR-6: Security and access control
Borrower data in transit shall be protected, access to scoring and decision records shall be
authenticated and authorized, and no borrower PII shall appear in application logs.
**Acceptance:** A security review confirms transport protection, role-based access to records,
and absence of PII in logs.

### NFR-7: Fairness and non-discrimination
The model shall be assessed for disparate impact across protected classes prior to deployment
and on a recurring basis, with results documented in the model card.
**Acceptance:** A current fair-lending assessment exists in the governance documentation and is
reviewed by Compliance before each model promotion.

### NFR-8: Observability
The System shall log each request outcome (including failures) with its audit key and shall
surface unexpected errors as structured, non-silent responses.
**Acceptance:** Induced failures produce structured error responses and corresponding log
entries keyed by `request_id`.

---

## 6. Requirements traceability matrix

| Requirement | Business driver | Primary owner | Verification |
|---|---|---|---|
| FR-1 | 2.1 Capital accuracy | MRM | Test + validation report |
| FR-2 | 2.1 / 2.3 | Credit Risk Committee | Test + policy review |
| FR-3 | 2.2 Compliance | Compliance | Sampled decline review |
| FR-4 | 2.3 Auditability | Internal Audit | Record inspection |
| FR-5 | 2.2 / 2.3 | MRM | Endpoint test |
| FR-6 | 2.1 / 2.3 | MRM | Negative test suite |
| FR-7 | NFR-3 | Tech Ops | Health-check test |
| FR-8 | 2.1 / 2.2 | Credit Risk Committee | Stress-test report |
| FR-9 | 2.3 | Tech Ops | Integration test |
| FR-10 | NFR-5 | Tech Ops | Config review |
| NFR-1 | 2.2 | Compliance | Reproduction test |
| NFR-2 | Service quality | Tech Ops | Load test |
| NFR-3 | Service quality | Tech Ops | Availability report |
| NFR-4 | 2.2 | Compliance | Retention audit |
| NFR-5 | 2.3 | MRM | Re-score test |
| NFR-6 | Security | Tech Ops | Security review |
| NFR-7 | 2.2 | Compliance | Fair-lending assessment |
| NFR-8 | 2.3 | Tech Ops | Failure-injection test |

---

## 7. Assumptions and constraints

- **A-1:** The System recommends; final credit authority rests with a human adjudicator or an
  approved downstream policy engine.
- **A-2:** Training and validation data are governed under existing data-management controls;
  data quality at source is out of scope for this document.
- **A-3:** The documented decision threshold (0.20) is the Committee-approved default and may be
  re-set only through the Committee's change-control process.
- **C-1:** The System operates on retail exposures only.
- **C-2:** Any model change is subject to MRM validation and Committee approval before promotion.

---

## 8. RACI matrix

R = Responsible, A = Accountable, C = Consulted, I = Informed.

| Activity | Credit Risk Committee | Compliance | MRM | Tech Ops | Internal Audit |
|---|---|---|---|---|---|
| Approve business requirements | A | C | C | I | I |
| Define decision threshold and risk appetite | A | C | R | I | I |
| Validate model (conceptual soundness, performance) | I | C | A/R | C | I |
| Approve adverse-action / explanation content | C | A/R | C | I | I |
| Build and operate the scoring service | I | I | C | A/R | I |
| Approve go-live | A | C | C | R | I |
| Stress testing and capital impact review | A | C | R | C | I |
| Periodic monitoring and fair-lending review | I | A/R | R | C | I |
| Records retention and retrieval | I | A | C | R | C |
| Independent assurance over controls | I | C | C | C | A/R |

---

## 9. Sign-off

The signatories below approve this Business and Functional Requirements document, version 1.0,
as the agreed basis for development, validation, and deployment of the Retail Credit Default
Scoring System. Approval is contingent on the acceptance criteria in sections 4 and 5 being
evidenced prior to go-live.

| Role | Name | Signature | Date |
|---|---|---|---|
| Chair, Credit Risk Committee | | | |
| Head of Compliance | | | |
| Head of Model Risk Management | | | |
| Head of Technology / Platform Operations | | | |
| Head of Internal Audit (acknowledgement) | | | |

---

*Document control: changes to this document are managed under the Credit Risk Committee's
change-control process. Superseded versions are retained per NFR-4.*
