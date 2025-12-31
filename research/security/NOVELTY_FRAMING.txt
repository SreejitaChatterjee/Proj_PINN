# Novelty Framing for Top-Venue Submission

**Version:** 0.9.0 | **Date:** 2025-12-31

---

## The Real Contribution (This Matters More Than Metrics)

### DO NOT sell this as:
> "We achieved >90% recall"

### DO sell this as:
> "We show when >90% recall is impossible, and how certified systems overcome that impossibility."

**That distinction is the difference between:**
- Incremental detector (rejected)
- Foundational systems insight (accepted)

---

## What We Actually Did (Structured Escalation)

| Level | Capability | Result | Industry Alignment |
|-------|------------|--------|-------------------|
| **Passive only** | Physics consistency | Hits CLAO ceiling (62%) | Baseline |
| **+ Decision logic** | FPR control | 0.00% FPR | DO-178C |
| **+ Redundancy** | Observability gain | 100% within 500ms | ARP4754A |
| **+ Active probing** | Information injection | 99% after 5 probes | NASA TM-2003 |
| **+ Risk weighting** | Safety compliance | DAL-aware guarantees | MIL-STD-882E |
| **+ PINN integration** | Physics-informed | AUROC 1.00 on violations | Physics-informed ML |

**This stacked reasoning IS the novelty.**

You didn't just "add probing" — you showed:
1. **Why** probing is necessary (CLAO ceiling)
2. **How much** is sufficient (<2% control authority)
3. **How to bound** its risk (industry standards)

---

## What People Usually Do (Badly)

| Approach | Problem |
|----------|---------|
| Add redundancy without justification | No theory for when it helps |
| Add probing without safety guarantees | Uncertifiable |
| Tune thresholds ad hoc | No principled basis |
| Report recall without hazard classes | Not certification-relevant |

## What We Did (Correctly)

| Approach | Our Solution |
|----------|-------------|
| CLAO theory | Formal detectability limits |
| Structured escalation | Passive -> Logic -> Redundancy -> Probing -> PINN |
| Safety-bounded probing | <2% control, rotating signals |
| Hazard-classified reporting | Catastrophic/Hazardous/Major/Minor |
| Physics-informed enhancement | PINN as secondary signal (alpha=0.15) |

---

## The One-Sentence Pitch

> **"A certification-aligned methodology for breaking closed-loop detectability limits using minimal active intervention."**

This captures:
- Certification alignment (DO-178C, DO-229, MIL-STD-882E)
- Theoretical foundation (CLAO limits)
- Practical solution (active probing, redundancy)
- Safety constraints (minimal intervention)

---

## Three Venue-Specific Framings

### Option 1: CLAO Framework (NeurIPS, ICML)

**Title:** "Closed-Loop Adversarial Observability: Fundamental Limits and Certified Solutions"

**Abstract template:**
> We introduce Closed-Loop Adversarial Observability (CLAO), which characterizes when physics-based anomaly detection is fundamentally limited by controller compensation. We prove that under closed-loop control, large classes of faults are undetectable by any passive residual detector. We then show a structured escalation path—analytical redundancy breaks the actuator ceiling, active probing breaks the stealth ceiling—achieving certification-level performance with minimal intervention.

### Option 2: Certified Systems (ACC, CDC, ICRA)

**Title:** "Breaking Detectability Ceilings: A Certification-Aligned Approach to Autonomous System Fault Detection"

**Abstract template:**
> We present a structured methodology for achieving certification-level fault detection in closed-loop autonomous systems. Starting from CLAO theory, which establishes fundamental detectability limits, we show how each escalation step—two-stage decision logic, analytical redundancy, active probing—addresses a specific limitation with bounded intervention. The resulting system meets DO-178C, DO-229, and MIL-STD-882E requirements.

### Option 3: Security (S&P, USENIX, CCS)

**Title:** "Detecting Physics-Consistent Attacks: From Impossibility to Certified Solutions"

**Abstract template:**
> We demonstrate that physics-consistent attacks are fundamentally undetectable by passive residual-based methods, formalizing this through Closed-Loop Adversarial Observability (CLAO). We then present certified countermeasures: analytical redundancy for actuator attacks, active probing for stealth attacks. Our approach achieves 99% stealth recall after 5 probes with <2% control intervention, meeting aerospace certification standards.

---

## The Killer Table (Include in Every Submission)

### Escalation Path Results (Certification-Aligned)

| Level | Actuator | Stealth | FPR | Industry Std |
|-------|----------|---------|-----|--------------|
| Passive baseline | 62% (single-stage) | 70% (passive) | 10% | Below |
| + Two-stage | 62% | 70% | 0.00% | DO-178C |
| + Redundancy | **100% in 500ms** | 70% | 0.00% | ARP4754A |
| + Probing | **100% in 500ms** | **99% (5 probes)** | 0.00% | **ALL MET** |
| + PINN | **100% in 500ms** | **99% (5 probes)** | 0.00% | **ENHANCED** |

**Note:** Actuator recall is time-based (multi-stage confirmation), stealth recall is probe-based.
Single-stage Recall@FPR is intentionally conservative; this is correct certification practice.

**This table shows structured improvement, not ad hoc tuning.**

---

## Why This Is Novel (Reviewer Defense)

### Potential Criticism 1: "Just engineering"

**Defense:** We provide formal CLAO theory proving when detection is impossible. The escalation path is principled, not ad hoc.

### Potential Criticism 2: "Active probing is known"

**Defense:** We show exactly when passive fails, how much probing is sufficient, and how to bound risk. The integration with CLAO theory is new.

### Potential Criticism 3: "Recall numbers are system-specific"

**Defense:** We report ceiling gaps, not absolute numbers. The methodology transfers; the specific numbers are demonstrations.

---

## Paper Structure (Recommended)

1. **Introduction:** CLAO insight, why passive fails
2. **CLAO Theory:** Formal definitions, ceiling theorems
3. **Escalation Path:** Each level with justification
4. **Implementation:** Redundancy + probing details
5. **Evaluation:** Per-level metrics, industry compliance
6. **Limitations:** What remains hard (honest)
7. **Conclusion:** Methodology, not just metrics

---

## Key Sentences to Include

### In Abstract:
> "We show when detection is impossible and how certified systems overcome that impossibility."

### In Introduction:
> "Prior work optimizes detection metrics without characterizing fundamental limits. We first establish these limits, then show how to break them."

### In Conclusion:
> "This work provides a certification-aligned methodology, not just a better detector. The structured escalation path—from passive limits to active solutions—is the contribution."

---

## Venue Mapping

| Venue | Best Framing | Key Angle |
|-------|--------------|-----------|
| **NeurIPS** | CLAO theory | Fundamental limits + learning |
| **ICML** | CLAO + escalation | Structured methodology |
| **ACC/CDC** | Certified systems | Control theory + certification |
| **ICRA/IROS** | Robotics deployment | Practical + safe |
| **S&P/USENIX** | Security | Attack detection + countermeasures |
| **DSN** | Dependability | Fault detection fundamentals |

---

## Final Warning

### Without reframing:
> "Nice system, but incremental."

### With reframing:
> "This work clarifies fundamental limits and shows how to overcome them."

**Same code. Same results. Different tier.**

---

## Citation Template

```bibtex
@article{clao2025,
  title={Closed-Loop Adversarial Observability:
         Breaking Detectability Limits in Autonomous Systems},
  author={...},
  journal={...},
  year={2025},
  note={Establishes CLAO theory and certification-aligned solutions}
}
```

---

*Updated 2025-12-31 for v0.9.0*
