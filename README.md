# ASTE 404 Mini-Project Report
## Spacecraft Reliability Monte Carlo Tool

**Author:** Lilly Clements  
**Date:** December 15, 2025  
**Revised:** October 10, 2026 (convergence study redone, common-cause extension added, numbers corrected)  
**GitHub Repository:** https://github.com/lillyclements/mini-project

---

### Summary

- **Problem:** Estimate the probability that a spacecraft with several independent subsystems completes its mission, and decide where redundancy is worth adding.
- **Method:** Monte Carlo simulation in Python (NumPy), validated against closed-form reliability theory.
- **Key results:** Baseline mission reliability 92.24%. One backup on the highest-risk subsystem raises it to 94.55% (+2.31 percentage points). Monte Carlo error scales as N^-0.50 (fitted slope -0.498 ± 0.010).
- **Extension:** A beta-factor common-cause model shows that correlated failures erode the value of redundancy (full duplication: 99.86% at β = 0 vs. 98.32% at β = 0.2).

---

## 1. Project Overview

### 1.1 Motivation and Objectives

This project addresses a fundamental question in spacecraft systems engineering: **Given a spacecraft with multiple independent subsystems, each with a known failure probability, what is the probability that the mission succeeds?**

For series reliability systems where all components must function correctly, even small individual failure rates can compound to create significant mission risk. This tool was developed to:

1. Estimate mission reliability using Monte Carlo simulation
2. Verify results against analytical solutions
3. Model the impact of redundant backup systems
4. Perform trade studies to optimize redundancy placement
5. Demonstrate proper numerical methods and software engineering practices

### 1.2 Why This Matters

In spacecraft design, engineers must make critical decisions about system architecture, component selection, and redundancy strategies. This tool provides:

- **Quick reliability estimates** for proposed configurations
- **Trade study capability** to evaluate cost vs. reliability
- **Verification framework** to ensure numerical accuracy
- **Extensible codebase** for future mission-specific analysis

### 1.3 Project Scope

The tool implements:
- Monte Carlo simulation for series reliability estimation
- Analytical verification for independent subsystems
- Redundancy modeling (active parallel redundancy)
- Convergence analysis demonstrating O(1/√N) behavior
- Statistical variance analysis
- Edge case verification tests
- Command-line interface for flexible usage
- Common-cause failure modeling (beta-factor) with analytical verification
- Vectorized Monte Carlo and a repeated-run convergence study (`reliability_extensions.py`)

---

## 2. Numerical Method

### 2.1 Monte Carlo Simulation

Monte Carlo integration estimates the probability of mission success by random sampling:

**Algorithm:**
```
For i = 1 to N (number of samples):
    mission_success = True
    For each subsystem j:
        Draw random number r ~ Uniform(0,1)
        If r < p_j (failure probability):
            mission_success = False
            Break
    If mission_success:
        count = count + 1
        
Reliability = count / N
```

**Why Monte Carlo is appropriate:**
- Simple to implement and understand
- Naturally extends to complex scenarios (redundancy, non-independent failures)
- Provides probabilistic confidence intervals
- No assumptions about distribution shapes required

### 2.2 Convergence and Accuracy

**Theoretical behavior:** Monte Carlo error decreases as O(1/√N), where N is the number of samples.

For a binomial proportion with true value p:
- Expected error: σ ≈ √(p(1-p)/N)
- To halve the error, need 4× more samples
- 95% confidence interval: p ± 1.96σ

**Practical implications** (95% confidence half-width, 1.96σ, for this project's baseline R ≈ 0.92):
- N = 10,000 gives about ±0.5% (±0.0052)
- N = 100,000 gives about ±0.17% (±0.0017)
- N = 1,000,000 gives about ±0.05% (±0.0005)

This convergence rate is **slow** compared to deterministic methods, but the simplicity and flexibility of MC often outweigh this disadvantage for early-stage analysis.

### 2.3 Stability Considerations

Monte Carlo has **no stability constraints** in the traditional sense (not solving differential equations). However, considerations include:

1. **Statistical variance:** Inherent uncertainty in estimates
2. **Rare event bias:** For very high reliability (>99.9%), need many samples to accurately estimate low failure rates
3. **Numerical precision:** For extremely small probabilities, floating-point precision can become an issue

**Limitations:**
- Slow convergence (√N) compared to analytical methods
- Requires many samples for high accuracy
- Statistical uncertainty always present
- Not suitable for real-time applications

### 2.4 Analytical Verification Method

For independent subsystems, the exact reliability is:

R = P(all work) = ∏(1 - p_i) for i = 1 to N

For redundant subsystems with redundancy level k:

R_subsystem = 1 - p^k

This provides closed-form solutions to verify Monte Carlo results. Note that adding one backup to subsystem i raises system reliability by exactly R·p_i (because (1 − p_i²)/(1 − p_i) = 1 + p_i), so the subsystem with the largest failure probability always benefits most.

### 2.5 Common-Cause Failures (Beta-Factor Model)

The independent model assumes backups fail independently of the primary, which is optimistic: a single event (radiation, a power-bus fault, a shared software bug) can disable every unit in a redundant group. The beta-factor model splits each unit's failure probability p into:

- an **independent part**, (1 − β)·p, where units fail separately, and
- a **common-cause part**, β·p, a single event that fails all k units at once.

A redundant group (k > 1) fails if the common-cause event occurs **or** all k units fail independently:

P_group_fail = 1 − (1 − β·p) · (1 − ((1 − β)·p)^k)

For k = 1 the group failure probability is simply p. Setting β = 0 recovers the independent model, and β = 1 makes a backup worthless. The closed form above is used to verify the Monte Carlo implementation (Section 5.6); the simulation is what extends to models with no closed form.

---

## 3. Implementation Notes

### 3.1 Code Structure

```
mini-project/
├── reliability.py              # Main Python module (750+ lines)
│   ├── monte_carlo_reliability()
│   ├── analytical_reliability()
│   ├── convergence_study()
│   ├── monte_carlo_reliability_with_redundancy()
│   ├── redundancy_trade_study()
│   ├── monte_carlo_variance_analysis()
│   ├── edge_case_tests()
│   └── main() [CLI]
├── reliability_extensions.py   # Vectorized MC, repeated-run convergence, common-cause model
│   ├── mc_reliability() [vectorized]
│   ├── convergence_study() [RMS error, fitted slope]
│   ├── common_cause_study()
│   └── verify_common_cause()
├── requirements.txt            # numpy, matplotlib
├── README.md                   # Documentation
├── convergence_rms.png         # Generated plot (repeated-run convergence)
├── common_cause_study.png      # Generated plot (beta-factor study)
├── convergence_study.png       # Generated plot (original single-run version)
├── redundancy_trade_study.png  # Generated plot
└── variance_analysis.png       # Generated plot
```

### 3.2 Key Design Decisions

**1. Separate functions for MC and analytical solutions**
- Enables direct comparison and verification
- Clean separation 

**2. NumPy for random number generation**
- Industry-standard library
- Well-tested pseudorandom number generator implementation
- Core `reliability.py` uses explicit loops for clarity; `reliability_extensions.py` uses vectorized NumPy (much faster)

**3. Matplotlib for visualization**
- Standard plotting library in scientific Python
- Saves to PNG for report inclusion

**4. Command-line interface**
- Flexible usage without code modification
- Parameterized inputs (failure rates, sample sizes, redundancy)
- Multiple analysis modes (basic, convergence, redundancy, all)

**5. Modular function design**
- Each function has single responsibility
- Reusable components

### 3.3 How to Run

**Basic usage:**
```bash
python reliability.py
```
Runs all analyses with default 5-subsystem configuration.

**CLI examples:**
```bash
# Custom failure rates
python reliability.py --run basic --failures 0.01 0.02 0.015

# With redundancy
python reliability.py --run basic --failures 0.01 0.02 --redundancy 1 2

# Specific analysis
python reliability.py --run convergence
```

**Extensions (repeated-run convergence study and common-cause model):**
```bash
python reliability_extensions.py --run convergence   # saves convergence_rms.png
python reliability_extensions.py --run commoncause   # saves common_cause_study.png
python reliability_extensions.py --run all
```

**Dependencies:**
```bash
pip install -r requirements.txt
```

---

## 4. Progress Log

**Initial Setup:**
- Created GitHub repository: `mini-project`
- Set up Python environment with numpy and matplotlib
- Initialized git with .gitignore
- Created basic project structure

**Core Implementation:**
- Implemented `monte_carlo_reliability()` function
- Implemented `analytical_reliability()` for verification
- First successful run: MC estimate matched analytical to 3 decimal places
- **Commit:** "Initial Monte Carlo implementation"
- **Commit:** "Add core Monte Carlo functions and verification"

**Convergence Study:**
- Implemented `convergence_study()` to test N = [100, 500, 1000, ..., 500,000]
- Created dual-panel plot: estimate convergence + error scaling
- First version used one MC run per N. Each run's error is itself random: the N=100 run happened to start with a large error (~0.06), which inflated the apparent improvement to ~325× versus the theoretical √5000 ≈ 70×
- Reworked into a repeated-run study (200 runs per N, RMS error) in `reliability_extensions.py`: fitted log-log slope = -0.498 ± 0.010 (theory -0.5)
- **Commit:** "Add convergence study with O(1/sqrt(N)) verification"

**Redundancy Feature:**
- Implemented `monte_carlo_reliability_with_redundancy()`
- Added analytical verification for redundant systems
- Key insight: subsystem with highest failure rate (p=0.025) benefits most from backup
- Adding redundancy to that subsystem: 92.2% → 94.5% reliability (+2.3%)
- **Commit:** "Add redundancy modeling and trade study analysis"

**Trade Study Analysis:**
- Implemented `redundancy_trade_study()` to evaluate all subsystems
- Generated bar chart + scatter plot visualization
- Confirmed higher failure rate → higher value from redundancy
- Trade study directly applicable to engineering decision-making

**CLI Development:**
- Added argparse-based command-line interface
- Support for custom failure rates, sample sizes, redundancy configs
- Multiple run modes: basic, convergence, redundancy, all
- **Commit:** "Add CLI interface and comprehensive README"

**Variance Analysis:**
- Ran 50 independent MC trials with N=10,000 each
- Observed std: 0.00269, Theoretical std: 0.00267 (ratio ≈ 1.01; with 50 runs the std is only known to about ±10%) ✓
- Histogram is consistent with the theoretical normal distribution
- 48 of 50 estimates fall inside the 95% confidence band (96%; 95% expected)
- **Key learning:** MC uncertainty is predictable and matches theory

**Edge Case Verification:**
- Test 1: Perfect system (all p=0) → R = 1.0 ✓
- Test 2: Failed system (all p=1) → R = 0.0 ✓
- Test 3: Single subsystem → matches analytical ✓
- Test 4: High reliability system → numerically stable ✓
- Test 5: Redundancy calculation → matches expected value ✓
- **All tests passed**
- **Commit:** "Add variance analysis and edge case verification tests"

**Documentation:**
- Updated README with comprehensive usage examples
- Added method explanations, limitations, future work
- Created clear file structure documentation
- **Commit:** "Add comprehensive docstring and final code polish"

**Review and Extensions (October 2026):**
- Corrected the analytical baseline from 0.921951 to 0.922438 (the product of the (1 − p_i) terms; every generated plot already showed 0.922438)
- Made confidence-interval and accuracy figures consistent (σ = √(R(1−R)/N))
- Replaced "cold spares" with "active parallel redundancy" (the model has no switching logic or standby failures)
- Added `reliability_extensions.py`: vectorized MC, repeated-run convergence study, beta-factor common-cause model with analytical verification
- **Commit:** "Add RMS convergence study and common-cause model; correct README values"

### Challenges Encountered

1. **Git setup confusion:** Initially tried to use existing homework repo, created nested git repositories. Solution: Started fresh in new directory.

2. **Indentation error in CLI:** Mixed indentation levels in the `if args.run in ['all']:` block. Fixed by ensuring consistent 4-space indentation throughout.

3. **Understanding convergence rate:** My first convergence plot used one MC run per N and showed a ~325× error improvement from N=100 to N=500,000, versus ~70× expected from √N scaling. The cause was that a single run's error is itself random (the N=100 run started with a large error). Averaging 200 runs per N and fitting the log-log slope fixed it: -0.498 ± 0.010.

### What Worked Well

- **Incremental development:** Building feature-by-feature with commits after each working piece
- **Verification-first approach:** Implementing analytical solutions immediately enabled confidence in MC results
- **Visualization:** Plots made convergence behavior and trade-off insights immediately clear
- **Modular design:** Clean functions made adding features straightforward

---

## 5. Verification and Validation Evidence

### 5.1 Analytical Comparison

**Test system:** 5 subsystems with failure rates [0.01, 0.02, 0.015, 0.01, 0.025]

**Results:**
- Analytical solution: R = ∏(1 − pᵢ) = **0.922438** (92.24%)
- Expected statistical uncertainty of a single MC run at N = 100,000: σ = √(R(1−R)/N) = 0.00085 (95% CI ±0.0017)
- Repeated-run check (200 independent MC runs at N = 100,000, see 5.2): RMS error = 0.000932 vs. theoretical σ = 0.000846 (ratio 1.10)

**Interpretation:** MC estimates scatter around the analytical value with the spread theory predicts, at every N tested (RMS error within 10% of theory, see 5.2).

*Note:* The Session 2 transcript quotes an analytical value of 0.921951. The correct product is 0.922438 (the value shown in every generated plot). The conclusion is unchanged: the single-run MC estimate quoted there (0.921847) is within one standard error of the correct value.

✓ **VERIFICATION PASSED**

### 5.2 Convergence Study

![Convergence Study](convergence_rms.png)

**Method:** For each N, 200 independent MC estimates were generated and the RMS error relative to the analytical value was computed. A single run's error is itself random, so averaging over repeats is needed to see the true error scaling.

**Result:** The measured RMS error follows the theoretical standard error √(R(1−R)/N) closely. A least-squares fit on the log-log data gives a slope of **-0.498 ± 0.010**, consistent with the theoretical -0.5.

**Quantitative results:**
| N | RMS error (200 runs) | Theory √(R(1−R)/N) | Ratio |
|---|---|---|---|
| 100 | 0.028392 | 0.026748 | 1.06 |
| 300 | 0.015005 | 0.015443 | 0.97 |
| 1,000 | 0.008325 | 0.008458 | 0.98 |
| 3,000 | 0.005028 | 0.004884 | 1.03 |
| 10,000 | 0.002570 | 0.002675 | 0.96 |
| 30,000 | 0.001471 | 0.001544 | 0.95 |
| 100,000 | 0.000932 | 0.000846 | 1.10 |

Ratios stay within 10% of 1, as expected: an RMS computed from 200 runs is itself uncertain by about 5%.

*Original single-run version:* `convergence_study.png` (one MC run per N, up to N = 500,000) shows the same overall trend but is noisy, which is why it was replaced by the repeated-run study above.

✓ **CONVERGENCE VERIFIED**

### 5.3 Redundancy Trade Study

![Redundancy Trade Study](redundancy_trade_study.png)

**Left panel:** Adding one backup to subsystem 5 (highest failure rate, p=0.025) provides about +2.3 percentage points of reliability improvement (analytical: +2.31)—more than any other subsystem.

**Right panel:** Clear correlation between baseline failure probability and value of redundancy. Higher-risk components benefit more from backups.

**Analytical check:** Adding one backup to subsystem i raises system reliability by exactly R·pᵢ, where R = 0.922438 is the baseline:

| Subsystem | pᵢ | Gain from one backup (analytical) |
|---|---|---|
| 1 | 0.010 | +0.92 percentage points |
| 2 | 0.020 | +1.84 percentage points |
| 3 | 0.015 | +1.38 percentage points |
| 4 | 0.010 | +0.92 percentage points |
| 5 | 0.025 | +2.31 percentage points |

Monte Carlo estimates of these gains scatter around these values; the noise on a difference of two single-run estimates is about ±0.12 percentage points at N = 100,000.

**Analytical verification for redundancy:**
- Subsystem with p=0.025, redundancy=2
- Expected reliability: 1 - (0.025)² = 0.999375
- MC estimate: 0.9994 (within statistical error)

✓ **REDUNDANCY MODEL VERIFIED**

### 5.4 Variance Analysis

![Variance Analysis](variance_analysis.png)

**Left panel:** Distribution of 50 MC runs (N = 10,000 each) shows:
- Estimates cluster around the analytical value (red line, 0.922438)
- Spread is consistent with the theoretical normal curve (green); with only 50 runs the histogram is naturally lumpy
- Width matches the theoretical prediction

**Right panel:** 48 of 50 runs (96%) fall inside the 95% confidence band (green); about 2.5 outside is expected.

**Statistical checks:**
- Theoretical std: √(R(1−R)/N) = 0.002675
- Observed std: 0.002689
- Ratio: 1.01 (a sample std from 50 runs is only accurate to about ±10%, so agreement this close is partly luck)

✓ **STATISTICAL BEHAVIOR VERIFIED**

### 5.5 Edge Case Tests

All edge case tests passed:

1. **Perfect system** (p=[0,0,0]): MC = 1.0, Analytical = 1.0 ✓
2. **Failed system** (p=[1,1,1]): MC = 0.0, Analytical = 0.0 ✓
3. **Single subsystem** (p=[0.1]): MC = 0.8998, Analytical = 0.9 (error < 0.5%) ✓
4. **High reliability** (p=[0.0001, 0.0002, 0.0001]): Numerically stable, error < 0.1% ✓
5. **Redundancy** (p=[0.2], k=2): MC = 0.9595, Expected = 0.96 ✓

✓ **ALL EDGE CASES PASSED**

### 5.6 Common-Cause Failure Extension (Beta-Factor Model)

![Common-Cause Study](common_cause_study.png)

The independent model cannot represent a single event that disables all backups. The beta-factor model (Section 2.5) adds this: a fraction β of each unit's failure probability is a common-cause event. Because the model also has a closed form, the analytical expression verifies the simulation.

**Verification (one backup on subsystem 5, N = 1,000,000):**

| β | MC estimate | Analytical | Error (in standard errors) | Status |
|---|---|---|---|---|
| 0.0 | 0.945877 | 0.945499 | 1.66 | ✓ PASS |
| 0.1 | 0.943359 | 0.943248 | 0.48 | ✓ PASS |
| 0.3 | 0.938690 | 0.938707 | 0.07 | ✓ PASS |
| 1.0 | 0.922477 | 0.922438 | 0.14 | ✓ PASS |

- β = 0 reproduces the independent model (0.945499)
- β = 1 makes the backup worthless: reliability equals the no-backup baseline (0.922438)

**Results:** Baseline without redundancy is 92.24%.

| β | Gain from one backup on subsystem 5 | Reliability, all subsystems duplicated |
|---|---|---|
| 0.00 | +2.31 percentage points | 99.86% |
| 0.10 | +2.08 percentage points | 99.09% |
| 0.20 | +1.85 percentage points | 98.32% |

**Takeaways:**
- **Left panel:** Every subsystem's gain from a backup shrinks as β grows. The ranking is unchanged (the highest-risk subsystem still benefits most) but the absolute benefit falls. Subsystems 1 and 4 share the same failure probability, so their curves overlap.
- **Right panel:** Full duplication looks excellent in the independent model (99.86%), but at β = 0.2 mission failure probability rises from 0.14% to 1.68%, about 12× higher. Reliability under full duplication falls to roughly 96% at β = 0.5.
- Independence assumptions for redundant hardware must be justified, not assumed.

✓ **COMMON-CAUSE MODEL VERIFIED**

### 5.7 Verification Summary

| Test Type | Method | Result | Status |
|-----------|--------|--------|--------|
| Analytical comparison | Repeated runs vs. closed form | RMS error within 10% of theoretical σ at every N | ✓ PASS |
| Convergence rate | Log-log slope (200 runs per N) | -0.498 ± 0.010 | ✓ PASS |
| Statistical variance | 50 independent runs | Std ratio ≈ 1.01; 48/50 inside 95% CI | ✓ PASS |
| Edge cases | 5 boundary tests | All passed | ✓ PASS |
| Redundancy model | Analytical formula | Matches MC | ✓ PASS |
| Common-cause model | β = 0, 0.1, 0.3, 1 vs. closed form | All within 2 standard errors | ✓ PASS |

**Overall:** Implementation is verified to be correct within statistical uncertainty.

---

## 6. Results and Interpretation

### 6.1 Baseline System Performance

**Configuration:** 5 subsystems, failure rates [0.01, 0.02, 0.015, 0.01, 0.025]

**Key finding:** Mission reliability is **92.2%**

**Interpretation:**
- Even with individually reliable components (97.5-99% each), compounding failures reduce overall reliability to 92%
- Weakest link (2.5% failure rate) disproportionately impacts system
- This demonstrates why redundancy is critical for high-reliability missions

### 6.2 Value of Redundancy

Adding a single backup to subsystem 5 (highest failure rate):
- Baseline: 92.2%
- With redundancy: 94.5%
- **Improvement: +2.3 percentage points**
- With a 10% common-cause fraction (β = 0.1) the gain drops to +2.08 points (Section 5.6)

**Cost-benefit insight:** Focusing redundancy on highest-risk components provides best return on investment (mass, cost, complexity).

### 6.3 Convergence Insights

**Practical recommendation:** For early-stage analysis, N=10,000-100,000 provides adequate accuracy (±0.5% to ±0.17% at 95% confidence). For final design verification, use N≥500,000 or analytical methods where available.

**Time vs. accuracy tradeoff:**
- N=10,000: ~0.1 seconds, ±0.5% accuracy
- N=100,000: ~1 second, ±0.17% accuracy
- N=1,000,000: ~10 seconds, ±0.05% accuracy

(Timings are approximate for the explicit-loop implementation in `reliability.py`. The vectorized version in `reliability_extensions.py` is much faster: the full 200-repeat convergence study up to N = 100,000 runs in about one second.)

For interactive exploration, 10k samples is sufficient. For reporting, use 100k+.

### 6.4 Statistical Uncertainty

MC provides not just an estimate, but also confidence intervals:
- At N=10,000: R = 0.922 ± 0.005 (95% CI)
- This uncertainty should be propagated to downstream analyses

### 6.5 Engineering Implications

1. **Series reliability compounds quickly:** Five 98% components → 90% system
2. **Target weakest links first:** Redundancy on highest-risk subsystem gives best ROI
3. **Uncertainty matters:** Report confidence intervals, not just point estimates
4. **MC is practical:** Simple to implement, easy to extend to complex scenarios
5. **Correlated failures erode redundancy:** With 20% common-cause failures, full duplication gives 98.3% rather than 99.9%, so independence must be justified

---

## 7. Reflection and Next Steps

### 7.1 What I Learned

**Technical skills:**
- Implementing and verifying Monte Carlo methods
- Convergence analysis and error scaling
- Statistical validation techniques
- Professional software engineering practices (git, CLI, documentation)

**Engineering insights:**
- How quickly reliability degrades in series systems
- The value of quantitative trade studies in design decisions
- Importance of verification at multiple levels (analytical, convergence, edge cases)

**Process lessons:**
- Incremental development with frequent commits is effective
- Visualization makes numerical results much more interpretable
- Verification should be built in from the start, not added later

### 7.2 Challenges Overcome

- **Initial setup confusion** with git repositories → learned to start clean
- **Understanding convergence rate** → deepened knowledge of MC behavior
- **Balancing features vs. time** → focused on core functionality first, added extras incrementally

### 7.3 If I Had More Time

**Extensions I would implement:**

1. **Time-dependent failure rates**
   - Weibull distributions for aging components
   - Mission duration as parameter

2. **Richer correlated-failure models**
   - Beta-factor common-cause model is implemented (Section 5.6); next: subsystem-specific β values and the multiple-Greek-letter model
   - Conditional probabilities

3. **Cost modeling**
   - Integrate component costs
   - Pareto frontier: cost vs. reliability
   - Optimization framework

4. **Interactive web visualization**
   - Real-time parameter adjustment
   - Live plot updates

5. **Variance reduction techniques**
   - Importance sampling for rare events
   - Stratified sampling
   - Quasi-Monte Carlo (low-discrepancy sequences)

### 7.4 Key Takeaways

1. **Monte Carlo is powerful for complex systems:** While slower than analytical methods, the flexibility to model arbitrary scenarios makes MC invaluable for early-stage design.

2. **Verification is not optional:** Multiple verification approaches (analytical, convergence, edge cases, variance) give confidence in results.

3. **Visualization clarifies understanding:** Plots made convergence behavior and trade-off insights immediately obvious.

4. **Software engineering matters:** Clean code, good documentation, and version control make projects maintainable and extensible.

5. **Numerical methods enable engineering decisions:** This tool could directly inform spacecraft design choices about redundancy allocation.

### 7.5 Application to Future Work

The skills developed in this project—MC simulation, convergence analysis, verification techniques, and professional software practices—are directly applicable to:
- Graduate research in astrodynamics/GNC
- Industry roles in spacecraft systems engineering
- Any computational project requiring numerical methods

---

## 8. LLM Transcript
**LLM Used:** Claude (Anthropic)  
**Development Period:** December 6-15, 2025 (core project); October 2026 (review, convergence rework, common-cause extension)

---

## Session 1
**Date:** December 6, 2025, 3 PM

**Me:**
I need to do a mini-project for ASTE 404 that uses numerical methods. I'm thinking about doing something with Monte Carlo simulation for spacecraft reliability - like when you have multiple subsystems and want to know the probability that everything works. Does this sound like a good scope for a final project?

**Claude:**
That's an excellent project idea! Monte Carlo is a solid numerical method choice and spacecraft reliability is very relevant to ASTE. The scope sounds perfect because:

1. Core MC implementation is straightforward
2. You can verify against analytical solutions for independent systems
3. Natural extensions: redundancy, convergence studies, trade-offs
4. Directly applicable to real engineering decisions

For minimum requirements you'd need:
- MC sampling function
- Analytical verification
- Convergence analysis showing O(1/√N) behavior
- Some form of visualization

Would you want to include redundancy modeling? That would make it more interesting.

**Me:**
Yes! I want to model redundancy. So like, if I have 5 subsystems and each has a failure probability, I want to: calculate baseline reliability (all must work), see what happens if I add backup components to different subsystems, figure out which subsystem benefits most from redundancy. I'm thinking this could be like a trade study tool for spacecraft design. Can you help me build this?

**Claude:**
Absolutely! This is a great engineering tool. Let me help you build the complete implementation. Here's what I propose:

**Core Structure:**
1. Basic Monte Carlo reliability estimation
2. Analytical verification functions
3. Redundancy modeling
4. Convergence study
5. Trade study analysis
6. Command-line interface
7. Visualization

I'll build the complete `reliability.py` file for you. Let's start with the foundational functions and then add the advanced features.

---

## Session 2
**Date:** December 12, 2025, 7PM

**Me:**
Ok help me start building. I want the basic MC simulation first - just looping through samples and checking if all subsystems work

**Claude:**
Perfect. Here's the core Monte Carlo function:

```python
import numpy as np
import matplotlib.pyplot as plt

def monte_carlo_reliability(failure_probs, n_samples):
    """
    Estimate system reliability using Monte Carlo simulation.
    
    For a system with multiple independent subsystems, estimates the
    probability that ALL subsystems work (mission success).
    """
    successes = 0
    
    for i in range(n_samples):
        mission_success = True
        
        for fail_prob in failure_probs:
            random_draw = np.random.random()
            if random_draw < fail_prob:
                mission_success = False
                break
        
        if mission_success:
            successes += 1
    
    return successes / n_samples
```

And for verification, here's the analytical solution:

```python
def analytical_reliability(failure_probs):
    """
    Calculate exact reliability for independent subsystems.
    R = (1 - p₁) × (1 - p₂) × ... × (1 - pₙ)
    """
    reliability = 1.0
    for fail_prob in failure_probs:
        reliability *= (1 - fail_prob)
    return reliability
```

**Me:**
Great I tested these with failure rates [0.01, 0.02, 0.015, 0.01, 0.025] with 100k samples.

Results:
- MC estimate: 0.921847
- Analytical: 0.921951  
- Error: 0.000104

So it's working correctly?

**Claude:**
Excellent! That error is well within statistical expectations (~0.0008 standard error). The match confirms your MC implementation is correct. Now let's build the convergence study.

---

## Session 3
**Date:** December 13, 2025, 1 PM

**Me:**
I want to show how the estimate gets better as I increase sample size. Can you build the convergence study function and the plotting code for me?

**Claude:**
Absolutely. I'll create comprehensive functions for convergence analysis with professional visualizations. Here's the complete code:

```python
def convergence_study(failure_probs, n_values):
    """
    Study how Monte Carlo estimate converges as sample size increases.
    Demonstrates the O(1/√N) convergence rate of Monte Carlo methods.
    """
    analytical = analytical_reliability(failure_probs)
    estimates = []
    errors = []
    
    print(f"Running convergence study with {len(n_values)} different sample sizes...")
    
    for i, n in enumerate(n_values):
        estimate = monte_carlo_reliability(failure_probs, n)
        error = abs(estimate - analytical)
        
        estimates.append(estimate)
        errors.append(error)
        
        print(f"  [{i+1}/{len(n_values)}] N = {n:>7,}: "
              f"Estimate = {estimate:.6f}, Error = {error:.6f}")
    
    return estimates, errors


def plot_convergence(n_values, estimates, errors, analytical_value, 
                     save_path='convergence_study.png'):
    """
    Create visualization of convergence behavior with two plots:
    1. MC estimates vs. sample size
    2. Error vs. sample size on log-log scale (showing √N trend)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # LEFT PLOT: Convergence of Estimate
    ax1.plot(n_values, estimates, 'bo-', linewidth=2, markersize=6, 
             label='Monte Carlo Estimate')
    ax1.axhline(analytical_value, color='red', linestyle='--', linewidth=2,
                label=f'Analytical Solution = {analytical_value:.6f}')
    ax1.set_xlabel('Number of Samples (N)', fontsize=12)
    ax1.set_ylabel('Reliability Estimate', fontsize=12)
    ax1.set_title('Convergence of Monte Carlo Estimate', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # RIGHT PLOT: Error vs Sample Size (Log-Log)
    ax2.loglog(n_values, errors, 'bo-', linewidth=2, markersize=6, 
               label='Actual Error')
    
    # Theoretical 1/√N line
    if len(errors) > 0 and errors[0] > 0:
        theoretical_constant = errors[0] * np.sqrt(n_values[0])
        theoretical_errors = theoretical_constant / np.sqrt(np.array(n_values))
        ax2.loglog(n_values, theoretical_errors, 'r--', linewidth=2,
                   label='Theoretical O(1/√N)')
    
    ax2.set_xlabel('Number of Samples (N)', fontsize=12)
    ax2.set_ylabel('Absolute Error', fontsize=12)
    ax2.set_title('Error Convergence Rate', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')
    
    ax2.text(0.05, 0.05, 'Expected slope: -0.5\n(1/√N behavior)', 
             transform=ax2.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to {save_path}")
    plt.show()


def run_convergence_test():
    """Run a complete convergence study with visualization."""
    print("\n" + "=" * 70)
    print("CONVERGENCE STUDY")
    print("=" * 70)
    print()
    
    failure_probs = [0.01, 0.02, 0.015, 0.01, 0.025]
    analytical = analytical_reliability(failure_probs)
    
    print(f"System: {len(failure_probs)} subsystems")
    print(f"Analytical reliability: {analytical:.6f}")
    print()
    
    n_values = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000]
    
    estimates, errors = convergence_study(failure_probs, n_values)
    plot_convergence(n_values, estimates, errors, analytical)
    
    print()
    print("Summary:")
    print(f"  Error at N=100:     {errors[0]:.6f}")
    print(f"  Error at N=500,000: {errors[-1]:.6f}")
    print(f"  Improvement factor: {errors[0]/errors[-1]:.1f}x")
    print(f"  Expected from √N:   {np.sqrt(500000/100):.1f}x")
    print()
    print("=" * 70)
```

This gives you complete convergence analysis with publication-quality plots.

---

## Session 4
**Date:** December 13, 2025, 10 PM

**Me:**
Now I want to add the redundancy modeling. I need a function that can handle cases where some subsystems have backups. Like if redundancy_config = [1, 2, 1, 1, 1], that means subsystem 2 has one backup.

**Claude:**
I'll build the complete redundancy implementation for you, including MC simulation, analytical verification, and trade study analysis:

```python
def monte_carlo_reliability_with_redundancy(failure_probs, redundancy_config, n_samples):
    """
    Calculate reliability with redundant subsystems using Monte Carlo.
    
    For subsystems with redundancy > 1, ALL redundant copies must fail
    for the subsystem to be considered failed.
    """
    successes = 0
    
    for i in range(n_samples):
        mission_success = True
        
        for subsys_idx, (fail_prob, redundancy_level) in enumerate(zip(failure_probs, redundancy_config)):
            # For this subsystem, check if ALL redundant copies fail
            all_copies_failed = True
            
            for copy_num in range(redundancy_level):
                if np.random.random() >= fail_prob:
                    all_copies_failed = False
                    break
            
            if all_copies_failed:
                mission_success = False
                break
        
        if mission_success:
            successes += 1
    
    return successes / n_samples


def analytical_reliability_with_redundancy(failure_probs, redundancy_config):
    """
    Calculate exact reliability with redundancy (for independent failures).
    
    For a subsystem with redundancy level k and failure probability p:
    - Probability all k copies fail: p^k
    - Probability at least one works: 1 - p^k
    """
    reliability = 1.0
    
    for fail_prob, redundancy_level in zip(failure_probs, redundancy_config):
        prob_subsys_works = 1.0 - (fail_prob ** redundancy_level)
        reliability *= prob_subsys_works
    
    return reliability


def redundancy_trade_study(failure_probs, n_samples=100000):
    """
    Study the impact of adding redundancy to each subsystem.
    
    For each subsystem, calculate the reliability improvement from
    adding one backup component (redundancy level 1 -> 2).
    """
    n_subsystems = len(failure_probs)
    base_config = [1] * n_subsystems
    
    print("\n" + "=" * 70)
    print("REDUNDANCY TRADE STUDY")
    print("=" * 70)
    print()
    
    # Calculate baseline reliability
    print("Baseline configuration (no redundancy):")
    base_mc = monte_carlo_reliability_with_redundancy(failure_probs, base_config, n_samples)
    base_analytical = analytical_reliability_with_redundancy(failure_probs, base_config)
    print(f"  MC estimate:  {base_mc:.6f} ({base_mc*100:.2f}%)")
    print(f"  Analytical:   {base_analytical:.6f} ({base_analytical*100:.2f}%)")
    print()
    
    print("Testing redundancy on each subsystem:")
    improvements = []
    
    for i in range(n_subsystems):
        test_config = base_config.copy()
        test_config[i] = 2  # Add one backup
        
        new_mc = monte_carlo_reliability_with_redundancy(failure_probs, test_config, n_samples)
        new_analytical = analytical_reliability_with_redundancy(failure_probs, test_config)
        
        improvement = new_mc - base_mc
        improvements.append(improvement)
        
        print(f"  Subsystem {i+1} (p={failure_probs[i]:.3f}):")
        print(f"    New reliability: {new_mc:.6f} ({new_mc*100:.2f}%)")
        print(f"    Improvement:     {improvement:.6f} ({improvement*100:.2f} percentage points)")
        print(f"    Analytical:      {new_analytical:.6f}")
    
    print()
    print("Summary:")
    best_idx = improvements.index(max(improvements))
    print(f"  Best ROI: Add redundancy to subsystem {best_idx+1}")
    print(f"  (Has failure rate {failure_probs[best_idx]:.3f}, gives +{improvements[best_idx]*100:.2f}% reliability)")
    print("=" * 70)
    
    return improvements, base_mc


def plot_redundancy_trade_study(failure_probs, improvements, baseline_reliability,
                                 save_path='redundancy_trade_study.png'):
    """
    Visualize the redundancy trade study results.
    """
    n_subsystems = len(failure_probs)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # LEFT PLOT: Reliability Improvement
    subsystem_labels = [f'Sub {i+1}\n(p={p:.3f})' for i, p in enumerate(failure_probs)]
    colors = ['#1f77b4' if i != improvements.index(max(improvements)) else '#ff7f0e' 
              for i in range(n_subsystems)]
    
    bars = ax1.bar(range(n_subsystems), [imp*100 for imp in improvements], color=colors)
    ax1.set_xlabel('Subsystem', fontsize=12)
    ax1.set_ylabel('Reliability Improvement (%)', fontsize=12)
    ax1.set_title('Impact of Adding Redundancy to Each Subsystem', 
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(range(n_subsystems))
    ax1.set_xticklabels(subsystem_labels)
    ax1.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, imp) in enumerate(zip(bars, improvements)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{imp*100:.2f}%',
                ha='center', va='bottom', fontsize=9)
    
    # RIGHT PLOT: Failure Rate vs Improvement
    ax2.scatter(failure_probs, [imp*100 for imp in improvements], s=150, alpha=0.7)
    ax2.set_xlabel('Subsystem Failure Probability', fontsize=12)
    ax2.set_ylabel('Reliability Improvement (%)', fontsize=12)
    ax2.set_title('Higher Failure Rate → More Value from Redundancy',
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    for i, (fp, imp) in enumerate(zip(failure_probs, improvements)):
        ax2.annotate(f'Sub {i+1}', (fp, imp*100), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Trade study plot saved to {save_path}")
    plt.show()
```

This complete implementation includes MC simulation, analytical verification, and visualization for the trade study.

---

## Session 5
**Date:** December 14, 2025, 11 AM

**Me:**
I'm trying to set up my GitHub repo but getting errors about nested repositories. What should I do?

**Claude:**
Sounds like you might be initializing git inside an existing repo. Let's start fresh:

```bash
# Navigate to where you want the project
cd C:\Users\lilli\OneDrive\ASTE404

# Create new folder
mkdir spacecraft-reliability
cd spacecraft-reliability

# Initialize git
git init

# Create your files, then:
git add .
git commit -m "Initial project setup"

# Connect to GitHub (create repo on GitHub first)
git remote add origin https://github.com/YOUR-USERNAME/mini-project.git
git push -u origin main
```

**Me:**
Got it working! I want to add a command-line interface so I can run different analyses without editing the code. Can you build the full CLI for me?

**Claude:**
Absolutely! I'll create a comprehensive CLI with argparse. Here's the complete implementation:

```python
def main():
    """Main CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Spacecraft Reliability Monte Carlo Tool - ASTE 404 Mini-Project',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python reliability.py --run basic
  python reliability.py --run basic --failures 0.01 0.02 0.015 --samples 50000
  python reliability.py --run convergence
  python reliability.py --run redundancy
  python reliability.py --run all
  python reliability.py --run basic --failures 0.01 0.02 0.03 --redundancy 1 2 1
        """
    )
    
    parser.add_argument('--run', type=str, 
                       choices=['basic', 'convergence', 'redundancy', 'all'],
                       default='all',
                       help='Which analysis to run (default: all)')
    
    parser.add_argument('--failures', type=float, nargs='+',
                       default=[0.01, 0.02, 0.015, 0.01, 0.025],
                       help='Failure probability for each subsystem')
    
    parser.add_argument('--samples', type=int, default=100000,
                       help='Number of Monte Carlo samples (default: 100000)')
    
    parser.add_argument('--redundancy', type=int, nargs='+', default=None,
                       help='Redundancy level for each subsystem')
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.redundancy is not None:
        if len(args.redundancy) != len(args.failures):
            print("ERROR: Number of redundancy values must match number of subsystems")
            return
    else:
        args.redundancy = [1] * len(args.failures)
    
    # Print header
    print("\n" + "=" * 70)
    print("SPACECRAFT RELIABILITY MONTE CARLO TOOL")
    print("=" * 70)
    print()
    print("System Configuration:")
    print(f"  Number of subsystems: {len(args.failures)}")
    print(f"  Failure probabilities: {args.failures}")
    print(f"  Redundancy levels:     {args.redundancy}")
    print(f"  Monte Carlo samples:   {args.samples:,}")
    print()
    
    # Run requested analysis
    if args.run in ['basic', 'all']:
        print("=" * 70)
        print("BASIC RELIABILITY ANALYSIS")
        print("=" * 70)
        print()
        
        mc_result = monte_carlo_reliability_with_redundancy(
            args.failures, args.redundancy, args.samples)
        
        if all(r == 1 for r in args.redundancy):
            analytical_result = analytical_reliability(args.failures)
        else:
            analytical_result = analytical_reliability_with_redundancy(
                args.failures, args.redundancy)
        
        print(f"Monte Carlo estimate: {mc_result:.6f} ({mc_result*100:.2f}%)")
        print(f"Analytical solution:  {analytical_result:.6f} ({analytical_result*100:.2f}%)")
        print(f"Absolute error:       {abs(mc_result - analytical_result):.6f}")
        print()
        print(f"Mission Success Probability: {mc_result*100:.2f}%")
        print(f"Mission Failure Probability: {(1-mc_result)*100:.2f}%")
        print()
    
    if args.run in ['convergence', 'all']:
        base_failures = args.failures
        n_values = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000]
        
        print("\n" + "=" * 70)
        print("CONVERGENCE STUDY")
        print("=" * 70)
        print()
        
        analytical = analytical_reliability(base_failures)
        print(f"Analytical reliability: {analytical:.6f}")
        print()
        
        estimates, errors = convergence_study(base_failures, n_values)
        plot_convergence(n_values, estimates, errors, analytical)
    
    if args.run in ['redundancy', 'all']:
        improvements, baseline = redundancy_trade_study(args.failures, n_samples=args.samples)
        plot_redundancy_trade_study(args.failures, improvements, baseline)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        main()
    else:
        # Default behavior when run without arguments
        [... default test code ...]
```

This CLI supports custom failure rates, sample sizes, redundancy configurations, and multiple analysis modes.

## Session 6
**Date:** December 14, 2025, 4 PM

**Me:**
I want to show that MC uncertainty matches the theoretical standard deviation. Can you build variance analysis and edge case tests for me?

**Claude:**
Absolutely! Here's the complete variance analysis implementation:

```python
def monte_carlo_variance_analysis(failure_probs, n_samples, n_runs=50):
    """
    Analyze variance in Monte Carlo estimates across multiple runs.
    Demonstrates that statistical uncertainty matches theory.
    """
    print("\n" + "=" * 70)
    print("MONTE CARLO VARIANCE ANALYSIS")
    print("=" * 70)
    print()
    
    analytical = analytical_reliability(failure_probs)
    print(f"System configuration: {len(failure_probs)} subsystems")
    print(f"Analytical reliability: {analytical:.6f}")
    print(f"Running {n_runs} independent MC simulations with {n_samples:,} samples each...")
    print()
    
    results = []
    for i in range(n_runs):
        estimate = monte_carlo_reliability(failure_probs, n_samples)
        results.append(estimate)
        if (i + 1) % 10 == 0:
            print(f"  Completed {i+1}/{n_runs} runs...")
    
    results = np.array(results)
    
    mean_estimate = np.mean(results)
    std_estimate = np.std(results)
    min_estimate = np.min(results)
    max_estimate = np.max(results)
    
    theoretical_std = np.sqrt(analytical * (1 - analytical) / n_samples)
    
    print()
    print("Results:")
    print(f"  Mean estimate:        {mean_estimate:.6f}")
    print(f"  Standard deviation:   {std_estimate:.6f}")
    print(f"  Theoretical std:      {theoretical_std:.6f}")
    print(f"  Min estimate:         {min_estimate:.6f}")
    print(f"  Max estimate:         {max_estimate:.6f}")
    print(f"  Range:                {max_estimate - min_estimate:.6f}")
    print()
    print(f"  Mean error from true: {abs(mean_estimate - analytical):.6f}")
    print(f"  95% confidence:       ±{1.96 * theoretical_std:.6f}")
    print()
    
    ratio = std_estimate / theoretical_std
    if 0.8 <= ratio <= 1.2:
        print(f"✓ Observed std matches theory (ratio = {ratio:.2f})")
    else:
        print(f"⚠ Observed std differs from theory (ratio = {ratio:.2f})")
    
    print("=" * 70)
    
    return results, analytical, theoretical_std


def plot_variance_analysis(results, analytical, theoretical_std, n_samples,
                           save_path='variance_analysis.png'):
    """Visualize Monte Carlo variance across multiple runs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1.hist(results, bins=20, density=True, alpha=0.7, color='skyblue', 
             edgecolor='black', label='MC Estimates')
    ax1.axvline(analytical, color='red', linestyle='--', linewidth=2,
                label=f'Analytical = {analytical:.6f}')
    
    mean_result = np.mean(results)
    x_range = np.linspace(results.min(), results.max(), 100)
    theoretical_pdf = (1 / (theoretical_std * np.sqrt(2 * np.pi))) * \
                      np.exp(-0.5 * ((x_range - analytical) / theoretical_std) ** 2)
    ax1.plot(x_range, theoretical_pdf, 'g-', linewidth=2, 
             label=f'Theoretical N({analytical:.4f}, {theoretical_std:.4f}²)')
    
    ax1.set_xlabel('Reliability Estimate', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title(f'Distribution of MC Estimates (n={n_samples:,})', 
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Time series
    ax2.scatter(range(len(results)), results, alpha=0.6, s=50, color='steelblue')
    ax2.axhline(analytical, color='red', linestyle='--', linewidth=2,
                label='Analytical Solution')
    ax2.axhspan(analytical - 1.96*theoretical_std, 
                analytical + 1.96*theoretical_std,
                alpha=0.2, color='green', label='95% Confidence Interval')
    
    ax2.set_xlabel('Run Number', fontsize=12)
    ax2.set_ylabel('Reliability Estimate', fontsize=12)
    ax2.set_title('MC Estimate Variability Across Runs', 
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Variance analysis plot saved to {save_path}")
    plt.show()


def edge_case_tests():
    """Test edge cases to verify implementation correctness."""
    print("\n" + "=" * 70)
    print("EDGE CASE VERIFICATION TESTS")
    print("=" * 70)
    print()
    
    n_samples = 100000
    all_passed = True
    
    # Test 1: Perfect system
    print("Test 1: Perfect system (all failure probabilities = 0)")
    perfect_probs = [0.0, 0.0, 0.0]
    mc_result = monte_carlo_reliability(perfect_probs, n_samples)
    analytical_result = analytical_reliability(perfect_probs)
    print(f"  MC estimate:  {mc_result:.6f}")
    print(f"  Analytical:   {analytical_result:.6f}")
    if mc_result == 1.0 and analytical_result == 1.0:
        print("  ✓ PASSED: Perfect system has R = 1.0")
    else:
        print("  ✗ FAILED")
        all_passed = False
    print()
    
    # Test 2: Failed system
    print("Test 2: Failed system (all failure probabilities = 1)")
    failed_probs = [1.0, 1.0, 1.0]
    mc_result = monte_carlo_reliability(failed_probs, n_samples)
    analytical_result = analytical_reliability(failed_probs)
    print(f"  MC estimate:  {mc_result:.6f}")
    print(f"  Analytical:   {analytical_result:.6f}")
    if mc_result == 0.0 and analytical_result == 0.0:
        print("  ✓ PASSED: Failed system has R = 0.0")
    else:
        print("  ✗ FAILED")
        all_passed = False
    print()
    
    # Test 3: Single subsystem
    print("Test 3: Single subsystem with p = 0.1")
    single_prob = [0.1]
    mc_result = monte_carlo_reliability(single_prob, n_samples)
    analytical_result


---

## Session 7 (post-submission revision, summarized)
**Date:** October 10, 2026

**Me:**
I asked Claude to review the finished report for use in a portfolio. It flagged inconsistencies (date, convergence interpretation, confidence-interval figures, redundancy terminology) and suggested a repeated-run convergence study and a common-cause failure extension.

**Claude:**
Wrote `reliability_extensions.py` (vectorized MC, repeated-run convergence study, beta-factor common-cause model with analytical verification). I ran it, checked the outputs against the plots, and updated this report. While updating, the analytical baseline quoted in Section 5.1 and Session 2 (0.921951) was found to be wrong; the correct value is 0.922438.

*(This session is summarized, not verbatim.)*

---

## References

- Data Analysis Using Monte Carlo Simulation, Geeks for Geeks (23 Jul, 2025). 
- Spacecraft systems: Fortescue (2011). *Spacecraft Systems Engineering* (4th ed.)
- NumPy Documentation: https://numpy.org/doc/
- Matplotlib Documentation: https://matplotlib.org/
- Fleming, K. N. (1975). *A reliability model for common mode failures in redundant safety systems.* (origin of the beta-factor model)


---

**End of Report**
