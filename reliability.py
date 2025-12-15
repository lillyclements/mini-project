"""
Spacecraft Reliability Monte Carlo Tool
ASTE 404 Mini-Project - Fall 2025
Author: Lilly Clements

Estimates spacecraft mission reliability using Monte Carlo simulation
with support for redundant subsystems and comprehensive verification.

Key Features:
- Monte Carlo reliability estimation for series systems
- Analytical verification for independent subsystems
- Convergence analysis demonstrating O(1/√N) error scaling
- Redundancy modeling and trade study analysis
- Statistical variance analysis across multiple runs
- Edge case verification tests
- Command-line interface for flexible usage

Usage:
    python reliability.py                    # Run all analyses
    python reliability.py --help             # Show CLI options
    python reliability.py --run basic        # Run basic analysis only
"""

import numpy as np
import matplotlib.pyplot as plt


def monte_carlo_reliability(failure_probs, n_samples):
    """
    Estimate system reliability using Monte Carlo simulation.
    
    For a system with multiple independent subsystems, estimates the
    probability that ALL subsystems work (mission success).
    
    Args:
        failure_probs: list of failure probabilities for each subsystem
                      e.g., [0.01, 0.02, 0.015] means subsystem 1 has 1% 
                      failure rate, subsystem 2 has 2%, etc.
        n_samples: number of Monte Carlo samples to run
    
    Returns:
        float: estimated reliability (probability all subsystems work)
    
    Example:
        >>> failure_probs = [0.01, 0.02, 0.015]
        >>> reliability = monte_carlo_reliability(failure_probs, 100000)
        >>> print(f"Reliability: {reliability:.4f}")
    """
    n_subsystems = len(failure_probs)
    successes = 0
    
    # Run Monte Carlo simulation
    for i in range(n_samples):
        # Assume mission succeeds unless a subsystem fails
        mission_success = True
        
        # Check each subsystem
        for fail_prob in failure_probs:
            # Generate random number between 0 and 1
            random_draw = np.random.random()
            
            # If random draw < failure probability, this subsystem fails
            if random_draw < fail_prob:
                mission_success = False
                break  # No need to check other subsystems
        
        # Count successful missions
        if mission_success:
            successes += 1
    
    # Return the fraction of successful missions
    return successes / n_samples


def analytical_reliability(failure_probs):
    """
    Calculate exact reliability for independent subsystems.
    
    For independent subsystems, the probability that all work is:
    R = P(all work) = P(1 works) × P(2 works) × ... × P(N works)
    R = (1 - p₁) × (1 - p₂) × ... × (1 - pₙ)
    
    This provides an analytical solution to verify our Monte Carlo results.
    
    Args:
        failure_probs: list of failure probabilities for each subsystem
    
    Returns:
        float: exact reliability
    
    Example:
        >>> failure_probs = [0.01, 0.02, 0.015]
        >>> exact = analytical_reliability(failure_probs)
        >>> print(f"Exact reliability: {exact:.6f}")
    """
    reliability = 1.0
    
    # Multiply probabilities of each subsystem working
    for fail_prob in failure_probs:
        prob_works = 1.0 - fail_prob  # P(works) = 1 - P(fails)
        reliability *= prob_works
    
    return reliability

def convergence_study(failure_probs, n_values):
    """
    Study how Monte Carlo estimate converges as sample size increases.
    
    This demonstrates the O(1/√N) convergence rate of Monte Carlo methods.
    As we increase the number of samples N, the error should decrease
    proportionally to 1/√N.
    
    Args:
        failure_probs: list of failure probabilities for each subsystem
        n_values: list of sample sizes to test (e.g., [100, 1000, 10000, ...])
    
    Returns:
        tuple: (estimates, errors) where:
            - estimates: list of MC estimates for each N
            - errors: list of absolute errors compared to analytical solution
    """
    # Calculate the true analytical value
    analytical = analytical_reliability(failure_probs)
    
    estimates = []
    errors = []
    
    print(f"Running convergence study with {len(n_values)} different sample sizes...")
    
    for i, n in enumerate(n_values):
        # Run Monte Carlo with this sample size
        estimate = monte_carlo_reliability(failure_probs, n)
        error = abs(estimate - analytical)
        
        estimates.append(estimate)
        errors.append(error)
        
        # Print progress
        print(f"  [{i+1}/{len(n_values)}] N = {n:>7,}: "
              f"Estimate = {estimate:.6f}, Error = {error:.6f}")
    
    return estimates, errors


def plot_convergence(n_values, estimates, errors, analytical_value, 
                     save_path='convergence_study.png'):
    """
    Create visualization of convergence behavior.
    
    Generates two plots:
    1. MC estimates vs. sample size (compared to analytical solution)
    2. Error vs. sample size on log-log scale (showing √N trend)
    
    Args:
        n_values: list of sample sizes tested
        estimates: list of MC estimates
        errors: list of absolute errors
        analytical_value: the exact analytical solution
        save_path: filename to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # ===== LEFT PLOT: Convergence of Estimate =====
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
    
    # ===== RIGHT PLOT: Error vs Sample Size (Log-Log) =====
    ax2.loglog(n_values, errors, 'bo-', linewidth=2, markersize=6, 
               label='Actual Error')
    
    # Add theoretical 1/√N line for comparison
    # Use first point to calibrate the constant
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
    
    # Add slope annotation
    ax2.text(0.05, 0.05, 'Expected slope: -0.5\n(1/√N behavior)', 
             transform=ax2.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to {save_path}")
    plt.show()

def monte_carlo_reliability_with_redundancy(failure_probs, redundancy_config, n_samples):
    """
    Calculate reliability with redundant subsystems using Monte Carlo.
    
    For subsystems with redundancy > 1, ALL redundant copies must fail
    for the subsystem to be considered failed. This models backup components.
    
    Args:
        failure_probs: list of base failure probabilities for each subsystem
        redundancy_config: list of redundancy levels for each subsystem
                          (1 = no redundancy, 2 = one backup, 3 = two backups, etc.)
        n_samples: number of Monte Carlo samples
    
    Returns:
        float: estimated reliability with redundancy
    
    Example:
        >>> failure_probs = [0.01, 0.02, 0.015]
        >>> redundancy = [1, 2, 1]  # Add backup to subsystem 2
        >>> rel = monte_carlo_reliability_with_redundancy(failure_probs, redundancy, 100000)
    """
    successes = 0
    
    for i in range(n_samples):
        mission_success = True
        
        # Check each subsystem
        for subsys_idx, (fail_prob, redundancy_level) in enumerate(zip(failure_probs, redundancy_config)):
            # For this subsystem, check if ALL redundant copies fail
            all_copies_failed = True
            
            for copy_num in range(redundancy_level):
                # Check if this copy works
                if np.random.random() >= fail_prob:
                    # At least one copy works, so subsystem works
                    all_copies_failed = False
                    break
            
            # If all copies failed, the mission fails
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
    
    Args:
        failure_probs: list of failure probabilities
        redundancy_config: list of redundancy levels
    
    Returns:
        float: exact reliability
    """
    reliability = 1.0
    
    for fail_prob, redundancy_level in zip(failure_probs, redundancy_config):
        # Probability that at least one copy works
        prob_subsys_works = 1.0 - (fail_prob ** redundancy_level)
        reliability *= prob_subsys_works
    
    return reliability


def redundancy_trade_study(failure_probs, n_samples=100000):
    """
    Study the impact of adding redundancy to each subsystem.
    
    For each subsystem, calculate the reliability improvement from
    adding one backup component (redundancy level 1 -> 2).
    
    Args:
        failure_probs: list of failure probabilities
        n_samples: number of MC samples for each configuration
    
    Returns:
        list: reliability improvement for adding redundancy to each subsystem
    """
    n_subsystems = len(failure_probs)
    base_config = [1] * n_subsystems  # No redundancy
    
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
        # Create config with redundancy on subsystem i
        test_config = base_config.copy()
        test_config[i] = 2  # Add one backup
        
        # Calculate new reliability
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
    
    Creates a bar chart showing reliability improvement from adding
    redundancy to each subsystem.
    """
    n_subsystems = len(failure_probs)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # ===== LEFT PLOT: Reliability Improvement =====
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
    
    # Add value labels on bars
    for i, (bar, imp) in enumerate(zip(bars, improvements)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{imp*100:.2f}%',
                ha='center', va='bottom', fontsize=9)
    
    # ===== RIGHT PLOT: Failure Rate vs Improvement =====
    ax2.scatter(failure_probs, [imp*100 for imp in improvements], s=150, alpha=0.7)
    ax2.set_xlabel('Subsystem Failure Probability', fontsize=12)
    ax2.set_ylabel('Reliability Improvement (%)', fontsize=12)
    ax2.set_title('Higher Failure Rate → More Value from Redundancy',
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add subsystem labels to points
    for i, (fp, imp) in enumerate(zip(failure_probs, improvements)):
        ax2.annotate(f'Sub {i+1}', (fp, imp*100), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Trade study plot saved to {save_path}")
    plt.show()

def monte_carlo_variance_analysis(failure_probs, n_samples, n_runs=50):
    """
    Analyze variance in Monte Carlo estimates across multiple runs.
    
    This demonstrates the statistical uncertainty inherent in MC methods.
    With n_samples per run, the standard deviation of estimates should
    be approximately σ ≈ sqrt(p(1-p)/n) where p is the true reliability.
    
    Args:
        failure_probs: list of failure probabilities
        n_samples: number of samples per MC run
        n_runs: number of independent MC runs to perform
    
    Returns:
        tuple: (results, analytical, theoretical_std)
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
    
    # Calculate statistics
    mean_estimate = np.mean(results)
    std_estimate = np.std(results)
    min_estimate = np.min(results)
    max_estimate = np.max(results)
    
    # Theoretical standard deviation for binomial proportion
    # σ ≈ sqrt(p(1-p)/n)
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
    
    # Check if observed std matches theory
    ratio = std_estimate / theoretical_std
    if 0.8 <= ratio <= 1.2:
        print(f"✓ Observed std matches theory (ratio = {ratio:.2f})")
    else:
        print(f"⚠ Observed std differs from theory (ratio = {ratio:.2f})")
    
    print("=" * 70)
    
    return results, analytical, theoretical_std


def plot_variance_analysis(results, analytical, theoretical_std, n_samples,
                           save_path='variance_analysis.png'):
    """
    Visualize Monte Carlo variance across multiple runs.
    
    Creates two plots:
    1. Histogram of MC estimates showing distribution
    2. Time series of estimates showing run-to-run variability
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # ===== LEFT PLOT: Histogram =====
    ax1.hist(results, bins=20, density=True, alpha=0.7, color='skyblue', 
             edgecolor='black', label='MC Estimates')
    
    # Add analytical solution line
    ax1.axvline(analytical, color='red', linestyle='--', linewidth=2,
                label=f'Analytical = {analytical:.6f}')
    
    # Add theoretical normal distribution
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
    
    # ===== RIGHT PLOT: Time Series =====
    ax2.scatter(range(len(results)), results, alpha=0.6, s=50, color='steelblue')
    ax2.axhline(analytical, color='red', linestyle='--', linewidth=2,
                label='Analytical Solution')
    
    # Add confidence interval bands
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
    """
    Test edge cases to verify implementation correctness.
    
    Tests:
    1. All subsystems perfect (p=0) → R should be 1.0
    2. All subsystems fail (p=1) → R should be 0.0
    3. Single subsystem → R = 1 - p
    4. Very high reliability system → numerically stable
    """
    print("\n" + "=" * 70)
    print("EDGE CASE VERIFICATION TESTS")
    print("=" * 70)
    print()
    
    n_samples = 100000
    all_passed = True
    
    # Test 1: Perfect system (all p=0)
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
    
    # Test 2: Failed system (all p=1)
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
    analytical_result = analytical_reliability(single_prob)
    error = abs(mc_result - analytical_result)
    print(f"  MC estimate:  {mc_result:.6f}")
    print(f"  Analytical:   {analytical_result:.6f} (= 1 - 0.1 = 0.9)")
    print(f"  Error:        {error:.6f}")
    if error < 0.005:  # Allow small statistical error
        print("  ✓ PASSED: Single subsystem test within tolerance")
    else:
        print("  ✗ FAILED: Error too large")
        all_passed = False
    print()
    
    # Test 4: High reliability system
    print("Test 4: High reliability system (very small failure rates)")
    high_rel_probs = [0.0001, 0.0002, 0.0001]
    mc_result = monte_carlo_reliability(high_rel_probs, n_samples)
    analytical_result = analytical_reliability(high_rel_probs)
    error = abs(mc_result - analytical_result)
    print(f"  MC estimate:  {mc_result:.6f}")
    print(f"  Analytical:   {analytical_result:.6f}")
    print(f"  Error:        {error:.6f}")
    if error < 0.001:
        print("  ✓ PASSED: High reliability test within tolerance")
    else:
        print("  ✗ FAILED: Error too large")
        all_passed = False
    print()
    
    # Test 5: Redundancy edge case
    print("Test 5: Redundancy - single component with backup")
    single_with_backup = [0.2]
    redundancy = [2]
    mc_result = monte_carlo_reliability_with_redundancy(single_with_backup, redundancy, n_samples)
    analytical_result = analytical_reliability_with_redundancy(single_with_backup, redundancy)
    expected = 1 - (0.2 ** 2)  # 1 - p^2
    print(f"  MC estimate:  {mc_result:.6f}")
    print(f"  Analytical:   {analytical_result:.6f}")
    print(f"  Expected:     {expected:.6f} (= 1 - 0.2²)")
    error = abs(mc_result - analytical_result)
    if error < 0.005:
        print("  ✓ PASSED: Redundancy calculation correct")
    else:
        print("  ✗ FAILED: Error too large")
        all_passed = False
    print()
    
    # Summary
    print("-" * 70)
    if all_passed:
        print("✓✓✓ ALL EDGE CASE TESTS PASSED ✓✓✓")
    else:
        print("⚠ SOME TESTS FAILED - Review implementation")
    print("=" * 70)
    
    return all_passed

# ============================================================================
# CONVERGENCE STUDY TEST
# ============================================================================

def run_convergence_test():
    """Run a complete convergence study with visualization."""
    print("\n" + "=" * 70)
    print("CONVERGENCE STUDY")
    print("=" * 70)
    print()
    
    # System configuration
    failure_probs = [0.01, 0.02, 0.015, 0.01, 0.025]
    analytical = analytical_reliability(failure_probs)
    
    print(f"System: {len(failure_probs)} subsystems")
    print(f"Analytical reliability: {analytical:.6f}")
    print()
    
    # Test different sample sizes (spanning 4 orders of magnitude)
    n_values = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000]
    
    # Run convergence study
    estimates, errors = convergence_study(failure_probs, n_values)
    
    # Create visualization
    plot_convergence(n_values, estimates, errors, analytical)
    
    # Summary
    print()
    print("Summary:")
    print(f"  Error at N=100:     {errors[0]:.6f}")
    print(f"  Error at N=500,000: {errors[-1]:.6f}")
    print(f"  Improvement factor: {errors[0]/errors[-1]:.1f}x")
    print(f"  Expected from √N:   {np.sqrt(500000/100):.1f}x")
    print()
    print("=" * 70)
 
# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Spacecraft Reliability Monte Carlo Tool - ASTE 404 Mini-Project',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run with default 5-subsystem configuration
  python reliability.py --run basic
  
  # Custom failure probabilities
  python reliability.py --run basic --failures 0.01 0.02 0.015 --samples 50000
  
  # Run convergence study
  python reliability.py --run convergence
  
  # Run redundancy trade study
  python reliability.py --run redundancy
  
  # Run all analyses
  python reliability.py --run all
  
  # Custom system with redundancy
  python reliability.py --run basic --failures 0.01 0.02 0.03 --redundancy 1 2 1 --samples 100000
        """
    )
    
    parser.add_argument('--run', type=str, 
                       choices=['basic', 'convergence', 'redundancy', 'all'],
                       default='all',
                       help='Which analysis to run (default: all)')
    
    parser.add_argument('--failures', type=float, nargs='+',
                       default=[0.01, 0.02, 0.015, 0.01, 0.025],
                       help='Failure probability for each subsystem (default: [0.01, 0.02, 0.015, 0.01, 0.025])')
    
    parser.add_argument('--samples', type=int, default=100000,
                       help='Number of Monte Carlo samples (default: 100000)')
    
    parser.add_argument('--redundancy', type=int, nargs='+', default=None,
                       help='Redundancy level for each subsystem (default: all 1, no redundancy)')
    
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
        
        # Run MC simulation
        mc_result = monte_carlo_reliability_with_redundancy(
            args.failures, args.redundancy, args.samples)
        
        # Calculate analytical (only if no redundancy or simple redundancy)
        if all(r == 1 for r in args.redundancy):
            analytical_result = analytical_reliability(args.failures)
            print(f"Monte Carlo estimate: {mc_result:.6f} ({mc_result*100:.2f}%)")
            print(f"Analytical solution:  {analytical_result:.6f} ({analytical_result*100:.2f}%)")
            print(f"Absolute error:       {abs(mc_result - analytical_result):.6f}")
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
        # Use base configuration (no redundancy) for convergence study
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
    
        if args.run in ['all']:
            # Variance analysis
            results, analytical, theoretical_std = monte_carlo_variance_analysis(
                args.failures, n_samples=10000, n_runs=50)
            plot_variance_analysis(results, analytical, theoretical_std, 10000)
    
            # Edge case tests
            edge_case_tests()

        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)
        print()


# Entry point when run as script
if __name__ == "__main__":
    # Check if running with command line arguments
    import sys
    
    if len(sys.argv) > 1:
        # CLI mode
        main()
    else:
        # Default behavior - run all tests (original behavior)
        print("=" * 70)
        print("SPACECRAFT RELIABILITY MONTE CARLO TOOL")
        print("=" * 70)
        print()
        
        failure_probs = [0.01, 0.02, 0.015, 0.01, 0.025]
        
        print("Test System Configuration:")
        print(f"  Number of subsystems: {len(failure_probs)}")
        print(f"  Failure probabilities: {failure_probs}")
        print()
        
        # Basic test
        n_samples = 100000
        print(f"Running Monte Carlo with {n_samples:,} samples...")
        mc_result = monte_carlo_reliability(failure_probs, n_samples)
        analytical_result = analytical_reliability(failure_probs)
        
        print()
        print("Results:")
        print(f"  Monte Carlo estimate: {mc_result:.6f} ({mc_result*100:.2f}%)")
        print(f"  Analytical solution:  {analytical_result:.6f} ({analytical_result*100:.2f}%)")
        print(f"  Absolute error:       {abs(mc_result - analytical_result):.6f}")
        print()
        
        if abs(mc_result - analytical_result) < 0.001:
            print("✓ VERIFICATION PASSED: MC estimate matches analytical solution!")
        
        print("=" * 70)
        
        # Run all studies
        run_convergence_test()
        improvements, baseline = redundancy_trade_study(failure_probs, n_samples=100000)
        plot_redundancy_trade_study(failure_probs, improvements, baseline)
        
        # Variance analysis
        print("\nRunning variance analysis...")
        results, analytical, theoretical_std = monte_carlo_variance_analysis(
            failure_probs, n_samples=10000, n_runs=50)
        plot_variance_analysis(results, analytical, theoretical_std, 10000)

        # Edge case tests
        edge_case_tests()

        print("\n" + "=" * 70)
        print("ALL TESTS COMPLETE")
        print("=" * 70)
        print("\nTip: Run 'python reliability.py --help' to see CLI options")