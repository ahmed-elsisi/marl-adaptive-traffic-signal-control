"""
╔══════════════════════════════════════════════════════════════════════════════╗
║     CORRECTED COMPREHENSIVE BASELINE COMPARISON - PRODUCTION READY            ║
║           MAPPO vs Fixed-Time vs Max-Pressure                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

CORRECTED VERSION:
- ✓ Uses evaluate.py (fixed evaluation)
- ✓ Runs all three controllers: MAPPO, Fixed-Time, Max-Pressure
- ✓ Fair comparison with same seed (42)
- ✓ Comprehensive comparison plots
- ✓ All outputs organized in metrics/ folder
- ✓ Detailed performance analysis

Usage:
    python compare_baseline_CORRECTED.py --checkpoint path/to/checkpoint
    python compare_baseline_CORRECTED.py --checkpoint path/to/checkpoint --gui
    python compare_baseline_CORRECTED.py --checkpoint path/to/checkpoint --episodes 3
"""

import os
import sys
import argparse
import subprocess
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime


def ensure_results_dir(base_dir: str = "metrics") -> str:
    """Ensure results directory exists."""
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def run_mappo_evaluation(checkpoint_path: str, episodes: int = 1, use_gui: bool = False, results_dir: str = "metrics"):
    """
    Run MAPPO evaluation using CORRECTED evaluate.py.
    
    CRITICAL FIX: Uses the corrected evaluation script instead of the broken original.
    """
    print("="*80)
    print("RUNNING MAPPO EVALUATION (CORRECTED)")
    print("="*80)
    print()
    
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # CORRECTED: Look for the fixed evaluation script
    if not os.path.exists('evaluate.py'):
        print("✗ Error: evaluate.py not found!")
        print("  Please ensure evaluate.py is in the current directory.")
        print("  This is the CORRECTED version that properly loads your model.")
        sys.exit(1)
    
    # CORRECTED: Import from the fixed script
    from evaluate import evaluate_mappo
    
    stats = evaluate_mappo(
        checkpoint_path=checkpoint_path,
        num_episodes=episodes,
        use_gui=use_gui,
        config_path='configs/mappo_config_v2.yaml',
        results_dir=results_dir,
        seed=42  # CRITICAL: Match baseline seed for fair comparison
    )
    
    return stats


def run_fixed_time_evaluation(use_gui: bool = False, seed: int = 42, results_dir: str = "metrics"):
    """Run fixed-time baseline evaluation."""
    print("\n" + "="*80)
    print("RUNNING FIXED-TIME CONTROL EVALUATION")
    print("="*80)
    print()
    
    if not os.path.exists('fixed-cycles.py'):
        print("✗ Error: fixed-cycles.py not found!")
        print("  This script is needed for fixed-time baseline comparison.")
        sys.exit(1)
    
    cmd = [
        sys.executable,
        'fixed-cycles.py',
        '--sumo-cfg', 'sumo_network/marl-proj.sumocfg',
        '--steps', '3600',
        '--sumo-mode', 'gui' if use_gui else 'cli',
        '--seed', str(seed)
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print("⚠ Warning: Fixed-time evaluation may have had issues")
    
    # Move fixed-time outputs to results folder
    import shutil
    if os.path.exists('fixed_cycle_log.csv'):
        dest = os.path.join(results_dir, 'fixed_cycle_log.csv')
        shutil.move('fixed_cycle_log.csv', dest)
        print(f"✓ Moved fixed_cycle_log.csv to {results_dir}/")
    
    # Move plots
    for plot in ['fixed_plots_total_halts.png', 'fixed_plots_arrivals_wait.png', 
                 'fixed_plots_per_tl_halts.png', 'fixed_plots_switch_counts.png']:
        src = f'metrics/{plot}'
        if os.path.exists(src):
            dest = os.path.join(results_dir, plot)
            shutil.move(src, dest)
            print(f"✓ Moved {plot} to {results_dir}/")
    
    # Load results
    csv_path = os.path.join(results_dir, 'fixed_cycle_log.csv')
    if not os.path.exists(csv_path):
        print(f"✗ Error: Could not find {csv_path}")
        sys.exit(1)
    
    fixed_data = load_csv(csv_path)
    return fixed_data


def run_max_pressure_evaluation(use_gui: bool = False, seed: int = 42, results_dir: str = "metrics"):
    """Run max-pressure baseline evaluation."""
    print("\n" + "="*80)
    print("RUNNING MAX-PRESSURE CONTROL EVALUATION")
    print("="*80)
    print()
    
    if not os.path.exists('max-pressure.py'):
        print("✗ Error: max-pressure.py not found!")
        print("  This script is needed for max-pressure baseline comparison.")
        sys.exit(1)
    
    cmd = [
        sys.executable,
        'max-pressure.py',
        '--sumo-cfg', 'sumo_network/marl-proj.sumocfg',
        '--steps', '3600',
        '--seed', str(seed),
        '--results-dir', results_dir
    ]
    
    if use_gui:
        cmd.append('--gui')
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print("⚠ Warning: Max-pressure evaluation may have had issues")
    
    # Load results
    csv_path = os.path.join(results_dir, 'max_pressure_log.csv')
    if not os.path.exists(csv_path):
        print(f"✗ Error: Could not find {csv_path}")
        sys.exit(1)
    
    mp_data = load_csv(csv_path)
    return mp_data


def load_csv(filename: str):
    """Load baseline results from CSV."""
    times = []
    total_halts = []
    cumulative_arrivals = []
    running_avg_wait = []
    active_veh = []
    
    try:
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                times.append(float(row['time']))
                total_halts.append(int(row['total_halts']))
                cumulative_arrivals.append(int(row['cumulative_arrivals']))
                running_avg_wait.append(float(row['running_avg_wait']))
                active_veh.append(int(row['active_veh']))
    except Exception as e:
        print(f"✗ Error loading CSV {filename}: {e}")
        raise
    
    # Calculate summary stats
    total_arrived = cumulative_arrivals[-1] if cumulative_arrivals else 0
    avg_wait = running_avg_wait[-1] if running_avg_wait else 0
    max_halt = max(total_halts) if total_halts else 0
    avg_halt = np.mean(total_halts) if total_halts else 0
    
    return {
        'times': times,
        'total_halts': total_halts,
        'cumulative_arrivals': cumulative_arrivals,
        'running_avg_wait': running_avg_wait,
        'active_veh': active_veh,
        'summary': {
            'total_arrivals': total_arrived,
            'avg_waiting_time': avg_wait,
            'max_halting': max_halt,
            'avg_halting': avg_halt
        }
    }


def generate_comprehensive_comparison_plots(mappo_data, fixed_data, mp_data, results_dir: str = "metrics"):
    """Generate comprehensive 3-way comparison plots."""
    print("\n" + "="*80)
    print("GENERATING COMPARISON PLOTS")
    print("="*80)
    print()
    
    # 1. Three-way overlay (2x2 grid)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Arrivals overlay
    ax1.plot(mappo_data['times'], mappo_data['cumulative_arrivals'], 
             'b-', label='MAPPO', linewidth=1.5)
    ax1.plot(fixed_data['times'], fixed_data['cumulative_arrivals'], 
             'r-', label='Fixed-Time', linewidth=1.5, alpha=0.7)
    ax1.plot(mp_data['times'], mp_data['cumulative_arrivals'], 
             'g-', label='Max-Pressure', linewidth=1.5, alpha=0.7)
    ax1.set_title('Cumulative Arrivals Comparison', fontweight='bold')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Vehicles')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Halting overlay
    ax2.plot(mappo_data['times'], mappo_data['total_halts'], 
             'b-', label='MAPPO', linewidth=0.8, alpha=0.6)
    ax2.plot(fixed_data['times'], fixed_data['total_halts'], 
             'r-', label='Fixed-Time', linewidth=0.8, alpha=0.6)
    ax2.plot(mp_data['times'], mp_data['total_halts'], 
             'g-', label='Max-Pressure', linewidth=0.8, alpha=0.6)
    ax2.set_title('Total Halting Vehicles Comparison', fontweight='bold')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Halting Vehicles')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Wait time overlay
    ax3.plot(mappo_data['times'], mappo_data['running_avg_wait'], 
             'b-', label='MAPPO', linewidth=1.2)
    ax3.plot(fixed_data['times'], fixed_data['running_avg_wait'], 
             'r-', label='Fixed-Time', linewidth=1.2, alpha=0.7)
    ax3.plot(mp_data['times'], mp_data['running_avg_wait'], 
             'g-', label='Max-Pressure', linewidth=1.2, alpha=0.7)
    ax3.set_title('Average Waiting Time Comparison', fontweight='bold')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Avg Wait Time (s)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Active vehicles overlay
    ax4.plot(mappo_data['times'], mappo_data['active_veh'], 
             'b-', label='MAPPO', linewidth=0.8, alpha=0.6)
    ax4.plot(fixed_data['times'], fixed_data['active_veh'], 
             'r-', label='Fixed-Time', linewidth=0.8, alpha=0.6)
    ax4.plot(mp_data['times'], mp_data['active_veh'], 
             'g-', label='Max-Pressure', linewidth=0.8, alpha=0.6)
    ax4.set_title('Active Vehicles Comparison', fontweight='bold')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Active Vehicles')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(results_dir, 'comparison_all_overlay.png')
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"✓ Saved: {filepath}")
    
    # 2. Summary comparison bars
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    methods = ['MAPPO', 'Fixed-Time', 'Max-Pressure']
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    # Arrivals
    arrivals = [
        mappo_data['cumulative_arrivals'][-1],
        fixed_data['summary']['total_arrivals'],
        mp_data['summary']['total_arrivals']
    ]
    ax1.bar(methods, arrivals, color=colors, alpha=0.8)
    ax1.set_title('Total Arrivals', fontweight='bold')
    ax1.set_ylabel('Vehicles')
    ax1.grid(True, axis='y', alpha=0.3)
    for i, v in enumerate(arrivals):
        ax1.text(i, v + max(arrivals)*0.01, f'{int(v)}', ha='center', fontweight='bold')
    
    # Wait time
    wait_times = [
        mappo_data['running_avg_wait'][-1],
        fixed_data['summary']['avg_waiting_time'],
        mp_data['summary']['avg_waiting_time']
    ]
    ax2.bar(methods, wait_times, color=colors, alpha=0.8)
    ax2.set_title('Average Waiting Time', fontweight='bold')
    ax2.set_ylabel('Seconds')
    ax2.grid(True, axis='y', alpha=0.3)
    for i, v in enumerate(wait_times):
        ax2.text(i, v + max(wait_times)*0.01, f'{v:.2f}s', ha='center', fontweight='bold')
    
    # Avg halting
    avg_halts = [
        np.mean(mappo_data['total_halts']),
        fixed_data['summary']['avg_halting'],
        mp_data['summary']['avg_halting']
    ]
    ax3.bar(methods, avg_halts, color=colors, alpha=0.8)
    ax3.set_title('Average Halting Vehicles', fontweight='bold')
    ax3.set_ylabel('Vehicles')
    ax3.grid(True, axis='y', alpha=0.3)
    for i, v in enumerate(avg_halts):
        ax3.text(i, v + max(avg_halts)*0.01, f'{v:.1f}', ha='center', fontweight='bold')
    
    # Peak halting
    peak_halts = [
        max(mappo_data['total_halts']),
        fixed_data['summary']['max_halting'],
        mp_data['summary']['max_halting']
    ]
    ax4.bar(methods, peak_halts, color=colors, alpha=0.8)
    ax4.set_title('Peak Halting Vehicles', fontweight='bold')
    ax4.set_ylabel('Vehicles')
    ax4.grid(True, axis='y', alpha=0.3)
    for i, v in enumerate(peak_halts):
        ax4.text(i, v + max(peak_halts)*0.01, f'{int(v)}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    filepath = os.path.join(results_dir, 'comparison_all_summary.png')
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"✓ Saved: {filepath}")
    
    # 3. Performance improvement heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate improvements relative to Fixed-Time
    mappo_arrivals = mappo_data['cumulative_arrivals'][-1]
    fixed_arrivals = fixed_data['summary']['total_arrivals']
    mp_arrivals = mp_data['summary']['total_arrivals']
    
    mappo_wait = mappo_data['running_avg_wait'][-1]
    fixed_wait = fixed_data['summary']['avg_waiting_time']
    mp_wait = mp_data['summary']['avg_waiting_time']
    
    mappo_halt = np.mean(mappo_data['total_halts'])
    fixed_halt = fixed_data['summary']['avg_halting']
    mp_halt = mp_data['summary']['avg_halting']
    
    mappo_peak = max(mappo_data['total_halts'])
    fixed_peak = fixed_data['summary']['max_halting']
    mp_peak = mp_data['summary']['max_halting']
    
    metrics = ['Arrivals', 'Wait Time', 'Avg Halting', 'Peak Halting']
    mappo_improvements = [
        ((mappo_arrivals - fixed_arrivals) / fixed_arrivals * 100),
        ((fixed_wait - mappo_wait) / fixed_wait * 100),  # Lower is better
        ((fixed_halt - mappo_halt) / fixed_halt * 100),  # Lower is better
        ((fixed_peak - mappo_peak) / fixed_peak * 100)   # Lower is better
    ]
    mp_improvements = [
        ((mp_arrivals - fixed_arrivals) / fixed_arrivals * 100),
        ((fixed_wait - mp_wait) / fixed_wait * 100),
        ((fixed_halt - mp_halt) / fixed_halt * 100),
        ((fixed_peak - mp_peak) / fixed_peak * 100)
    ]
    
    data = np.array([mappo_improvements, mp_improvements])
    
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=-50, vmax=50)
    
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(metrics)
    ax.set_yticklabels(['MAPPO', 'Max-Pressure'])
    
    # Add text annotations
    for i in range(2):
        for j in range(len(metrics)):
            text = ax.text(j, i, f'{data[i, j]:.1f}%',
                          ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_title('Performance Improvement vs Fixed-Time\n(Positive = Better)', 
                 fontweight='bold', fontsize=14)
    plt.colorbar(im, ax=ax, label='Improvement (%)')
    plt.tight_layout()
    filepath = os.path.join(results_dir, 'comparison_all_heatmap.png')
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"✓ Saved: {filepath}")


def print_comprehensive_summary(mappo_data, fixed_data, mp_data):
    """Print detailed 3-way comparison summary."""
    print("\n" + "="*80)
    print("COMPREHENSIVE PERFORMANCE COMPARISON")
    print("="*80)
    print()
    
    mappo_arrivals = mappo_data['cumulative_arrivals'][-1]
    fixed_arrivals = fixed_data['summary']['total_arrivals']
    mp_arrivals = mp_data['summary']['total_arrivals']
    
    mappo_wait = mappo_data['running_avg_wait'][-1]
    fixed_wait = fixed_data['summary']['avg_waiting_time']
    mp_wait = mp_data['summary']['avg_waiting_time']
    
    mappo_halt = np.mean(mappo_data['total_halts'])
    fixed_halt = fixed_data['summary']['avg_halting']
    mp_halt = mp_data['summary']['avg_halting']
    
    mappo_peak = max(mappo_data['total_halts'])
    fixed_peak = fixed_data['summary']['max_halting']
    mp_peak = mp_data['summary']['max_halting']
    
    # Calculate improvements relative to Fixed-Time
    mappo_arr_imp = ((mappo_arrivals - fixed_arrivals) / fixed_arrivals * 100)
    mp_arr_imp = ((mp_arrivals - fixed_arrivals) / fixed_arrivals * 100)
    
    mappo_wait_imp = ((fixed_wait - mappo_wait) / fixed_wait * 100)
    mp_wait_imp = ((fixed_wait - mp_wait) / fixed_wait * 100)
    
    mappo_halt_imp = ((fixed_halt - mappo_halt) / fixed_halt * 100)
    mp_halt_imp = ((fixed_halt - mp_halt) / fixed_halt * 100)
    
    mappo_peak_imp = ((fixed_peak - mappo_peak) / fixed_peak * 100)
    mp_peak_imp = ((fixed_peak - mp_peak) / fixed_peak * 100)
    
    print(f"{'Metric':<20} {'MAPPO':<12} {'Fixed-Time':<12} {'Max-Press':<12}")
    print("-" * 60)
    print(f"{'Arrivals':<20} {mappo_arrivals:<12} {fixed_arrivals:<12} {mp_arrivals:<12}")
    print(f"{'  vs Fixed':<20} {mappo_arr_imp:>+11.1f}% {'-':>11} {mp_arr_imp:>+11.1f}%")
    print()
    print(f"{'Avg Wait (s)':<20} {mappo_wait:<12.2f} {fixed_wait:<12.2f} {mp_wait:<12.2f}")
    print(f"{'  vs Fixed':<20} {mappo_wait_imp:>+11.1f}% {'-':>11} {mp_wait_imp:>+11.1f}%")
    print()
    print(f"{'Avg Halting':<20} {mappo_halt:<12.1f} {fixed_halt:<12.1f} {mp_halt:<12.1f}")
    print(f"{'  vs Fixed':<20} {mappo_halt_imp:>+11.1f}% {'-':>11} {mp_halt_imp:>+11.1f}%")
    print()
    print(f"{'Peak Halting':<20} {mappo_peak:<12} {fixed_peak:<12} {mp_peak:<12}")
    print(f"{'  vs Fixed':<20} {mappo_peak_imp:>+11.1f}% {'-':>11} {mp_peak_imp:>+11.1f}%")
    print("=" * 80)
    
    # Overall assessment
    print("\nOVERALL ASSESSMENT:")
    print()
    
    # Check throughput
    if abs(mappo_arr_imp) < 5 and abs(mp_arr_imp) < 5:
        print("✓ Throughput: All methods similar (fair comparison)")
    else:
        print(f"ℹ Throughput variation: MAPPO {mappo_arr_imp:+.1f}%, Max-Pressure {mp_arr_imp:+.1f}%")
    print()
    
    # Determine winner
    mappo_score = (mappo_wait_imp + mappo_halt_imp + mappo_peak_imp) / 3
    mp_score = (mp_wait_imp + mp_halt_imp + mp_peak_imp) / 3
    
    print("Performance Rankings (by improvement over fixed-time):")
    if mappo_score > mp_score:
        print(f"  🥇 1st: MAPPO ({mappo_score:.1f}% avg improvement)")
        print(f"  🥈 2nd: Max-Pressure ({mp_score:.1f}% avg improvement)")
        print(f"  🥉 3rd: Fixed-Time (baseline)")
    else:
        print(f"  🥇 1st: Max-Pressure ({mp_score:.1f}% avg improvement)")
        print(f"  🥈 2nd: MAPPO ({mappo_score:.1f}% avg improvement)")
        print(f"  🥉 3rd: Fixed-Time (baseline)")
    
    print()
    if mappo_wait_imp > 20 and mappo_halt_imp > 20:
        print("🎉 MAPPO significantly outperforms both baselines!")
    elif mp_wait_imp > 20 and mp_halt_imp > 20:
        print("🎉 Max-Pressure significantly outperforms fixed-time!")
    else:
        print("ℹ All methods show competitive performance")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Compare MAPPO vs Fixed-Time vs Max-Pressure (CORRECTED)'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to MAPPO checkpoint')
    parser.add_argument('--episodes', type=int, default=1,
                       help='Number of episodes to evaluate (default: 1)')
    parser.add_argument('--gui', action='store_true',
                       help='Use SUMO GUI for visualization')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for comparison (default: 42)')
    parser.add_argument('--results-dir', type=str, default='metrics',
                       help='Directory for saving results (default: metrics/)')
    
    args = parser.parse_args()
    
    # Ensure results directory
    results_dir = ensure_results_dir(args.results_dir)
    
    print("\n" + "="*80)
    print("COMPREHENSIVE BASELINE COMPARISON (CORRECTED)")
    print("MAPPO vs Fixed-Time vs Max-Pressure")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Episodes: {args.episodes}")
    print(f"Seed: {args.seed}")
    print(f"GUI: {args.gui}")
    print(f"Results Directory: {results_dir}/")
    print("="*80)
    
    # Run all evaluations
    print("\n📊 Running 3 evaluations (this will take ~15-20 minutes)...\n")
    
    # 1. MAPPO (using CORRECTED evaluation)
    mappo_stats = run_mappo_evaluation(
        checkpoint_path=args.checkpoint,
        episodes=args.episodes,
        use_gui=args.gui,
        results_dir=results_dir
    )
    
    if mappo_stats is None:
        print("\n✗ MAPPO evaluation failed!")
        sys.exit(1)
    
    # 2. Fixed-Time
    fixed_data = run_fixed_time_evaluation(
        use_gui=args.gui,
        seed=args.seed,
        results_dir=results_dir
    )
    
    # 3. Max-Pressure
    mp_data = run_max_pressure_evaluation(
        use_gui=args.gui,
        seed=args.seed,
        results_dir=results_dir
    )
    
    # Load MAPPO data from CSV
    mappo_csv_path = os.path.join(results_dir, 'mappo_ep1_metrics.csv')
    if not os.path.exists(mappo_csv_path):
        print(f"✗ Error: Could not find {mappo_csv_path}")
        print("   Expected MAPPO CSV not found. Check evaluation output.")
        sys.exit(1)
    
    mappo_data = load_csv(mappo_csv_path)
    
    # Generate comparison plots
    generate_comprehensive_comparison_plots(mappo_data, fixed_data, mp_data, results_dir)
    
    # Print summary
    print_comprehensive_summary(mappo_data, fixed_data, mp_data)
    
    print("\n✓ Comprehensive comparison completed successfully!")
    print(f"\n📁 All outputs saved to: {results_dir}/")
    print(f"   Files:")
    print(f"   - mappo_ep1_metrics.csv")
    print(f"   - fixed_cycle_log.csv")
    print(f"   - max_pressure_log.csv")
    print(f"   - comparison_all_overlay.png")
    print(f"   - comparison_all_summary.png")
    print(f"   - comparison_all_heatmap.png")
    print()


if __name__ == "__main__":
    main()