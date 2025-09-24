#!/usr/bin/env python3
"""
Check Analysis Environments

Utility script to check and compare environment information across all analyses
stored in a zarr file. Helps identify version inconsistencies and track
which code versions produced which results.
"""

import zarr
import argparse
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich import box
import json


def check_environments(zarr_path: str):
    """Check all environment information in zarr analyses."""
    console = Console()
    root = zarr.open(str(zarr_path), mode='r')
    
    # Analysis groups to check
    analysis_groups = [
        'bout_phase_analysis',
        'bout_analysis',
        'behavior_metrics',
        'filtered_runs',
        'preprocessing'
    ]
    
    environments = []
    
    for group_name in analysis_groups:
        if group_name not in root:
            continue
        
        group = root[group_name]
        
        # Check for runs
        if 'runs' in group:
            runs_group = group['runs']
            for run_name in sorted(runs_group.keys()):
                run_group = runs_group[run_name]
                
                env_info = {
                    'analysis': group_name,
                    'run': run_name,
                    'created_at': run_group.attrs.get('created_at', 'unknown'),
                    'environment': run_group.attrs.get('environment', {})
                }
                environments.append(env_info)
        
        # Also check for latest
        elif 'latest' in group.attrs:
            latest = group.attrs['latest']
            if latest in group:
                run_group = group[latest]
                env_info = {
                    'analysis': group_name,
                    'run': latest,
                    'created_at': run_group.attrs.get('created_at', 'unknown'),
                    'environment': run_group.attrs.get('environment', {})
                }
                environments.append(env_info)
    
    if not environments:
        console.print("[yellow]No analysis runs found with environment tracking.[/yellow]")
        return
    
    # Create comparison table
    table = Table(title="🔬 Analysis Environment Comparison", box=box.ROUNDED)
    table.add_column("Analysis", style="cyan")
    table.add_column("Run", style="yellow")
    table.add_column("Date", style="green")
    table.add_column("Git Commit", style="magenta")
    table.add_column("Dirty", style="red")
    table.add_column("Python", style="blue")
    table.add_column("NumPy", style="blue")
    table.add_column("SciPy", style="blue")
    
    # Track unique versions
    git_commits = set()
    python_versions = set()
    numpy_versions = set()
    
    for env_data in environments:
        env = env_data['environment']
        
        # Parse date
        try:
            dt = datetime.fromisoformat(env_data['created_at'].replace('Z', '+00:00'))
            date_str = dt.strftime("%Y-%m-%d %H:%M")
        except:
            date_str = env_data['created_at'][:16] if len(env_data['created_at']) > 16 else env_data['created_at']
        
        # Get git info
        git_commit = env.get('git_commit_short', 'no tracking')
        git_dirty = "⚠️ Yes" if env.get('git_dirty') else "✓ No" if 'git_dirty' in env else "-"
        
        # Get package versions
        python_ver = env.get('python_version', '-')
        pkg_versions = env.get('package_versions', {})
        numpy_ver = pkg_versions.get('numpy', '-')
        scipy_ver = pkg_versions.get('scipy', '-')
        
        # Track uniques
        if git_commit != 'no tracking' and git_commit != 'unknown':
            git_commits.add(git_commit)
        if python_ver != '-':
            python_versions.add(python_ver)
        if numpy_ver != '-':
            numpy_versions.add(numpy_ver)
        
        # Shorten run name
        run_name = env_data['run'].split('/')[-1] if '/' in env_data['run'] else env_data['run']
        
        table.add_row(
            env_data['analysis'].replace('_', ' ').title(),
            run_name,
            date_str,
            git_commit,
            git_dirty,
            python_ver,
            numpy_ver,
            scipy_ver
        )
    
    console.print(table)
    
    # Summary
    console.print("\n[bold]Summary:[/bold]")
    
    if len(git_commits) > 1:
        console.print(f"  ⚠️  Multiple git commits found: {', '.join(sorted(git_commits))}")
        console.print(f"     Different code versions were used for analyses!")
    elif len(git_commits) == 1:
        console.print(f"  ✓ All analyses use same git commit: {list(git_commits)[0]}")
    else:
        console.print(f"  ❌ No git tracking found - add environment tracking to scripts")
    
    if len(python_versions) > 1:
        console.print(f"  ⚠️  Multiple Python versions: {', '.join(sorted(python_versions))}")
    
    if len(numpy_versions) > 1:
        console.print(f"  ⚠️  Multiple NumPy versions: {', '.join(sorted(numpy_versions))}")
    
    # Check for analyses without environment tracking
    missing_tracking = []
    for group_name in analysis_groups:
        if group_name in root:
            has_env = False
            group = root[group_name]
            if 'runs' in group:
                for run_name in group['runs'].keys():
                    if 'environment' in group['runs'][run_name].attrs:
                        has_env = True
                        break
            if not has_env:
                missing_tracking.append(group_name)
    
    if missing_tracking:
        console.print(f"\n[yellow]Analyses without environment tracking:[/yellow]")
        for name in missing_tracking:
            console.print(f"  • {name}")
        console.print("  [dim]Consider re-running with updated scripts[/dim]")
    
    # Git recommendations
    console.print("\n[bold]Recommendations:[/bold]")
    
    # Check if any analyses had dirty git
    any_dirty = any(env['environment'].get('git_dirty') for env in environments)
    if any_dirty:
        console.print("  ⚠️  Some analyses were run with uncommitted changes")
        console.print("     Commit your code before running analyses for better reproducibility")
    
    console.print("\n[dim]💡 Tip: To see what changed between commits:[/dim]")
    if len(git_commits) == 2:
        commits = sorted(git_commits)
        console.print(f"[dim]   git diff {commits[0]}..{commits[1]}[/dim]")
    elif len(git_commits) == 1:
        commit = list(git_commits)[0]
        console.print(f"[dim]   git diff {commit}..HEAD  # Changes since analysis[/dim]")


def main():
    parser = argparse.ArgumentParser(
        description="Check environment consistency across analyses in zarr file"
    )
    parser.add_argument('zarr_path', help='Path to zarr file')
    
    args = parser.parse_args()
    
    print("="*70)
    print("ANALYSIS ENVIRONMENT CHECK")
    print("="*70)
    
    check_environments(args.zarr_path)


if __name__ == '__main__':
    main()