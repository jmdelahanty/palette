"""
System information utilities for tracking processing environment.
Optimized for HPC environments, particularly Janelia's LSF cluster.
"""

import platform
import socket
import os
import subprocess
import json
import sys
import shutil
import shlex
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

# === Helper Functions ===

def _which(cmd: str) -> bool:
    """Check if command exists."""
    return shutil.which(cmd) is not None

def _run(cmd: List[str], cwd: Optional[Path] = None, timeout: float = 5.0) -> Optional[str]:
    """Run command safely with timeout."""
    try:
        # Convert Path to string if provided - subprocess needs strings
        working_dir = str(cwd) if cwd else None
        
        result = subprocess.run(
            cmd, 
            cwd=working_dir,
            stdout=subprocess.PIPE,  # Capture stdout
            stderr=subprocess.DEVNULL,  # Suppress stderr
            text=True, 
            timeout=timeout,
            env=os.environ.copy()
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return None

def _find_git_root(start: Path, max_depth: int = 8) -> Optional[Path]:
    """Walk up directory tree to find .git directory."""
    p = start.resolve()
    for _ in range(max_depth):
        if (p / ".git").exists():
            return p
        if p.parent == p:  # Reached root
            break
        p = p.parent
    return None


def _to_jsonable(value: Any) -> Any:
    """Convert runtime values into JSON-safe primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _serialize_args(args: Any) -> Dict[str, Any]:
    if args is None:
        return {}
    if isinstance(args, dict):
        payload = args
    elif hasattr(args, "__dict__"):
        payload = vars(args)
    else:
        return {"value": _to_jsonable(args)}
    return {str(k): _to_jsonable(v) for k, v in payload.items()}

# === Git Information ===

def get_git_info(repo_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Get git repository information.
    
    Args:
        repo_path: Optional path to git repo. If None, searches from current directory.
        
    Returns:
        Dict containing commit hash and dirty status
    """
    # Start from current directory, not __file__ (more robust)
    if repo_path is None:
        repo_path = _find_git_root(Path.cwd())
    
    if not repo_path or not _which("git"):
        return {
            'commit_hash': 'N/A',
            'branch': 'N/A',
            'is_dirty': False,
            'error': 'Not a git repository or git not found'
        }
    
    # Get commit info
    commit_hash = _run(['git', 'rev-parse', 'HEAD'], cwd=repo_path) or 'N/A'
    branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=repo_path) or 'DETACHED'
    status = _run(['git', 'status', '--porcelain'], cwd=repo_path) or ''
    remote_url = _run(['git', 'config', '--get', 'remote.origin.url'], cwd=repo_path) or 'unknown'
    
    # Parse dirty files (cap at 50 to avoid huge metadata)
    dirty_files = status.split('\n')[:50] if status else []
    
    return {
        'top_level': str(repo_path),
        'commit_hash': commit_hash,
        'short_hash': commit_hash[:8] if commit_hash != 'N/A' else 'N/A',
        'branch': branch,
        'is_dirty': bool(status),
        'dirty_files': [f for f in dirty_files if f],  # Remove empty strings
        'remote_url': remote_url
    }

# === Platform Information ===

def get_platform_info(collect_ip: bool = False, disk_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get comprehensive platform and system information.
    Important for multi-machine workflows to track where processing happened.
    
    Args:
        collect_ip: Whether to collect IP address (may hang on some networks)
        disk_path: Path to check disk usage for (default: current directory)
    
    Returns:
        Dict with system, CPU, and environment details
    """
    import getpass
    
    info = {
        # OS Information
        'system': platform.system(),
        'release': platform.release(),
        'version': platform.version(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        
        # Network/Machine identification
        'hostname': socket.gethostname(),
        'fqdn': socket.getfqdn(),
        
        # Python Information  
        'python_version': platform.python_version(),
        'python_implementation': platform.python_implementation(),
        
        # Hardware
        'cpu_cores': os.cpu_count(),
        
        # User
        'username': getpass.getuser(),
    }
    
    # IP address collection (opt-in to avoid hangs)
    if collect_ip:
        try:
            # Use local resolution, avoid external connection
            info['ip_address'] = socket.gethostbyname(info['hostname'])
        except:
            info['ip_address'] = 'unknown'
    
    # CPU details
    cpu_details = get_cpu_info()
    if cpu_details:
        info['cpu_details'] = cpu_details
    
    # Memory and disk info
    try:
        import psutil
        memory = psutil.virtual_memory()
        info['memory'] = {
            'total_gb': round(memory.total / (1024**3), 2),
            'available_gb': round(memory.available / (1024**3), 2),
            'percent_used': memory.percent
        }
        
        # Disk info for specified path (useful for checking zarr write location)
        check_path = Path(disk_path) if disk_path else Path.cwd()
        disk = psutil.disk_usage(str(check_path))
        info['disk'] = {
            'path': str(check_path),
            'total_gb': round(disk.total / (1024**3), 2),
            'available_gb': round(disk.free / (1024**3), 2),
            'percent_used': disk.percent
        }
    except (ImportError, Exception):
        pass
    
    # Check for LSF (Janelia's scheduler)
    if os.environ.get('LSB_JOBID'):
        info['lsf'] = {
            'job_id': os.environ.get('LSB_JOBID'),
            'job_name': os.environ.get('LSB_JOBNAME'),
            'job_index': os.environ.get('LSB_JOBINDEX'),
            'queue': os.environ.get('LSB_QUEUE'),
            'num_processors': os.environ.get('LSB_DJOB_NUMPROC'),
            'hosts': os.environ.get('LSB_HOSTS'),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'cuda_visible_devices_orig': os.environ.get('CUDA_VISIBLE_DEVICES_ORIG'),
        }
    
    # Check for SLURM (common HPC scheduler)
    elif os.environ.get('SLURM_JOB_ID'):
        info['slurm'] = {
            'job_id': os.environ.get('SLURM_JOB_ID'),
            'job_name': os.environ.get('SLURM_JOB_NAME'),
            'node_list': os.environ.get('SLURM_NODELIST'),
            'num_nodes': os.environ.get('SLURM_NNODES'),
            'proc_id': os.environ.get('SLURM_PROCID'),
            'cpus_per_task': os.environ.get('SLURM_CPUS_PER_TASK'),
        }
    
    return info

# === CPU Information ===

def get_cpu_info() -> Optional[Dict[str, Any]]:
    """
    Get detailed CPU information.
    
    Returns:
        Dict with CPU details or None if not available
    """
    # Try cpuinfo library first
    try:
        import cpuinfo
        cpu_info = cpuinfo.get_cpu_info()
        return {
            'model': cpu_info.get('brand_raw', 'N/A'),
            'arch': cpu_info.get('arch_string_raw', 'N/A'),
            'hz_advertised': cpu_info.get('hz_advertised_friendly', 'N/A'),
            'l2_cache_size': cpu_info.get('l2_cache_size', 'N/A'),
            'l3_cache_size': cpu_info.get('l3_cache_size', 'N/A'),
            'flags': cpu_info.get('flags', [])[:10]  # First 10 CPU flags
        }
    except (ImportError, Exception):
        pass
    
    # Fallback to /proc/cpuinfo on Linux
    if platform.system() == 'Linux':
        return get_cpu_info_linux()
    
    return None

def get_cpu_info_linux() -> Optional[Dict[str, Any]]:
    """
    Get CPU info from /proc/cpuinfo on Linux systems.
    
    Returns:
        Dict with CPU details or None if not available
    """
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
        
        # Extract model name
        model = None
        for line in cpuinfo.split('\n'):
            if 'model name' in line:
                model = line.split(':')[1].strip()
                break
        
        # Count physical CPUs
        physical_ids = set()
        for line in cpuinfo.split('\n'):
            if 'physical id' in line:
                physical_ids.add(line.split(':')[1].strip())
        
        return {
            'model': model or 'Unknown',
            'physical_cpus': len(physical_ids) if physical_ids else 1,
            'source': '/proc/cpuinfo'
        }
    except:
        return None

# === GPU Information ===

def get_gpu_info() -> Dict[str, Any]:
    """
    Get GPU information from various sources.
    
    Returns:
        Dict with GPU details from available libraries
    """
    gpu_info = {
        'available': False,
        'devices': [],
        'backend': None
    }
    
    # Check for LSF GPU allocation (Janelia specific)
    if os.environ.get('CUDA_VISIBLE_DEVICES'):
        gpu_info['lsf_allocated_gpus'] = os.environ.get('CUDA_VISIBLE_DEVICES')
    if os.environ.get('CUDA_VISIBLE_DEVICES_ORIG'):
        gpu_info['lsf_original_gpus'] = os.environ.get('CUDA_VISIBLE_DEVICES_ORIG')
    
    # Try PyTorch first (most commonly used)
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info['available'] = True
            gpu_info['backend'] = 'cuda'
            gpu_info['cuda_version'] = torch.version.cuda
            
            # Check for ROCm
            if hasattr(torch.version, 'hip') and torch.version.hip:
                gpu_info['backend'] = 'rocm'
                gpu_info['hip_version'] = torch.version.hip
            
            for i in range(torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(i)
                device_info = {
                    'index': i,
                    'name': torch.cuda.get_device_name(i),
                    'compute_capability': f"{device_props.major}.{device_props.minor}",
                    'total_memory_gb': round(device_props.total_memory / (1024**3), 2),
                    'multi_processor_count': device_props.multi_processor_count
                }
                
                # Get current memory usage if CUDA is initialized
                try:
                    if torch.cuda.is_initialized():
                        device_info['memory_allocated_gb'] = round(
                            torch.cuda.memory_allocated(i) / (1024**3), 2
                        )
                        device_info['memory_reserved_gb'] = round(
                            torch.cuda.memory_reserved(i) / (1024**3), 2
                        )
                except:
                    pass
                
                gpu_info['devices'].append(device_info)
                
    except ImportError:
        pass
    
    # Try CuPy as fallback
    if not gpu_info['available']:
        try:
            import cupy as cp
            gpu_info['available'] = True
            gpu_info['backend'] = 'cupy'
            
            device = cp.cuda.Device()
            gpu_info['devices'].append({
                'index': 0,
                'name': device.name.decode('utf-8') if hasattr(device.name, 'decode') else str(device.name),
                'compute_capability': device.compute_capability,
                'total_memory_gb': round(device.mem_info[1] / (1024**3), 2),
                'free_memory_gb': round(device.mem_info[0] / (1024**3), 2)
            })
        except (ImportError, Exception):
            pass
    
    # Try nvidia-ml-py for additional telemetry
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        
        # If we don't have devices from other sources, create entries
        if not gpu_info['devices']:
            gpu_info['available'] = True
            gpu_info['backend'] = 'nvml'
            gpu_info['devices'] = [{'index': i} for i in range(device_count)]
        
        # Enrich existing entries with telemetry
        for i in range(min(device_count, len(gpu_info['devices']))):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            
            # Add NVML-specific telemetry
            try:
                gpu_info['devices'][i].update({
                    'driver_version': pynvml.nvmlSystemGetDriverVersion().decode('utf-8'),
                    'temperature_c': pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
                    'power_draw_w': round(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0, 1),
                    'power_limit_w': round(pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0, 1),
                    'utilization_percent': pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
                    'memory_utilization_percent': pynvml.nvmlDeviceGetUtilizationRates(handle).memory
                })
            except pynvml.NVMLError:
                pass  # Some queries may not be supported
        
        pynvml.nvmlShutdown()
    except (ImportError, Exception):
        pass
    
    # Fallback to nvidia-smi if nothing else worked
    if not gpu_info['available'] and _which('nvidia-smi'):
        nvidia_smi_info = get_nvidia_smi_info()
        if nvidia_smi_info:
            gpu_info.update(nvidia_smi_info)
    
    return gpu_info

def get_nvidia_smi_info() -> Optional[Dict[str, Any]]:
    """
    Get GPU info using nvidia-smi command.
    
    Returns:
        Dict with GPU info or None if not available
    """
    output = _run([
        'nvidia-smi',
        '--query-gpu=name,memory.total,memory.free,temperature.gpu,utilization.gpu',
        '--format=csv,noheader,nounits'
    ], timeout=3.0)
    
    if not output:
        return None
    
    devices = []
    for i, line in enumerate(output.split('\n')):
        if not line:
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 5:
            devices.append({
                'index': i,
                'name': parts[0],
                'total_memory_mb': int(parts[1]),
                'free_memory_mb': int(parts[2]),
                'temperature_c': int(parts[3]),
                'utilization_percent': int(parts[4])
            })
    
    if devices:
        return {
            'available': True,
            'backend': 'nvidia-smi',
            'devices': devices
        }
    
    return None

# === Package Discovery ===

def get_packages_from_importlib() -> Dict[str, str]:
    """
    Get installed packages using importlib.metadata (fastest method).
    
    Returns:
        Dict of package name to version
    """
    packages = {}
    
    try:
        if sys.version_info >= (3, 8):
            from importlib import metadata
            for dist in metadata.distributions():
                name = dist.metadata['Name'].lower().replace('_', '-')
                packages[name] = dist.version
        else:
            # For Python < 3.8, use pkg_resources
            import pkg_resources
            for dist in pkg_resources.working_set:
                name = dist.project_name.lower().replace('_', '-')
                packages[name] = dist.version
    except Exception:
        pass
    
    return packages

def get_pip_packages() -> Dict[str, str]:
    """
    Get all pip-installed packages using pip list.
    
    Returns:
        Dict of package name to version
    """
    output = _run([sys.executable, '-m', 'pip', 'list', '--format=json'], timeout=6.0)
    
    if not output:
        return {}
    
    try:
        pip_packages = json.loads(output)
        return {
            pkg['name'].lower().replace('_', '-'): pkg['version']
            for pkg in pip_packages
        }
    except (json.JSONDecodeError, KeyError):
        return {}

def get_conda_environment_info() -> Optional[Dict[str, Any]]:
    """
    Get information about the current conda environment.
    
    Returns:
        Dict with environment name, path, and installed packages
    """
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if not conda_prefix:
        return None
    
    env_info = {
        'name': os.environ.get('CONDA_DEFAULT_ENV', 'unknown'),
        'prefix': conda_prefix,
        'python_path': sys.executable,
        'packages': {}
    }
    
    # Try conda list (with short timeout)
    output = _run(['conda', 'list', '--json'], timeout=4.5)
    if output:
        try:
            packages = json.loads(output)
            env_info['packages'] = {
                pkg['name']: pkg['version']
                for pkg in packages
                if pkg.get('name') and pkg.get('version')
            }
        except (json.JSONDecodeError, KeyError):
            pass
    
    # Fallback to conda-meta if needed
    if not env_info['packages']:
        env_info['packages'] = get_conda_packages_from_meta(conda_prefix)
    
    return env_info

def get_conda_packages_from_meta(conda_prefix: str) -> Dict[str, str]:
    """
    Read package versions directly from conda-meta directory.
    
    Args:
        conda_prefix: Path to conda environment
        
    Returns:
        Dict of package name to version
    """
    packages = {}
    conda_meta = Path(conda_prefix) / 'conda-meta'
    
    if not conda_meta.exists():
        return packages
    
    for json_file in conda_meta.glob('*.json'):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                name = data.get('name')
                version = data.get('version')
                if name and version:
                    packages[name] = version
        except (json.JSONDecodeError, KeyError, IOError):
            continue
    
    return packages

def get_software_versions() -> Dict[str, str]:
    """
    Get versions of all installed packages.
    Uses fastest method first (importlib), then pip, then conda.
    
    Returns:
        Dict mapping package names to version strings
    """
    # Try fast methods first
    packages = get_packages_from_importlib()
    
    # If we got very few packages, try pip
    if len(packages) < 50:
        pip_packages = get_pip_packages()
        if pip_packages:
            packages = pip_packages
    
    # Finally check conda (slowest)
    conda_info = get_conda_environment_info()
    if conda_info and conda_info['packages']:
        # Merge conda packages (conda takes precedence)
        packages.update(conda_info['packages'])
    
    # Add Python version at the top
    return {'python': platform.python_version(), **packages}

def get_relevant_packages(
    all_packages: Dict[str, str],
    relevant_patterns: Optional[List[str]] = None
) -> Dict[str, str]:
    """
    Filter packages to only include relevant ones for the pipeline.
    
    Args:
        all_packages: Dict of all installed packages
        relevant_patterns: List of package name patterns to include
        
    Returns:
        Filtered dict of packages
    """
    if relevant_patterns is None:
        # Default patterns for packages relevant to FishEye pipeline
        relevant_patterns = [
            # Core scientific
            'numpy', 'scipy', 'pandas', 'xarray',
            # Image processing
            'scikit-image', 'opencv', 'pillow', 'imageio',
            # Deep learning
            'torch', 'torchvision', 'tensorflow', 'keras',
            # Video
            'decord', 'av', 'ffmpeg',
            # Data storage
            'zarr', 'h5py', 'netcdf4',
            # Visualization
            'matplotlib', 'seaborn', 'plotly', 'bokeh',
            # ML
            'scikit-learn', 'xgboost', 'lightgbm',
            # Parallel computing
            'dask', 'ray', 'joblib', 'numba',
            # GPU
            'cupy', 'cuda', 'cudnn',
            # YOLO/Detection
            'ultralytics', 'yolo',
            # Utils
            'tqdm', 'rich', 'click', 'pyyaml',
            # Jupyter
            'jupyter', 'ipython', 'notebook'
        ]
    
    filtered = {}
    for pattern in relevant_patterns:
        for pkg_name, version in all_packages.items():
            if pattern in pkg_name.lower() and pkg_name not in filtered:
                filtered[pkg_name] = version
    
    return filtered

def get_environment_summary() -> Dict[str, Any]:
    """
    Get a comprehensive summary of the environment.
    
    Returns:
        Dict with environment type, key packages, and stats
    """
    all_packages = get_software_versions()
    conda_prefix = os.environ.get("CONDA_PREFIX")
    conda_default_env = os.environ.get("CONDA_DEFAULT_ENV")
    environment_name = conda_default_env or "none"
    if conda_prefix and Path(sys.prefix).name and environment_name in {"", "base", "none"}:
        # `scripts/py` can run an env Python without `conda activate`, leaving
        # CONDA_DEFAULT_ENV as `base`. sys.prefix is the authoritative runtime.
        environment_name = Path(sys.prefix).name
    
    summary = {
        'environment_type': 'conda' if os.environ.get('CONDA_PREFIX') else 'pip',
        'environment_name': environment_name,
        'conda_default_env': conda_default_env,
        'conda_prefix': conda_prefix,
        'python_executable': sys.executable,
        'sys_prefix': sys.prefix,
        'base_prefix': sys.base_prefix,
        'python_version': platform.python_version(),
        'total_packages': len(all_packages),
    }
    
    # Detect key frameworks
    if 'torch' in all_packages:
        summary['deep_learning_framework'] = f"PyTorch {all_packages['torch']}"
        try:
            import torch
            if hasattr(torch.version, 'cuda'):
                summary['cuda_support'] = True
                summary['cuda_version'] = torch.version.cuda
        except:
            pass
    elif 'tensorflow' in all_packages:
        summary['deep_learning_framework'] = f"TensorFlow {all_packages['tensorflow']}"
    
    # Check for GPU libraries
    gpu_libs = []
    for lib in ['cupy', 'cuda-python', 'pycuda', 'numba']:
        if lib in all_packages:
            gpu_libs.append(f"{lib} {all_packages[lib]}")
    if gpu_libs:
        summary['gpu_libraries'] = gpu_libs
    
    # Get relevant packages for compact storage
    summary['key_packages'] = get_relevant_packages(all_packages)
    
    return summary


def build_invocation_record(
    *,
    tool: str,
    args: Optional[Any] = None,
    argv: Optional[List[str]] = None,
    include_git: bool = True,
    include_environment: bool = True,
) -> Dict[str, Any]:
    """
    Build a compact invocation record for auditability/reproducibility.

    Args:
        tool: Stable tool identifier (e.g. module path).
        args: Parsed args object (argparse.Namespace or dict).
        argv: Raw argv tokens (without interpreter). Defaults to sys.argv[1:].
        include_git: Include git commit/branch/dirty metadata.
        include_environment: Include key package + runtime environment summary.
    """
    argv_tokens = [str(token) for token in (list(argv) if argv is not None else list(sys.argv[1:]))]
    entrypoint = str(sys.argv[0]) if sys.argv else ""
    command_tokens = [entrypoint, *argv_tokens] if entrypoint else argv_tokens
    platform_info = get_platform_info(collect_ip=False)

    record: Dict[str, Any] = {
        "tool": tool,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "cwd": str(Path.cwd()),
        "entrypoint": entrypoint,
        "argv": argv_tokens,
        "command": " ".join(shlex.quote(token) for token in command_tokens),
        "args": _serialize_args(args),
        "platform": {
            "hostname": platform_info.get("hostname"),
            "username": platform_info.get("username"),
            "system": platform_info.get("system"),
            "release": platform_info.get("release"),
        },
    }
    if include_git:
        record["git"] = _to_jsonable(get_git_info())
    if include_environment:
        record["environment"] = _to_jsonable(get_environment_summary())
    return record

# === Main Entry Point ===

def get_environment_info(
    include_all_packages: bool = False,
    disk_path: Optional[str] = None,
    collect_ip: bool = False,
    capture_env_vars: bool = True  # New parameter
) -> Dict[str, Any]:
    """Get comprehensive environment information."""
    env_info = {
        'git': get_git_info(),
        'platform': get_platform_info(collect_ip=collect_ip, disk_path=disk_path),
        'gpu': get_gpu_info(),
        'environment': get_environment_summary()
    }
    
    if capture_env_vars:
        env_info['env_vars'] = get_critical_env_vars()
    
    if include_all_packages:
        env_info['all_packages'] = get_software_versions()
    
    return env_info

def get_gds_config() -> Dict[str, Any]:
    """
    Capture GDS/kvikIO configuration for reproducibility.
    Includes cufile.json and runtime settings.
    """
    gds_info = {}
    
    # Check for cufile.json in standard locations
    cufile_paths = [
        Path('/etc/cufile.json'),
        Path.home() / '.cufile.json',
        Path(os.environ.get('CUFILE_CONFIG_PATH', '/nonexistent'))
    ]
    
    for path in cufile_paths:
        if path.exists():
            try:
                with open(path, 'r') as f:
                    gds_info['cufile_json'] = json.load(f)
                    gds_info['cufile_json_path'] = str(path)
                break
            except Exception as e:
                gds_info['cufile_json_error'] = str(e)
    
    # Check kvikIO runtime settings if available
    try:
        import kvikio
        import kvikio.defaults
        
        # In system.py, fix the kvikio section:
        gds_info['kvikio'] = {
            'compat_mode': kvikio.defaults.get("compat_mode"),
            'thread_pool_size': kvikio.defaults.get("num_threads"),
            'task_size': kvikio.defaults.get("task_size"),
            'gds_enabled': not kvikio.defaults.get("compat_mode"),
            'bounce_buffer_size': kvikio.defaults.get("bounce_buffer_size") if hasattr(kvikio.defaults, 'bounce_buffer_size') else None,
        }
        
        # Try to get kvikIO version
        if hasattr(kvikio, '__version__'):
            gds_info['kvikio']['version'] = kvikio.__version__
            
    except ImportError:
        gds_info['kvikio'] = {'available': False}
    except Exception as e:
        gds_info['kvikio'] = {'error': str(e)}
    
    # Check for GDS driver/module
    gds_module_check = subprocess.run(
        ['lsmod'], 
        capture_output=True, 
        text=True
    )
    if gds_module_check.returncode == 0:
        for line in gds_module_check.stdout.split('\n'):
            if 'nvidia_fs' in line:
                gds_info['nvidia_fs_module'] = True
                break
    
    # Check GDS mount points
    mount_check = subprocess.run(
        ['mount'], 
        capture_output=True, 
        text=True
    )
    if mount_check.returncode == 0:
        gds_mounts = []
        for line in mount_check.stdout.split('\n'):
            if 'gpfs' in line or 'lustre' in line or 'nvme' in line:
                # These are typical GDS-capable filesystems
                gds_mounts.append(line.strip())
        if gds_mounts:
            gds_info['gds_capable_mounts'] = gds_mounts[:5]  # Limit to 5
    
    return gds_info


def get_critical_env_vars() -> Dict[str, str]:
    """
    Capture critical environment variables for reproducibility.
    Only captures relevant variables to avoid storing sensitive data.
    """
    # Define which environment variables to capture
    relevant_patterns = [
        # CUDA/GPU related
        'CUDA_', 'CUDNN_', 'NCCL_', 'NVIDIA_',
        # kvikIO/GDS
        'KVIKIO_', 'GDS_',
        # Performance/threading
        'OMP_', 'MKL_', 'BLOSC_', 'NUMEXPR_',
        # Python/conda
        'CONDA_', 'VIRTUAL_ENV', 'PYTHONPATH',
        # HPC/scheduler
        'LSB_', 'SLURM_', 'PBS_',
        # Library paths (but not full PATH for security)
        'LD_LIBRARY_PATH', 'LD_PRELOAD',
    ]
    
    captured_vars = {}
    for var, value in os.environ.items():
        # Check if variable matches any relevant pattern
        for pattern in relevant_patterns:
            if var.startswith(pattern):
                captured_vars[var] = value
                break
        # Also capture exact matches
        if var in ['HOME', 'USER', 'SHELL', 'TERM']:
            captured_vars[var] = value
    
    return captured_vars
