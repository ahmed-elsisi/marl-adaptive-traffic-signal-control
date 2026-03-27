#!/usr/bin/env python3
"""
MAPPO Traffic Control - Setup and Validation Script

This script:
1. Validates project structure
2. Checks dependencies
3. Tests SUMO integration
4. Verifies configuration files
5. Provides setup instructions
"""

import os
import sys
from pathlib import Path
import subprocess

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    """Print formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}\n")

def print_success(text):
    """Print success message."""
    print(f"{Colors.GREEN}✓{Colors.END} {text}")

def print_error(text):
    """Print error message."""
    print(f"{Colors.RED}✗{Colors.END} {text}")

def print_warning(text):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠{Colors.END} {text}")

def check_python_version():
    """Check Python version."""
    print_header("Checking Python Version")
    
    version = sys.version_info
    if 3.9 <= version.major + version.minor / 10 < 3.12:
        print_success(f"Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print_error(f"Python version {version.major}.{version.minor} not supported")
        print_warning("Requires Python 3.9-3.11")
        return False

def check_project_structure():
    """Validate project structure."""
    print_header("Validating Project Structure")
    
    required_dirs = [
        'marl_env',
        'models',
        'configs',
        'sumo_network',
        'tests',
        'docs',
    ]
    
    required_files = [
        'marl_env/sumo_env.py',
        'marl_env/obs_builder.py',
        'marl_env/reward_function.py',
        'models/mappo_model.py',
        'configs/mappo_config.yaml',
        'sumo_network/marl-proj.net.xml',
        'sumo_network/marl-proj.rou.xml',
        'sumo_network/marl-proj.add.xml',
        'train_mappo.py',
        'evaluate.py',
        'requirements.txt',
        'README.md',
    ]
    
    all_valid = True
    
    # Check directories
    for dir_name in required_dirs:
        if Path(dir_name).is_dir():
            print_success(f"Directory: {dir_name}")
        else:
            print_error(f"Missing directory: {dir_name}")
            all_valid = False
    
    # Check files
    for file_name in required_files:
        if Path(file_name).is_file():
            print_success(f"File: {file_name}")
        else:
            print_error(f"Missing file: {file_name}")
            all_valid = False
    
    return all_valid

def check_sumo_installation():
    """Check SUMO installation."""
    print_header("Checking SUMO Installation")
    
    # Check SUMO_HOME
    sumo_home = os.environ.get('SUMO_HOME')
    if sumo_home:
        print_success(f"SUMO_HOME: {sumo_home}")
    else:
        print_error("SUMO_HOME not set")
        print_warning("Please set SUMO_HOME environment variable:")
        print("  export SUMO_HOME='/usr/share/sumo'")
        return False
    
    # Check SUMO binary
    try:
        result = subprocess.run(
            ['sumo', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print_success(f"SUMO installed: {version}")
        else:
            print_error("SUMO not working correctly")
            return False
    except FileNotFoundError:
        print_error("SUMO binary not found")
        print_warning("Install SUMO:")
        print("  sudo apt-get install sumo sumo-tools")
        return False
    except Exception as e:
        print_error(f"Error checking SUMO: {e}")
        return False
    
    # Check TraCI
    try:
        tools_path = os.path.join(sumo_home, 'tools')
        if tools_path not in sys.path:
            sys.path.append(tools_path)
        import traci
        print_success("TraCI imported successfully")
    except ImportError:
        print_error("Cannot import TraCI")
        return False
    
    return True

def check_python_dependencies():
    """Check Python dependencies."""
    print_header("Checking Python Dependencies")
    
    required_packages = {
        'ray': '2.35.0',
        'torch': '2.0.0',
        'gymnasium': '0.29.0',
        'numpy': '1.24.0',
        'pandas': '2.0.0',
        'yaml': None,  # PyYAML
        'tensorboard': '2.13.0',
        'matplotlib': '3.7.0',
    }
    
    all_installed = True
    
    for package, min_version in required_packages.items():
        try:
            if package == 'yaml':
                import yaml
                print_success(f"PyYAML: installed")
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                print_success(f"{package}: {version}")
        except ImportError:
            print_error(f"{package}: NOT INSTALLED")
            all_installed = False
    
    if not all_installed:
        print_warning("\nInstall missing dependencies:")
        print("  pip install -r requirements.txt")
    
    return all_installed

def check_gpu_availability():
    """Check GPU availability."""
    print_header("Checking GPU Availability")
    
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            print_success(f"GPU available: {device_name}")
            print_success(f"CUDA devices: {device_count}")
            print_success(f"CUDA version: {torch.version.cuda}")
            return True
        else:
            print_warning("No GPU available - training will be slow")
            print_warning("CPU training time: ~100-150 hours")
            return False
    except ImportError:
        print_warning("PyTorch not installed - cannot check GPU")
        return False

def check_configuration():
    """Validate configuration file."""
    print_header("Validating Configuration")
    
    config_file = Path('configs/mappo_config.yaml')
    
    if not config_file.exists():
        print_error("Configuration file not found")
        return False
    
    try:
        import yaml
        with open(config_file) as f:
            config = yaml.safe_load(f)
        
        # Check critical settings
        mappo_config = config.get('mappo_config', {})
        
        # Check for common mistakes
        if mappo_config.get('num_sgd_iter', 0) > 10:
            print_warning(f"num_sgd_iter = {mappo_config['num_sgd_iter']} (should be 5-10)")
        
        sgd_mini = mappo_config.get('sgd_minibatch_size', 0)
        train_batch = mappo_config.get('train_batch_size', 0)
        if sgd_mini != train_batch:
            print_warning(f"sgd_minibatch_size ({sgd_mini}) != train_batch_size ({train_batch})")
            print_warning("Should be equal for MAPPO (no mini-batching)")
        
        if mappo_config.get('vf_share_layers', True):
            print_warning("vf_share_layers = True (should be False for MAPPO)")
        
        print_success("Configuration file loaded")
        return True
        
    except Exception as e:
        print_error(f"Error loading configuration: {e}")
        return False

def provide_next_steps(all_passed):
    """Provide next steps based on validation results."""
    print_header("Next Steps")
    
    if all_passed:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ All checks passed!{Colors.END}\n")
        print("You can now start training:\n")
        print(f"  {Colors.BOLD}1. Enable libsumo (8x speedup):{Colors.END}")
        print("     export LIBSUMO_AS_TRACI=1\n")
        print(f"  {Colors.BOLD}2. Start training:{Colors.END}")
        print("     python train_mappo.py\n")
        print(f"  {Colors.BOLD}3. Monitor with TensorBoard:{Colors.END}")
        print("     tensorboard --logdir=results/\n")
        print(f"  {Colors.BOLD}4. See training guide:{Colors.END}")
        print("     cat docs/TRAINING_GUIDE.md\n")
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ Some checks failed{Colors.END}\n")
        print("Please fix the issues above before training.\n")
        print(f"{Colors.BOLD}Common fixes:{Colors.END}\n")
        print("  1. Set SUMO_HOME:")
        print("     export SUMO_HOME='/usr/share/sumo'\n")
        print("  2. Install dependencies:")
        print("     pip install -r requirements.txt\n")
        print("  3. Install SUMO:")
        print("     sudo apt-get install sumo sumo-tools\n")

def main():
    """Run all validation checks."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("╔" + "═"*78 + "╗")
    print("║" + "MAPPO Traffic Control - Setup Validation".center(78) + "║")
    print("╚" + "═"*78 + "╝")
    print(f"{Colors.END}")
    
    checks = [
        ("Python Version", check_python_version),
        ("Project Structure", check_project_structure),
        ("SUMO Installation", check_sumo_installation),
        ("Python Dependencies", check_python_dependencies),
        ("GPU Availability", check_gpu_availability),
        ("Configuration", check_configuration),
    ]
    
    results = {}
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print_error(f"Error in {check_name}: {e}")
            results[check_name] = False
    
    # Summary
    print_header("Validation Summary")
    
    for check_name, passed in results.items():
        if passed:
            print_success(f"{check_name}: PASSED")
        else:
            print_error(f"{check_name}: FAILED")
    
    all_passed = all(results.values())
    
    # Next steps
    provide_next_steps(all_passed)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())