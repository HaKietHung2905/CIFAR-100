#!/usr/bin/env python
# CIFAR-100 Image Classification Project - Environment Setup Script

import subprocess
import sys
import os
import platform
import argparse


def check_python_version():
    """Check if Python version is compatible"""
    required_version = (3, 7)
    current_version = sys.version_info
    
    if current_version < required_version:
        print(f"Error: Python {required_version[0]}.{required_version[1]} or higher is required.")
        print(f"Current Python version: {current_version[0]}.{current_version[1]}.{current_version[2]}")
        sys.exit(1)
    
    print(f"Python version: {current_version[0]}.{current_version[1]}.{current_version[2]} ✓")


def check_gpu():
    """Check if CUDA is available"""
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            cuda_version = torch.version.cuda
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            
            print(f"CUDA available: Yes ✓")
            print(f"CUDA version: {cuda_version}")
            print(f"GPU count: {device_count}")
            print(f"GPU device: {device_name}")
        else:
            print("CUDA available: No ✗")
            print("Warning: GPU not detected. Training will be slow on CPU.")
    except ImportError:
        print("PyTorch not installed yet. Will check GPU after installation.")


def install_requirements(requirements_file="requirements.txt", use_conda=False, env_name="cifar100"):
    """Install required packages from requirements.txt"""
    if not os.path.exists(requirements_file):
        print(f"Error: Requirements file '{requirements_file}' not found.")
        sys.exit(1)
    
    print(f"Installing requirements from {requirements_file}...")
    
    if use_conda:
        # Check if conda is available
        try:
            subprocess.run(['conda', '--version'], check=True, stdout=subprocess.PIPE)
        except (subprocess.SubprocessError, FileNotFoundError):
            print("Error: Conda is not available. Make sure it's installed and in your PATH.")
            sys.exit(1)
        
        # Create conda environment if it doesn't exist
        try:
            print(f"Creating conda environment '{env_name}'...")
            subprocess.run(['conda', 'create', '-n', env_name, 'python=3.8', '-y'], check=True)
        except subprocess.SubprocessError:
            print(f"Note: Environment '{env_name}' might already exist.")
        
        # Install requirements in conda environment
        install_cmd = [
            'conda', 'install', '-n', env_name, '-y',
            'pytorch', 'torchvision', 'cudatoolkit=11.3', '-c', 'pytorch'
        ]
        
        print("Installing PyTorch with CUDA support...")
        subprocess.run(install_cmd, check=True)
        
        # Install other requirements with pip
        with open(requirements_file, 'r') as f:
            requirements = f.readlines()
        
        # Filter out PyTorch related packages as they're installed with conda
        requirements = [req for req in requirements if not req.startswith(('torch', 'torchvision'))]
        
        if requirements:
            with open('temp_requirements.txt', 'w') as f:
                f.writelines(requirements)
            
            print("Installing other requirements with pip...")
            pip_cmd = [
                'conda', 'run', '-n', env_name, 'pip', 'install', '-r', 'temp_requirements.txt'
            ]
            subprocess.run(pip_cmd, check=True)
            
            # Clean up temporary file
            os.remove('temp_requirements.txt')
        
        print(f"\nConda environment '{env_name}' set up successfully!")
        print(f"Activate it with: conda activate {env_name}")
    
    else:
        # Use pip for installation
        print("Installing with pip...")
        install_cmd = [sys.executable, '-m', 'pip', 'install', '-r', requirements_file]
        subprocess.run(install_cmd, check=True)
        
        print("\nRequirements installed successfully with pip!")


def create_directories():
    """Create necessary project directories if they don't exist"""
    directories = [
        'data',
        'models',
        'results',
        'results/plots',
        'results/model_analysis',
        'results/confusion_matrices'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory '{directory}' created ✓")


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Set up environment for CIFAR-100 project')
    parser.add_argument('--conda', action='store_true',
                        help='Use conda for installation instead of pip')
    parser.add_argument('--env-name', type=str, default='cifar100',
                        help='Conda environment name (default: cifar100)')
    parser.add_argument('--req-file', type=str, default='requirements.txt',
                        help='Requirements file path (default: requirements.txt)')
    return parser.parse_args()


def main():
    """Main function to set up the environment"""
    args = parse_args()
    
    print("=" * 80)
    print("CIFAR-100 Image Classification Project - Environment Setup")
    print("=" * 80)
    
    # Check Python version
    check_python_version()
    
    # Check GPU
    check_gpu()
    
    # Install requirements
    install_requirements(args.req_file, args.conda, args.env_name)
    
    # Create directories
    create_directories()
    
    print("\nEnvironment setup complete! You're ready to start the project.")
    
    if args.conda:
        print(f"\nRemember to activate your conda environment: conda activate {args.env_name}")
    

if __name__ == "__main__":
    main()