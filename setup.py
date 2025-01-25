#!/usr/bin/env python3
import os
import subprocess
import sys
import platform

def create_venv():
    """Create and activate virtual environment"""
    print("Creating virtual environment...")
    subprocess.run([sys.executable, "-m", "venv", "ai-predict"])
    
    # Activate virtual environment
    if platform.system() == "Windows":
        activate_script = os.path.join("ai-predict", "Scripts", "activate")
    else:
        activate_script = os.path.join("ai-predict", "bin", "activate")
    
    print(f"\nTo activate the virtual environment, run:\n")
    if platform.system() == "Windows":
        print(f"    {activate_script}")
    else:
        print(f"    source {activate_script}")

def install_dependencies():
    """Install Python dependencies"""
    print("\nInstalling Python dependencies...")
    pip_path = os.path.join("ai-predict", "Scripts" if platform.system() == "Windows" else "bin", "pip")
    subprocess.run([pip_path, "install", "-r", "requirements.txt"])

def setup_client():
    """Install client dependencies"""
    print("\nInstalling client dependencies...")
    os.chdir("client")
    subprocess.run(["npm", "install"])
    os.chdir("..")

def main():
    print("Starting setup process...")
    
    # Create and activate virtual environment
    create_venv()
    
    # Install dependencies
    install_dependencies()
    
    # Setup client
    if os.path.exists("client"):
        setup_client()
    
    print("\nSetup complete! 🎉")
    print("\nNext steps:")
    print("1. Create a .env file with your API keys")
    print("2. Activate your virtual environment using the command shown above")

if __name__ == "__main__":
    main() 