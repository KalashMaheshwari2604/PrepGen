"""
Setup script for PrepGen AI Service
Automates the installation and verification process
"""
import subprocess
import sys
import os


def run_command(command, description):
    """Run a shell command and print status"""
    print(f"\n{'='*60}")
    print(f"📦 {description}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✅ {description} - SUCCESS")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is 3.11+"""
    print("\n🔍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 11:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
        return True
    else:
        print(f"❌ Python 3.11+ required, found {version.major}.{version.minor}.{version.micro}")
        return False


def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    directories = ["logs", "cache", "cache/embeddings", "cache/indices", "temp_uploads", "tests"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✅ Created {directory}/")
    return True


def create_env_file():
    """Create .env file from .env.example if it doesn't exist"""
    if not os.path.exists(".env") and os.path.exists(".env.example"):
        print("\n📄 Creating .env file from template...")
        with open(".env.example", "r") as src:
            with open(".env", "w") as dst:
                dst.write(src.read())
        print("  ✅ Created .env file")
        print("  ⚠️  Please review and update .env with your settings")
        return True
    elif os.path.exists(".env"):
        print("\n✅ .env file already exists")
        return True
    else:
        print("\n⚠️  .env.example not found, skipping .env creation")
        return True


def main():
    """Main setup function"""
    print("""
    ╔════════════════════════════════════════════════════════╗
    ║                                                        ║
    ║        PrepGen AI Service - Setup Script              ║
    ║                  Version 2.0.0                         ║
    ║                                                        ║
    ╚════════════════════════════════════════════════════════╝
    """)
    
    # Check Python version
    if not check_python_version():
        print("\n❌ Setup aborted. Please install Python 3.11 or higher.")
        return False
    
    # Create directories
    if not create_directories():
        print("\n⚠️  Failed to create some directories")
    
    # Create .env file
    create_env_file()
    
    # Install requirements
    success = run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing dependencies from requirements.txt"
    )
    if not success:
        print("\n⚠️  Some dependencies may have failed to install")
        print("    You can try installing them manually:")
        print("    pip install -r requirements.txt")
    
    # Download NLTK data
    if os.path.exists("download_nltk.py"):
        run_command(
            f"{sys.executable} download_nltk.py",
            "Downloading NLTK data"
        )
    
    # Check if models exist
    print("\n🔍 Checking for AI models...")
    if os.path.exists("my_final_cnn_model"):
        print("  ✅ Custom T5 model found (my_final_cnn_model/)")
    else:
        print("  ⚠️  Custom T5 model not found (my_final_cnn_model/)")
        print("     Please ensure the model is in the correct location")
    
    if os.path.exists("models"):
        print("  ✅ Models directory found (models/)")
    else:
        print("  ⚠️  Models directory not found")
        print("     Mistral model will be downloaded on first run")
    
    # Run tests
    print("\n🧪 Running tests...")
    test_result = run_command(
        f"{sys.executable} -m pytest tests/test_prepgen.py -v --tb=short",
        "Running unit tests"
    )
    
    # Final summary
    print("\n" + "="*60)
    print("📊 SETUP SUMMARY")
    print("="*60)
    
    if success and test_result:
        print("✅ All setup steps completed successfully!")
        print("\n🚀 Next steps:")
        print("   1. Review and update .env file if needed")
        print("   2. Ensure AI models are in place")
        print("   3. Run the server: python main.py")
        print("   4. Test: curl http://localhost:8000/health")
    else:
        print("⚠️  Setup completed with some warnings")
        print("\n📝 Please check the errors above and resolve them")
        print("   You can re-run this script after fixing issues")
    
    print("\n📚 Documentation:")
    print("   - QUICKSTART.md      - Getting started guide")
    print("   - IMPROVEMENTS.md    - Detailed improvements")
    print("   - PROJECT_SUMMARY.md - Complete summary")
    print("="*60)
    
    return True


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Setup failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
