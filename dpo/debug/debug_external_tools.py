#!/usr/bin/env python3
"""
Debug script to verify external tool paths and functionality.
Comprehensive check of all evaluation tools used in the pipeline.
"""

from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def run_command(cmd: List[str], timeout: int = 30) -> Tuple[bool, str, str]:
    """Run a command and return success status, stdout, stderr."""
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except FileNotFoundError:
        return False, "", "Command not found"
    except Exception as e:
        return False, "", str(e)


def check_vienna_rna():
    """Check ViennaRNA tools availability."""
    print("\n" + "="*50)
    print("CHECKING VIENNA RNA TOOLS")
    print("="*50)
    
    vienna_tools = ['RNAfold', 'RNAeval', 'RNAheat', 'RNAplfold', 'RNAsubopt']
    
    all_good = True
    
    for tool in vienna_tools:
        success, stdout, stderr = run_command([tool, '--version'])
        if success:
            version = stdout.split('\n')[0] if stdout else 'Unknown version'
            print(f"✓ {tool}: {version}")
        else:
            print(f"✗ {tool}: {stderr or 'Not available'}")
            all_good = False
    
    # Test basic functionality
    if all_good:
        print(f"\nTesting basic ViennaRNA functionality...")
        test_seq = "GGGAAAUUUCCC"
        proc = subprocess.Popen(['RNAfold'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = proc.communicate(input=test_seq)
        success = proc.returncode == 0
        if success and "(" in stdout:
            print(f"✓ RNAfold functional test passed")
        else:
            print(f"✗ RNAfold functional test failed: {stderr}")
            all_good = False
    
    return all_good


def check_eternafold():
    """Check EternaFold availability."""
    print("\n" + "="*50)
    print("CHECKING ETERNAFOLD")
    print("="*50)
    
    eternafold_path = Path("/mnt/rna01/smh/projects/ribopo/tools/EternaFold")
    
    if not eternafold_path.exists():
        print(f"✗ EternaFold directory not found: {eternafold_path}")
        return False
    
    print(f"✓ EternaFold directory exists: {eternafold_path}")
    
    # Check contents
    try:
        contents = list(eternafold_path.iterdir())
        print(f"✓ Directory contents ({len(contents)} items):")
        for item in contents[:5]:
            print(f"  - {item.name}")
        if len(contents) > 5:
            print(f"  ... and {len(contents) - 5} more")
    except Exception as e:
        print(f"✗ Cannot list EternaFold directory: {e}")
        return False
    
    # Look for common EternaFold files
    key_files = ['eternafold.py', 'EternaFold.py', 'eternafold', 'EternaFold']
    found_executable = False
    
    for key_file in key_files:
        file_path = eternafold_path / key_file
        if file_path.exists():
            print(f"✓ Found potential executable: {key_file}")
            found_executable = True
            break
    
    if not found_executable:
        print(f"⚠ No obvious executable found. Check EternaFold installation.")
    
    return True


def check_rhofold():
    """Check RhoFold availability."""
    print("\n" + "="*50)
    print("CHECKING RHOFOLD")
    print("="*50)
    
    rhofold_path = Path("/mnt/rna01/smh/projects/ribopo/tools/rhofold")
    
    if not rhofold_path.exists():
        print(f"✗ RhoFold directory not found: {rhofold_path}")
        return False
    
    print(f"✓ RhoFold directory exists: {rhofold_path}")
    
    # Check for key RhoFold files based on the actual structure
    key_files_to_check = [
        'rf.py',           # Main RhoFold script
        '__init__.py',     # Python package
        'config.py',       # Configuration
        'model_20221010_params.pt',  # Model weights
        'model/__init__.py',  # Model package
        'utils/__init__.py',  # Utils package
        'relax/__init__.py'   # Relax package
    ]
    
    found_files = []
    for key_file in key_files_to_check:
        file_path = rhofold_path / key_file
        if file_path.exists():
            found_files.append(key_file)
            print(f"✓ Found: {key_file}")
    
    print(f"✓ Found {len(found_files)}/{len(key_files_to_check)} key RhoFold files")
    
    # Check key directories
    key_dirs = ['model', 'utils', 'relax']
    for dir_name in key_dirs:
        dir_path = rhofold_path / dir_name
        if dir_path.exists() and dir_path.is_dir():
            print(f"✓ Found directory: {dir_name}/")
        else:
            print(f"✗ Missing directory: {dir_name}/")
    
    # Check if RhoFold can be imported by adding to path
    import sys
    if str(rhofold_path.parent) not in sys.path:
        sys.path.insert(0, str(rhofold_path.parent))
    
    try:
        import rhofold
        print(f"✓ RhoFold Python package is importable")
        
        # Try to access main components
        try:
            from rhofold.rf import RhoFold
            print(f"✓ RhoFold main class accessible")
            return True
        except ImportError as e:
            print(f"⚠ RhoFold class not accessible: {e}")
            print(f"  (Structure prediction may still work)")
            
    except ImportError as e:
        print(f"⚠ RhoFold not importable: {e}")
        print(f"  (This may require specific environment setup)")
    
    # If we found the key files, consider it available
    if len(found_files) >= 4:
        print(f"✓ RhoFold appears to be properly installed")
        return True
    else:
        print(f"✗ RhoFold installation appears incomplete")
        return False


def check_usalign():
    """Check USalign availability."""
    print("\n" + "="*50)
    print("CHECKING USALIGN")
    print("="*50)
    
    usalign_path = Path("/mnt/rna01/smh/projects/ribopo/tools/USalign")
    
    if not usalign_path.exists():
        print(f"✗ USalign directory not found: {usalign_path}")
        return False
    
    print(f"✓ USalign directory exists: {usalign_path}")
    
    # Check for executable
    executable_path = usalign_path / "USalign"
    if executable_path.exists():
        print(f"✓ USalign executable found: {executable_path}")
        
        # Check if executable
        if os.access(executable_path, os.X_OK):
            print(f"✓ USalign is executable")
            
            # Test functionality
            success, stdout, stderr = run_command([str(executable_path)])
            if "USalign" in (stdout + stderr):
                print(f"✓ USalign functional test passed")
                return True
            else:
                print(f"⚠ USalign may not work properly")
                print(f"  stdout: {stdout[:100]}...")
                print(f"  stderr: {stderr[:100]}...")
        else:
            print(f"✗ USalign file exists but is not executable")
    else:
        print(f"✗ USalign executable not found: {executable_path}")
        
        # List directory contents
        try:
            contents = list(usalign_path.iterdir())
            print(f"Directory contents ({len(contents)} items):")
            for item in contents[:10]:
                print(f"  - {item.name}")
        except Exception as e:
            print(f"✗ Cannot list directory: {e}")
    
    return False


def check_rna_assessment():
    """Check RNA_assessment tools."""
    print("\n" + "="*50)
    print("CHECKING RNA_ASSESSMENT")
    print("="*50)
    
    rna_assess_path = Path("/mnt/rna01/smh/projects/ribopo/tools/RNA_assessment")
    
    if not rna_assess_path.exists():
        print(f"✗ RNA_assessment directory not found: {rna_assess_path}")
        return False
    
    print(f"✓ RNA_assessment directory exists: {rna_assess_path}")
    
    # Look for key executables/scripts
    key_items = [
        'bin/',
        'scripts/',
        'src/',
        'RNA_assessment.py',
        'lddt',
        'inf'
    ]
    
    found_items = []
    for item_name in key_items:
        item_path = rna_assess_path / item_name
        if item_path.exists():
            found_items.append(item_name)
            item_type = "directory" if item_path.is_dir() else "file"
            print(f"✓ Found {item_type}: {item_name}")
    
    if not found_items:
        print(f"⚠ No expected items found. Contents:")
        try:
            for item in rna_assess_path.iterdir():
                item_type = "dir" if item.is_dir() else "file"
                print(f"  - {item.name} ({item_type})")
        except Exception as e:
            print(f"✗ Cannot list directory: {e}")
    
    return len(found_items) > 0


def check_phenix():
    """Check Phenix tools and wrapper script."""
    print("\n" + "="*50)
    print("CHECKING PHENIX")
    print("="*50)
    
    phenix_path = Path("/mnt/rna01/smh/projects/ribopo/tools/phenix-1.21.2-5419")
    
    if not phenix_path.exists():
        print(f"✗ Phenix directory not found: {phenix_path}")
        return False
    
    print(f"✓ Phenix directory exists: {phenix_path}")
    
    # Check for phenix_env.sh (main environment script)
    env_script = phenix_path / "phenix_env.sh"
    if env_script.exists():
        print(f"✓ Phenix environment script found: {env_script}")
    else:
        print(f"✗ Phenix environment script not found: {env_script}")
        return False
    
    # Check for bin directory and key executables (try build/bin first)
    bin_paths = [phenix_path / "build" / "bin", phenix_path / "bin"]
    bin_path = None
    
    for potential_bin in bin_paths:
        if potential_bin.exists():
            bin_path = potential_bin
            print(f"✓ Phenix bin directory found: {bin_path}")
            break
    
    if bin_path is None:
        print(f"✗ Phenix bin directory not found in {bin_paths}")
        return False
        
    # Look for clash score tool
    key_tools = ['phenix.clashscore', 'clashscore', 'phenix.python']
    found_tools = []
    
    for tool in key_tools:
        tool_path = bin_path / tool
        if tool_path.exists():
            found_tools.append(tool)
            print(f"✓ Found Phenix tool: {tool}")
    
    if not found_tools:
        print(f"⚠ No expected Phenix tools found in bin/")
        # List some files to help debug
        try:
            bin_files = list(bin_path.glob("*clash*"))
            if bin_files:
                print(f"  Found clash-related files: {[f.name for f in bin_files[:3]]}")
        except:
            pass
    
    # Check wrapper script (most important)
    wrapper_path = Path("/mnt/rna01/smh/projects/ribopo/tools/run_phenix.sh")
    if wrapper_path.exists():
        print(f"✓ Found wrapper script: {wrapper_path}")
        
        # Check if wrapper is executable
        if os.access(wrapper_path, os.X_OK):
            print(f"✓ Wrapper script is executable")
            
            # Test wrapper functionality
            print(f"Testing wrapper script...")
            try:
                success, stdout, stderr = run_command([str(wrapper_path), 'phenix.clashscore', '--version'], timeout=15)
                if success or 'phenix.clashscore' in (stdout + stderr):
                    print(f"✓ Wrapper script functional test passed")
                    return True
                else:
                    print(f"⚠ Wrapper test inconclusive: {stderr[:100]}")
            except Exception as e:
                print(f"⚠ Wrapper test failed: {e}")
        else:
            print(f"✗ Wrapper script exists but is not executable")
            print(f"  Run: chmod +x {wrapper_path}")
    else:
        print(f"✗ Wrapper script not found: {wrapper_path}")
        print(f"  Create wrapper script as described in documentation")
    
    return False


def check_x3dna():
    """Check x3dna tools."""
    print("\n" + "="*50)
    print("CHECKING X3DNA")
    print("="*50)
    
    x3dna_path = Path("/mnt/rna01/smh/projects/ribopo/tools/x3dna-v2.4")
    
    if not x3dna_path.exists():
        print(f"✗ x3dna directory not found: {x3dna_path}")
        return False
    
    print(f"✓ x3dna directory exists: {x3dna_path}")
    
    # Check for bin directory
    bin_path = x3dna_path / "bin"
    if bin_path.exists():
        print(f"✓ x3dna bin directory found: {bin_path}")
        
        # Check for key executables
        try:
            executables = [f for f in bin_path.iterdir() if f.is_file()]
            print(f"✓ Found {len(executables)} executables in bin/")
            
            # Show first few
            for exe in executables[:5]:
                print(f"  - {exe.name}")
            if len(executables) > 5:
                print(f"  ... and {len(executables) - 5} more")
            
            return True
        except Exception as e:
            print(f"✗ Error checking executables: {e}")
    
    else:
        print(f"✗ x3dna bin directory not found: {bin_path}")
    
    return False


def check_python_packages():
    """Check required Python packages."""
    print("\n" + "="*50)
    print("CHECKING PYTHON PACKAGES")
    print("="*50)
    
    required_packages = [
        'numpy',
        'pandas', 
        'torch',
        'torch_geometric',
        'Bio',  # BioPython
        'sklearn',
        'tqdm',
        'wandb'
    ]
    
    all_good = True
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - NOT AVAILABLE")
            all_good = False
    
    return all_good


def generate_report(results: Dict[str, bool]):
    """Generate summary report."""
    print("\n" + "="*60)
    print("TOOL AVAILABILITY SUMMARY REPORT")
    print("="*60)
    
    working_tools = []
    broken_tools = []
    
    for tool, status in results.items():
        if status:
            working_tools.append(tool)
        else:
            broken_tools.append(tool)
    
    print(f"\n✓ WORKING TOOLS ({len(working_tools)}):")
    for tool in working_tools:
        print(f"  - {tool}")
    
    if broken_tools:
        print(f"\n✗ BROKEN/MISSING TOOLS ({len(broken_tools)}):")
        for tool in broken_tools:
            print(f"  - {tool}")
    
    print(f"\nOVERALL STATUS:")
    if len(broken_tools) == 0:
        print(f"🎉 All tools are working! Ready for comprehensive evaluation.")
    elif len(working_tools) >= len(broken_tools):
        print(f"⚠ Most tools working. Some metrics may be unavailable.")
    else:
        print(f"❌ Many tools missing. Comprehensive evaluation may fail.")
    
    print(f"\nRECOMMENDATIONS:")
    if 'ViennaRNA' in broken_tools:
        print(f"- Install ViennaRNA: conda install -c bioconda viennarna")
    if 'Python packages' in broken_tools:
        print(f"- Install missing Python packages: pip install <package_name>")
    if any(tool in broken_tools for tool in ['EternaFold', 'RhoFold', 'USalign']):
        print(f"- Check external tool installations in tools/ directory")


def main():
    print("🔧 EXTERNAL TOOLS AVAILABILITY CHECKER")
    print("="*60)
    
    results = {}
    
    # Check each category of tools
    results['ViennaRNA'] = check_vienna_rna()
    results['Python packages'] = check_python_packages()
    results['EternaFold'] = check_eternafold()
    results['RhoFold'] = check_rhofold()
    results['USalign'] = check_usalign()
    results['RNA_assessment'] = check_rna_assessment()
    results['Phenix'] = check_phenix()
    results['x3dna'] = check_x3dna()
    
    # Generate summary report
    generate_report(results)


if __name__ == "__main__":
    main()