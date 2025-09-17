# dpo/debug/inspect_package.py
import os
import sys
import site

print("--- 🔬 Package Inspection Script ---")

try:
    # First, try to import the base package
    import RNA_normalizer
    print("\n✅ SUCCESS: Base 'RNA_normalizer' package imported.")

    # Get the path to the package's main directory
    package_path = os.path.dirname(RNA_normalizer.__file__)
    print(f"   Package is installed at: {package_path}")

    # --- INSPECTION ---
    # 1. See what Python thinks is inside the package
    print("\n[INFO] Python sees these contents in the package (dir()):")
    package_contents_python = [item for item in dir(RNA_normalizer) if not item.startswith('__')]
    print(f"   {package_contents_python}")

    # 2. See what files are actually on the disk
    print("\n[INFO] These files are actually on disk in that directory (os.listdir()):")
    package_contents_disk = os.listdir(package_path)
    print(f"   {package_contents_disk}")

    # --- ANALYSIS ---
    print("\n[INFO] Analysis:")
    if 'INF.py' in package_contents_disk and 'lddt.py' in package_contents_disk:
        print("   ✅ The submodule files (INF.py, lddt.py) seem to be correctly installed.")
    else:
        print("   ❌ CRITICAL: The submodule files (e.g., INF.py, lddt.py) are MISSING from the installation directory.")
        print("      This indicates a broken installation is the root cause of the ModuleNotFoundError.")

except ImportError as e:
    print(f"\n❌ FAILURE: Could not even import the base 'RNA_normalizer' package.")
    print(f"   Error: {e}")

print("\n--- Inspection Complete ---")
