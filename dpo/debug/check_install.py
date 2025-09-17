# dpo/debug/check_install.py
import sys
import os

print("--- 🔬 DIAGNOSTIC SCRIPT RUNNING ---")

# 1. Verify the Python Environment
print(f"\n[INFO] Python executable being used:\n{sys.executable}\n")

# Check if we are in the 'grnade' conda/mamba environment
if 'grnade' in sys.executable:
    print("✅ Correct 'grnade' environment detected.")
else:
    print("⚠️ WARNING: Not running in the 'grnade' environment.")

# 2. Check the Python Path
print("\n[INFO] Python will search for modules in these directories (sys.path):")
for path in sys.path:
    print(f"  - {path}")

# 3. Attempt to import the RNA_normalizer package
print("\n[INFO] Attempting to import RNA_normalizer...")
try:
    import RNA_normalizer
    print("\n✅ SUCCESS: 'RNA_normalizer' package was imported successfully!")
    print(f"   Package location: {RNA_normalizer.__file__}")
except ImportError as e:
    print(f"\n❌ FAILURE: Could not import 'RNA_normalizer'.")
    print(f"   Error: {e}")

print("\n--- DIAGNOSTICS COMPLETE ---")
