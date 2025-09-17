#!/usr/bin/env python3

# Simple Vienna test
def test_vienna():
    try:
        import RNA
        print(f"✅ ViennaRNA import successful: {RNA.__version__}")
        
        # Simple test
        seq = "GCGCGCAAAGCGCGC"
        fc = RNA.fold_compound(seq)
        db, mfe = fc.mfe()
        print(f"✅ Basic Vienna fold successful: {mfe:.2f} kcal/mol, {db}")
        
        return True
    except Exception as e:
        print(f"❌ Vienna test failed: {e}")
        return False

if __name__ == "__main__":
    test_vienna()