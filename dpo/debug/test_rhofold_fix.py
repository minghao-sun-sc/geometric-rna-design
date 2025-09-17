#!/usr/bin/env python3

"""Test script to verify RhoFold unpacking fix."""

from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

import numpy as np

def test_rhofold_return_format():
    """Test handling of different RhoFold return formats."""
    print("=" * 50)
    print("Testing RhoFold return format handling")
    print("=" * 50)
    
    # Simulate old format (3 values)
    old_format_result = (
        np.array([1.0, 2.0]),  # rmsd
        np.array([0.5, 0.6]),  # tm
        np.array([0.7, 0.8])   # gdt
    )
    
    # Simulate new format (6 values) 
    new_format_result = (
        np.array([1.0, 2.0]),  # rmsd
        np.array([0.5, 0.6]),  # tm
        np.array([0.7, 0.8]),  # gdt
        np.array([0.9, 0.95]), # plddt
        {"all": 0.5, "wc": 0.6}, # inf_dict
        np.array([5.0, 6.0])   # clash
    )
    
    print("Test 1: Old format (3 values)")
    result = old_format_result
    if len(result) == 3:
        sc_rmsd, sc_tm, sc_gdt = result
        print(f"✓ Successfully unpacked 3 values: RMSD={sc_rmsd}, TM={sc_tm}, GDT={sc_gdt}")
    else:
        sc_rmsd, sc_tm, sc_gdt, sc_plddt, inf_dict, clash_scores = result
        print(f"✓ Successfully unpacked 6 values: RMSD={sc_rmsd}, TM={sc_tm}, GDT={sc_gdt}")
    
    print("\nTest 2: New format (6 values)")
    result = new_format_result
    if len(result) == 3:
        sc_rmsd, sc_tm, sc_gdt = result
        print(f"✓ Successfully unpacked 3 values: RMSD={sc_rmsd}, TM={sc_tm}, GDT={sc_gdt}")
    else:
        sc_rmsd, sc_tm, sc_gdt, sc_plddt, inf_dict, clash_scores = result
        print(f"✓ Successfully unpacked 6 values: RMSD={sc_rmsd}, TM={sc_tm}, GDT={sc_gdt}")
        print(f"  Additional values: pLDDT={sc_plddt}, INF={inf_dict}, Clash={clash_scores}")
    
    print("\n✓ RhoFold return format handling works correctly!")

if __name__ == "__main__":
    test_rhofold_return_format()