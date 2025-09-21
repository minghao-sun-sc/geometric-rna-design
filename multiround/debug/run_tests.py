#!/usr/bin/env python3
# multiround/debug/run_tests.py
"""Master test runner for multiround testing framework."""

import os
import sys
import unittest
import argparse
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dpo.env_bootstrap import bootstrap_env
bootstrap_env()

# Import testing utilities
sys.path.insert(0, str(Path(__file__).parent))
from utils.test_helpers import setup_test_environment, cleanup_test_environment


def discover_tests(test_dir, pattern="test_*.py"):
    """Discover tests in a directory."""
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir, pattern=pattern)
    return suite


def run_test_suite(suite, verbosity=2):
    """Run a test suite and return results."""
    runner = unittest.TextTestRunner(verbosity=verbosity, stream=sys.stdout, buffer=True)
    result = runner.run(suite)
    return result


def print_test_summary(results_dict):
    """Print a summary of test results."""
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    total_tests = 0
    total_failures = 0
    total_errors = 0
    total_skipped = 0
    
    for category, result in results_dict.items():
        tests_run = result.testsRun
        failures = len(result.failures)
        errors = len(result.errors)
        skipped = len(result.skipped)
        
        total_tests += tests_run
        total_failures += failures
        total_errors += errors
        total_skipped += skipped
        
        status = "PASS" if (failures == 0 and errors == 0) else "FAIL"
        
        print(f"{category:20} | {tests_run:3d} tests | {failures:2d} failures | {errors:2d} errors | {skipped:2d} skipped | {status}")
    
    print("-"*70)
    print(f"{'TOTAL':20} | {total_tests:3d} tests | {total_failures:2d} failures | {total_errors:2d} errors | {total_skipped:2d} skipped")
    
    overall_status = "PASS" if (total_failures == 0 and total_errors == 0) else "FAIL"
    print(f"\nOVERALL STATUS: {overall_status}")
    
    if total_failures > 0 or total_errors > 0:
        print(f"\n⚠️  {total_failures + total_errors} test(s) failed. See details above.")
        return False
    else:
        print(f"\n✅ All {total_tests} tests passed!")
        return True


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run multiround testing framework")
    parser.add_argument(
        "--category", 
        choices=["unit", "integration", "validation", "all"],
        default="all",
        help="Test category to run (default: all)"
    )
    parser.add_argument(
        "--pattern",
        default="test_*.py",
        help="Test file pattern (default: test_*.py)"
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        choices=[0, 1, 2],
        default=2,
        help="Test verbosity level (default: 2)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available tests without running them"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run only fast tests (skip slow integration tests)"
    )
    
    args = parser.parse_args()
    
    # Set up test environment
    print("Setting up test environment...")
    setup_test_environment()
    
    # Define test directories
    debug_dir = Path(__file__).parent
    test_dirs = {
        "unit": debug_dir / "unit",
        "integration": debug_dir / "integration", 
        "validation": debug_dir / "validation"
    }
    
    # Filter test directories based on category
    if args.category == "all":
        selected_dirs = test_dirs
    else:
        selected_dirs = {args.category: test_dirs[args.category]}
    
    # List tests if requested
    if args.list:
        print("\nAvailable tests:")
        print("-" * 50)
        for category, test_dir in selected_dirs.items():
            if test_dir.exists():
                print(f"\n{category.upper()} TESTS:")
                suite = discover_tests(str(test_dir), args.pattern)
                for test_group in suite:
                    for test_case in test_group:
                        if hasattr(test_case, '_tests'):
                            for test in test_case._tests:
                                print(f"  {test._testMethodName} ({test.__class__.__name__})")
                        else:
                            print(f"  {test_case._testMethodName} ({test_case.__class__.__name__})")
        return
    
    # Run tests
    print(f"\nRunning {args.category} tests...")
    print("=" * 70)
    
    start_time = time.time()
    results = {}
    
    for category, test_dir in selected_dirs.items():
        if not test_dir.exists():
            print(f"⚠️  Test directory does not exist: {test_dir}")
            continue
        
        print(f"\n🔍 Running {category} tests...")
        
        # Skip slow tests if --fast is specified
        if args.fast and category == "integration":
            print("⏩ Skipping integration tests (--fast mode)")
            continue
        
        # Discover and run tests
        suite = discover_tests(str(test_dir), args.pattern)
        
        if suite.countTestCases() == 0:
            print(f"⚠️  No tests found in {test_dir} matching pattern '{args.pattern}'")
            continue
        
        result = run_test_suite(suite, args.verbosity)
        results[category] = result
    
    end_time = time.time()
    
    # Print summary
    if results:
        success = print_test_summary(results)
        
        print(f"\nTest execution time: {end_time - start_time:.2f} seconds")
        
        # Return appropriate exit code
        sys.exit(0 if success else 1)
    else:
        print("⚠️  No tests were run.")
        sys.exit(1)


def run_unit_tests():
    """Convenience function to run only unit tests."""
    debug_dir = Path(__file__).parent
    unit_dir = debug_dir / "unit"
    
    setup_test_environment()
    
    print("Running unit tests...")
    suite = discover_tests(str(unit_dir))
    result = run_test_suite(suite)
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    return success


def run_integration_tests():
    """Convenience function to run only integration tests."""
    debug_dir = Path(__file__).parent
    integration_dir = debug_dir / "integration"
    
    setup_test_environment()
    
    print("Running integration tests...")
    suite = discover_tests(str(integration_dir))
    result = run_test_suite(suite)
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    return success


def run_validation_tests():
    """Convenience function to run only validation tests."""
    debug_dir = Path(__file__).parent
    validation_dir = debug_dir / "validation"
    
    setup_test_environment()
    
    print("Running validation tests...")
    suite = discover_tests(str(validation_dir))
    result = run_test_suite(suite)
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    return success


def run_smoke_tests():
    """Run a subset of critical tests for quick validation."""
    debug_dir = Path(__file__).parent
    
    setup_test_environment()
    
    print("Running smoke tests...")
    
    # Run a few key tests from each category
    smoke_tests = [
        "unit/test_trainer.py::TestMultiRoundDPOTrainer::test_trainer_initialization",
        "unit/test_pair_provider.py::TestMultiRoundPairProvider::test_static_pair_provider",
        "validation/test_data_validation.py::TestDataFormatValidation::test_preference_pairs_format_validation"
    ]
    
    # For now, just run all unit tests as smoke test
    # In a more advanced setup, we'd run specific test methods
    unit_dir = debug_dir / "unit"
    suite = discover_tests(str(unit_dir))
    result = run_test_suite(suite, verbosity=1)
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nSmoke test {'PASSED' if success else 'FAILED'}")
    return success


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Test runner failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean up test environment
        cleanup_test_environment()