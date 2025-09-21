# MultiRound Testing Framework

Comprehensive testing framework for the multi-round RiboPO implementation.

## Structure

```
multiround/debug/
├── unit/                    # Unit tests for individual components
│   ├── test_trainer.py         # MultiRoundDPOTrainer tests
│   ├── test_pair_provider.py   # MultiRoundPairProvider tests
│   ├── test_evaluator.py       # MultiRoundEvaluator tests
│   ├── test_wandb_manager.py   # MultiRoundWandBManager tests
│   └── test_ema.py             # EMA module tests
├── integration/             # Integration tests
│   ├── test_full_pipeline.py   # End-to-end pipeline tests
│   └── test_config_integration.py # Configuration system tests
├── validation/              # Data and metrics validation
│   ├── test_data_validation.py   # Data format/integrity tests
│   └── test_metrics_validation.py # Metrics correctness tests
├── utils/                   # Testing utilities
│   ├── test_helpers.py         # Common testing utilities
│   └── mock_data.py            # Mock data generation
├── run_tests.py            # Master test runner
└── README.md               # This file
```

## Quick Start

### Run All Tests
```bash
cd multiround/debug
python run_tests.py
```

### Run Specific Test Categories
```bash
# Unit tests only
python run_tests.py --category unit

# Integration tests only
python run_tests.py --category integration

# Validation tests only
python run_tests.py --category validation
```

### Run Fast Tests (Skip Slow Integration Tests)
```bash
python run_tests.py --fast
```

### List Available Tests
```bash
python run_tests.py --list
```

## Test Categories

### Unit Tests (`unit/`)

Test individual components in isolation with mocked dependencies.

- **test_trainer.py**: MultiRoundDPOTrainer initialization, round management, integrations
- **test_pair_provider.py**: Static/dynamic pair provision, validation, margin mapping  
- **test_evaluator.py**: Evaluation logic, temperature switching, pass@k calculations
- **test_wandb_manager.py**: WandB integration, logging, artifact management
- **test_ema.py**: Exponential Moving Average functionality

### Integration Tests (`integration/`)

Test interactions between components and full pipeline functionality.

- **test_full_pipeline.py**: End-to-end training pipeline, component integration
- **test_config_integration.py**: Configuration loading, inheritance, validation

### Validation Tests (`validation/`)

Test data integrity and metrics correctness.

- **test_data_validation.py**: Preference pairs format, test dataset structure, path validation
- **test_metrics_validation.py**: Evaluation metrics ranges, computation logic, thresholds

## Testing Utilities

### TestConfig (`utils/test_helpers.py`)
Creates minimal working configurations for testing:
```python
from utils.test_helpers import TestConfig
config = TestConfig.create_minimal_config()
```

### MockData (`utils/mock_data.py`)
Generates realistic mock RNA data:
```python
from utils.mock_data import setup_complete_mock_environment
mock_env = setup_complete_mock_environment(temp_dir, n_pairs=20)
```

### TempDirectory (`utils/test_helpers.py`)
Context manager for temporary directories:
```python
from utils.test_helpers import TempDirectory
with TempDirectory() as temp_dir:
    # Use temp_dir for testing
```

## Test Execution Examples

### Run Specific Test File
```bash
python -m unittest unit.test_trainer -v
```

### Run Specific Test Class
```bash
python -m unittest unit.test_trainer.TestMultiRoundDPOTrainer -v
```

### Run Specific Test Method
```bash
python -m unittest unit.test_trainer.TestMultiRoundDPOTrainer.test_trainer_initialization -v
```

### Run with Different Verbosity
```bash
# Minimal output
python run_tests.py --verbosity 0

# Normal output  
python run_tests.py --verbosity 1

# Detailed output (default)
python run_tests.py --verbosity 2
```

## Adding New Tests

### Unit Test Template
```python
import unittest
from unittest.mock import Mock, patch
from utils.test_helpers import TestConfig, TempDirectory, setup_test_environment

class TestNewComponent(unittest.TestCase):
    def setUp(self):
        setup_test_environment()
        self.config = TestConfig.create_minimal_config()
    
    def test_basic_functionality(self):
        # Test implementation
        pass
```

### Integration Test Template
```python
import unittest
from utils.test_helpers import TestConfig, TempDirectory, setup_test_environment
from utils.mock_data import setup_complete_mock_environment

class TestNewIntegration(unittest.TestCase):
    def setUp(self):
        setup_test_environment()
        self.config = TestConfig.create_minimal_config()
    
    def test_component_integration(self):
        with TempDirectory() as temp_dir:
            mock_env = setup_complete_mock_environment(temp_dir)
            # Test implementation
```

## Mock Data Generation

The framework provides comprehensive mock data generation:

### RNA Sequences
```python
from utils.mock_data import MockRNA
seq = MockRNA.random_sequence(length=20)  # Random RNA sequence
coords = MockRNA.create_coordinates(length=20)  # 3D coordinates
ss = MockRNA.create_secondary_structure(length=20)  # Dot-bracket structure
```

### Preference Pairs
```python
from utils.mock_data import MockDataset
dataset = MockDataset(temp_dir)
pairs_dir = dataset.create_preference_pairs(n_pairs=100, margin_type="25")
```

### Test Datasets
```python
test_file = dataset.create_test_dataset(n_structures=20)
```

### Model Checkpoints
```python
checkpoint_file = dataset.create_model_checkpoint(round_num=2)
```

## CI/CD Integration

### GitHub Actions Example
```yaml
- name: Run Tests
  run: |
    cd multiround/debug
    python run_tests.py --verbosity 1
```

### Pre-commit Hook
```bash
#!/bin/bash
cd multiround/debug
python run_tests.py --fast --verbosity 0
```

## Performance Notes

- **Unit tests**: ~30 seconds (mocked dependencies)
- **Integration tests**: ~2-5 minutes (requires file I/O, mock training)
- **Validation tests**: ~1 minute (data validation)
- **Full test suite**: ~3-6 minutes

Use `--fast` flag to skip slow integration tests during development.

## Debugging Failed Tests

### Common Issues

1. **Import Errors**: Ensure project root is in Python path
2. **Missing Dependencies**: Check that all required packages are installed
3. **File Permissions**: Ensure write access to temporary directories
4. **Mock Setup**: Verify mocks are properly configured for the test scope

### Debug Mode
```bash
# Run with maximum verbosity and don't capture output
python -m unittest unit.test_trainer.TestMultiRoundDPOTrainer.test_trainer_initialization -v -s
```

### Debugging Individual Tests
```python
import unittest
import sys

# Add to test method for debugging
import pdb; pdb.set_trace()
```

## Test Coverage

To generate test coverage reports:

```bash
pip install coverage
cd multiround/debug
coverage run --source=.. -m unittest discover
coverage report
coverage html  # Generate HTML report
```

## Contributing

1. Write tests for any new functionality
2. Ensure all tests pass before submitting PRs
3. Add integration tests for new component interactions
4. Update mock data generators if data formats change
5. Document any new testing utilities

## Troubleshooting

### Environment Issues
- Set `CUDA_VISIBLE_DEVICES=''` to force CPU mode
- Set `WANDB_MODE=disabled` to disable WandB during testing

### Memory Issues
- Use `--fast` flag to skip memory-intensive tests
- Reduce mock data sizes in test configurations

### Network Issues
- Tests should work offline (no external dependencies)
- WandB is mocked/disabled in test environment