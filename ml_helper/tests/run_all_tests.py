"""
Script to run all tests for the ml_helper package.
"""

import unittest

if __name__ == '__main__':
    # Automatically discover all test modules and run them
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('.')
    
    # Run the tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    test_runner.run(test_suite)
    
    print("All tests complete!")
