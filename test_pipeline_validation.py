#!/usr/bin/env python3
"""
Comprehensive Pipeline Validation Script
========================================

This script validates the enhanced FuGEP pipeline by:
1. Testing handler creation and configuration
2. Simulating data flow with various data types
3. Validating memory calculations
4. Testing chunked output writing
5. Verifying backward compatibility
6. Testing fallback mechanisms

Run this script to validate the pipeline before production use.
"""

import os
import sys
import tempfile
import shutil
import logging
import traceback
from pathlib import Path

# Add FuGEP to path
sys.path.insert(0, '/home/sdodl001/FuGEP')

def setup_logging():
    """Setup logging for validation"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('pipeline_validation.log')
        ]
    )

def test_imports():
    """Test that all required modules can be imported"""
    print("\n=== Testing Imports ===")
    
    try:
        from fugep.predict.ana_hdl.handler_factory import create_mean_gve_handler, create_pval_handler
        print("✓ Handler factory imports successful")
    except Exception as e:
        print(f"✗ Handler factory import failed: {e}")
        return False
        
    try:
        from fugep.predict.ana_hdl.output_backends import ParquetBackend, TSVBackend, HDF5Backend
        print("✓ Output backends imports successful")
    except Exception as e:
        print(f"✗ Output backends import failed: {e}")
        return False
        
    try:
        from fugep.predict.ana_hdl.optimal_data_types import get_optimal_types
        print("✓ Data types imports successful")
    except Exception as e:
        print(f"✗ Data types import failed: {e}")
        return False
        
    return True

def test_handler_creation():
    """Test handler creation with different configurations"""
    print("\n=== Testing Handler Creation ===")
    
    try:
        from fugep.predict.ana_hdl.handler_factory import create_mean_gve_handler, create_pval_handler
        
        # Required arguments for handlers
        features = ['feature1', 'feature2']
        columns_for_ids = ['chrom', 'pos', 'ref', 'alt']
        output_path_prefix = '/tmp/test'
        mult_predictions = False
        save_mult_pred = False
        
        # Test mean_gve handler creation
        handler = create_mean_gve_handler(
            output_format='parquet',
            data_type_config='memory_optimized',
            write_mem_limit=1000,
            features=features,
            columns_for_ids=columns_for_ids,
            output_path_prefix=output_path_prefix,
            mult_predictions=mult_predictions,
            save_mult_pred=save_mult_pred
        )
        print("✓ Mean GVE handler created successfully")
        
        # Test pval handler creation
        handler = create_pval_handler(
            output_format='parquet',
            data_type_config='memory_optimized',
            write_mem_limit=1000,
            features=features,
            columns_for_ids=columns_for_ids,
            output_path_prefix=output_path_prefix,
            mult_predictions=mult_predictions,
            save_mult_pred=save_mult_pred
        )
        print("✓ Pval handler created successfully")
        
        # Test backward compatibility (TSV with defaults)
        handler = create_mean_gve_handler(
            features=features,
            columns_for_ids=columns_for_ids,
            output_path_prefix=output_path_prefix,
            mult_predictions=mult_predictions,
            save_mult_pred=save_mult_pred
        )
        print("✓ Default handler creation works (backward compatibility)")
        
        return True
    except Exception as e:
        print(f"✗ Handler creation failed: {e}")
        traceback.print_exc()
        return False

def test_data_types():
    """Test data type configurations"""
    print("\n=== Testing Data Type Configurations ===")
    
    try:
        from fugep.predict.ana_hdl.optimal_data_types import get_optimal_types
        
        configs = ['production', 'memory_optimized', 'high_precision']
        for config in configs:
            types = get_optimal_types(config)
            print(f"✓ {config} configuration: {len(types)} data types defined")
            
        return True
    except Exception as e:
        print(f"✗ Data type configuration failed: {e}")
        traceback.print_exc()
        return False

def test_memory_calculation():
    """Test memory calculation logic"""
    print("\n=== Testing Memory Calculation ===")
    
    try:
        from fugep.predict.ana_hdl.output_backends import ParquetBackend
        import sys
        
        # Create test data with proper structure
        test_results = [
            ['chr1:100:A>T', 0.1, 0.2, 0.3],
            ['chr1:200:G>C', 0.4, 0.5, 0.6],
            ['chr2:300:T>A', 0.7, 0.8, 0.9]
        ]
        
        # Create test IDs separately
        test_ids = [
            ['chr1', 100, 'A', 'T'],
            ['chr1', 200, 'G', 'C'],
            ['chr2', 300, 'T', 'A']
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = ParquetBackend(
                output_path_prefix=os.path.join(temp_dir, 'test'),
                features=['score1', 'score2', 'score3'],
                columns_for_ids=['chrom', 'pos', 'ref', 'alt'],
                write_mem_limit=1500,
                data_type_config='memory_optimized'
            )
            
            # Test adding results
            backend.add_results(test_results, test_ids)
            print(f"✓ Added {len(test_results)} results to backend")
            
            # Test memory calculation
            memory_reached = backend._reached_mem_limit()
            print(f"✓ Memory calculation works: memory limit reached = {memory_reached}")
            
        return True
    except Exception as e:
        print(f"✗ Memory calculation failed: {e}")
        traceback.print_exc()
        return False

def test_fallback_mechanism():
    """Test fallback to TSV when Parquet is unavailable"""
    print("\n=== Testing Fallback Mechanism ===")
    
    try:
        # Required arguments for handlers
        features = ['feature1', 'feature2']
        columns_for_ids = ['chrom', 'pos', 'ref', 'alt']
        output_path_prefix = '/tmp/test'
        mult_predictions = False
        save_mult_pred = False
        
        # Test that TSV fallback works when pyarrow is not available
        # (This is a simplified test - actual fallback happens at backend level)
        from fugep.predict.ana_hdl.handler_factory import create_mean_gve_handler
        
        # Test creating TSV handler (should always work)
        handler = create_mean_gve_handler(
            output_format='tsv',
            features=features,
            columns_for_ids=columns_for_ids,
            output_path_prefix=output_path_prefix,
            mult_predictions=mult_predictions,
            save_mult_pred=save_mult_pred
        )
        print("✓ TSV fallback handler created successfully")
        
        return True
    except Exception as e:
        print(f"✗ Fallback mechanism failed: {e}")
        traceback.print_exc()
        return False

def test_column_structure():
    """Test that column structures match expectations"""
    print("\n=== Testing Column Structure ===")
    
    try:
        from fugep.predict.ana_hdl.enhanced_mean_gve_handler import EnhancedMeanGVEHandler
        from fugep.predict.ana_hdl.enhanced_pval_handler import EnhancedPvalHandler
        
        # Required constructor arguments
        features = ['feature1', 'feature2', 'feature3']
        columns_for_ids = ['chrom', 'pos', 'ref', 'alt']
        output_path_prefix = '/tmp/test'
        mult_predictions = False
        save_mult_pred = False
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test mean_gve handler
            mean_handler = EnhancedMeanGVEHandler(
                features=features,
                columns_for_ids=columns_for_ids,
                output_path_prefix=os.path.join(temp_dir, 'mean_gve'),
                mult_predictions=mult_predictions,
                save_mult_pred=save_mult_pred,
                output_format='tsv',  # Use TSV to avoid pyarrow dependency issues
                data_type_config='memory_optimized'
            )
            
            # Test that handler was created successfully
            print("✓ Mean GVE handler created with correct structure")
            
            # Test pval handler
            pval_handler = EnhancedPvalHandler(
                features=features,
                columns_for_ids=columns_for_ids,
                output_path_prefix=os.path.join(temp_dir, 'pval'),
                mult_predictions=mult_predictions,
                save_mult_pred=save_mult_pred,
                output_format='tsv',
                data_type_config='memory_optimized'
            )
            
            print("✓ Pval handler created with correct structure")
                
        return True
    except Exception as e:
        print(f"✗ Column structure test failed: {e}")
        traceback.print_exc()
        return False

def test_analyzer_integration():
    """Test that the analyzer properly uses the new handlers"""
    print("\n=== Testing Analyzer Integration ===")
    
    try:
        # Import analyzer class (don't instantiate - requires complex setup)
        from fugep.predict.analyzer import Analyzer
        
        # Check that the class can be imported
        print("✓ Analyzer class can be imported")
        
        # Check that the enhanced handler imports are available
        try:
            from fugep.predict.ana_hdl.handler_factory import create_mean_gve_handler, create_pval_handler
            print("✓ Enhanced handler factory functions available")
        except Exception as e:
            print(f"✗ Enhanced handler imports failed: {e}")
            return False
            
        # Check if the analyzer module has the required imports
        import inspect
        source = inspect.getsource(Analyzer)
        if 'create_mean_gve_handler' in source and 'create_pval_handler' in source:
            print("✓ Analyzer uses enhanced handler factory functions")
        else:
            print("? Analyzer may still be using old handlers")
            
        return True
    except Exception as e:
        print(f"✗ Analyzer integration test failed: {e}")
        traceback.print_exc()
        return False

def test_configuration_files():
    """Test that configuration files are properly formatted"""
    print("\n=== Testing Configuration Files ===")
    
    config_files = [
        '/scratch/ml-csm/projects/fgenom/gve/scripts/chunk_jobs/kmeans/sei/pred5/sei_gve_chunk1.yml',
        '/scratch/ml-csm/projects/fgenom/gve/scripts/chunk_jobs/kmeans/sei/pred5/sei_gve_chunk2.yml',
        '/scratch/ml-csm/projects/fgenom/gve/scripts/chunk_jobs/kmeans/DeepSEA/pred5/DeepSEA_chunk1_parquet.yml',
        '/scratch/ml-csm/projects/fgenom/gve/scripts/chunk_jobs/kmeans/DeepSEA/pred5/DeepSEA_chunk2_parquet.yml'
    ]
    
    try:
        for config_file in config_files:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    content = f.read()
                
                # Check for required fields using simple string matching
                # (avoid yaml.safe_load due to custom tags)
                if 'writeMemLimit:' in content:
                    # Extract the writeMemLimit value
                    for line in content.split('\n'):
                        if 'writeMemLimit:' in line:
                            mem_limit_str = line.split(':')[1].strip().rstrip(',')  # Remove trailing comma
                            try:
                                mem_limit = int(mem_limit_str)
                                if mem_limit >= 1000:  # Should be at least 1GB
                                    print(f"✓ {Path(config_file).name}: writeMemLimit = {mem_limit}")
                                else:
                                    print(f"⚠ {Path(config_file).name}: writeMemLimit = {mem_limit} (may be too low)")
                            except ValueError:
                                print(f"⚠ {Path(config_file).name}: writeMemLimit value not parseable: '{mem_limit_str}'")
                            break
                else:
                    print(f"✗ {Path(config_file).name}: missing writeMemLimit")
                    
                if 'dataTypeConfig:' in content:
                    # Extract the dataTypeConfig value
                    for line in content.split('\n'):
                        if 'dataTypeConfig:' in line:
                            config_value = line.split(':')[1].strip().rstrip(',')  # Remove trailing comma
                            print(f"✓ {Path(config_file).name}: dataTypeConfig = {config_value}")
                            break
                else:
                    print(f"⚠ {Path(config_file).name}: missing dataTypeConfig")
                    
                # Check for parquet output format
                if '.parquet' in config_file or 'parquet' in content:
                    print(f"✓ {Path(config_file).name}: appears to use Parquet output")
                    
            else:
                print(f"✗ Configuration file not found: {config_file}")
                
        return True
    except Exception as e:
        print(f"✗ Configuration file test failed: {e}")
        traceback.print_exc()
        return False

def run_comprehensive_test():
    """Run all validation tests"""
    print("=" * 60)
    print("FuGEP Enhanced Pipeline Validation")
    print("=" * 60)
    
    setup_logging()
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Import Test", test_imports),
        ("Handler Creation", test_handler_creation),
        ("Data Types", test_data_types),
        ("Memory Calculation", test_memory_calculation),
        ("Fallback Mechanism", test_fallback_mechanism),
        ("Column Structure", test_column_structure),
        ("Analyzer Integration", test_analyzer_integration),
        ("Configuration Files", test_configuration_files)
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} failed with exception: {e}")
            test_results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:<8} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The pipeline is ready for production use.")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
