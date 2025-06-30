#!/usr/bin/env python3
"""
Minimal End-to-End Pipeline Test
================================

This script performs a minimal test of the enhanced pipeline using
simulated data that matches the real pipeline structure.
"""

import os
import sys
import tempfile
import traceback
from pathlib import Path

# Add FuGEP to path
sys.path.insert(0, '/home/sdodl001/FuGEP')

def test_minimal_pipeline():
    """Test the pipeline with minimal simulated data"""
    print("=" * 50)
    print("Minimal End-to-End Pipeline Test")
    print("=" * 50)
    
    try:
        from fugep.predict.ana_hdl.handler_factory import create_mean_gve_handler, create_pval_handler
        
        # Create temporary directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Using temporary directory: {temp_dir}")
            
            # Test 1: Create handlers
            print("\n1. Creating handlers...")
            mean_handler = create_mean_gve_handler(
                output_dir=temp_dir,
                output_format='tsv',  # Use TSV to avoid dependency issues
                data_type_config='memory_optimized',
                write_mem_limit=100  # Small limit for testing
            )
            print("✓ Mean GVE handler created")
            
            pval_handler = create_pval_handler(
                output_dir=temp_dir,
                output_format='tsv',
                data_type_config='memory_optimized',
                write_mem_limit=100
            )
            print("✓ Pval handler created")
            
            # Test 2: Simulate processing results
            print("\n2. Processing simulated results...")
            
            # Simulate variant data as it comes from the pipeline
            test_variants = [
                (['chr1', 100, 'A', 'T'], [0.1, 0.2, 0.3]),  # (variant_id, features)
                (['chr1', 200, 'G', 'C'], [0.4, 0.5, 0.6]),
                (['chr2', 300, 'T', 'A'], [0.7, 0.8, 0.9])
            ]
            
            for i, (variant_id, features) in enumerate(test_variants):
                # Process with mean_gve handler
                result = [variant_id, features]
                mean_handler.process_variant_result(result)
                
                # Process with pval handler (convert scores to p-values)
                pval_features = [f * 0.01 for f in features]  # Convert to p-values
                pval_result = [variant_id, pval_features]
                pval_handler.process_variant_result(pval_result)
                
                print(f"✓ Processed variant {i+1}: {variant_id}")
            
            # Test 3: Force chunk writing
            print("\n3. Writing output chunks...")
            mean_handler.finalize()
            pval_handler.finalize()
            print("✓ Handlers finalized")
            
            # Test 4: Check output files
            print("\n4. Checking output files...")
            output_files = list(Path(temp_dir).glob('*.tsv'))
            
            for output_file in output_files:
                with open(output_file, 'r') as f:
                    lines = f.readlines()
                print(f"✓ {output_file.name}: {len(lines)} lines (including header)")
                
                # Show first few lines
                if len(lines) > 0:
                    print(f"  Header: {lines[0].strip()}")
                if len(lines) > 1:
                    print(f"  First data line: {lines[1].strip()}")
            
            print(f"\n✓ Generated {len(output_files)} output files")
            
            # Test 5: Memory usage check
            print("\n5. Checking memory usage...")
            mean_memory = mean_handler.backend._calculate_memory_usage()
            pval_memory = pval_handler.backend._calculate_memory_usage()
            print(f"✓ Mean handler memory: {mean_memory} bytes")
            print(f"✓ Pval handler memory: {pval_memory} bytes")
            
            print("\n🎉 Minimal pipeline test completed successfully!")
            return True
            
    except Exception as e:
        print(f"\n✗ Pipeline test failed: {e}")
        traceback.print_exc()
        return False

def test_memory_limits():
    """Test memory limit functionality"""
    print("\n" + "=" * 50)
    print("Memory Limit Test")
    print("=" * 50)
    
    try:
        from fugep.predict.ana_hdl.handler_factory import create_mean_gve_handler
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create handler with very small memory limit
            handler = create_mean_gve_handler(
                output_dir=temp_dir,
                output_format='tsv',
                write_mem_limit=1  # 1 byte - should trigger immediate writing
            )
            
            print("Created handler with 1-byte memory limit")
            
            # Add one result - should trigger immediate chunk write
            variant_id = ['chr1', 100, 'A', 'T']
            features = [0.1, 0.2, 0.3]
            result = [variant_id, features]
            
            print("Adding result...")
            handler.process_variant_result(result)
            
            # Check if chunk was written
            output_files = list(Path(temp_dir).glob('*.tsv'))
            if output_files:
                print(f"✓ Chunk written immediately: {output_files[0].name}")
                
                # Check content
                with open(output_files[0], 'r') as f:
                    content = f.read()
                    print(f"✓ File content:\n{content}")
                return True
            else:
                print("✗ No chunk written")
                return False
                
    except Exception as e:
        print(f"✗ Memory limit test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    success1 = test_minimal_pipeline()
    success2 = test_memory_limits()
    
    if success1 and success2:
        print("\n🎉 All tests passed!")
        return True
    else:
        print("\n⚠️  Some tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
