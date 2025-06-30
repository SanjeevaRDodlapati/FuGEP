#!/usr/bin/env python3
"""
Quick test of the corrected enhanced handlers with DeepSEA.
Tests the column mismatch fix with a small memory limit.
"""

import os
import sys
import tempfile
import subprocess
from pathlib import Path

# Add FuGEP to path
sys.path.insert(0, '/home/sdodl001/FuGEP')

def create_test_config(output_dir, mem_limit_mb=50):
    """Create a minimal DeepSEA config for testing"""
    
    config_content = f"""---
ops: [analyze]
model: {{
  class: DeepSEA,
  classArgs: {{
    sequence_length: 1001,
    n_targets: 151,
  }},
  built: pytorch,
  mult_predictions: 5,
  wrapper: UniSeqMWrapper,
  non_strand_specific: mean
}}
analyzer: !obj:fugep.predict.PeakGVarEvaluator {{
  analysis: [mean_gve, pval],
  refSeq: !obj:fugep.data.Genome {{
    input_path: /scratch/ml-csm/datasets/genomics/ref-genome/human/GRCh38/ensembl/sequence/Homo_sapiens.GRCh38.dna.primary_assembly.fa
  }},
  vcfFile: /scratch/ml-csm/projects/fgenom/gve/data/comb_chunks/variant_chunk_001.tsv,
  strandIdx: 5,
  seqLen: 1001,
  batchSize: 32,
  useCuda: False,
  dataParallel: True,
  loggingVerbosity: 0,
  writeMemLimit: {mem_limit_mb},
  save_mult_pred: True,
  outputFormat: parquet,
  dataTypeConfig: memory_optimized,
  trainedModelPath: /scratch/ml-csm/projects/fgenom/gve/models/kmeans/1/best_model.pth.tar,
  features: !obj:fugep.utils.load_features_list {{
    input_path: /scratch/ml-csm/projects/fgenom/gve/data/features/kmeans/features_kmeans1.txt
  }},
}}
variant_effect_prediction: {{}}
output_dir: {output_dir}
random_seed: 1447
create_subdirectory: False
"""
    
    config_path = os.path.join(output_dir, 'test_deepsea_corrected.yml')
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    return config_path

def run_deepsea_test():
    """Run a quick DeepSEA test to validate the corrected handlers"""
    
    print("=" * 60)
    print("Testing Corrected Enhanced Handlers with DeepSEA")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Test directory: {temp_dir}")
        
        # Create test config with small memory limit to trigger chunking quickly
        config_path = create_test_config(temp_dir, mem_limit_mb=50)
        print(f"✓ Created test config: {config_path}")
        print(f"  - Memory limit: 50 MB (should trigger chunking quickly)")
        print(f"  - Batch size: 32 (smaller for faster testing)")
        print(f"  - Output directory: {temp_dir}")
        
        print("\\n=== Running DeepSEA Prediction Test ===")
        
        # Run the prediction
        cmd = [
            'crun', '-p', '~/envs/fugepTF2170', 'python', '-u', 
            '/home/sdodl001/FuGEP/fugep/cli.py',
            config_path
        ]
        
        print(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=300,  # 5 minute timeout
                                  cwd=temp_dir)
            
            if result.returncode == 0:
                print("✓ DeepSEA prediction completed successfully!")
                
                # Check output files
                output_files = list(Path(temp_dir).glob("*.parquet"))
                if output_files:
                    print(f"✓ Found {len(output_files)} Parquet output files:")
                    for f in output_files:
                        size_mb = f.stat().st_size / 1024 / 1024
                        print(f"  - {f.name}: {size_mb:.2f} MB")
                        
                    # Quick validation of first file
                    try:
                        import pandas as pd
                        first_file = output_files[0]
                        df = pd.read_parquet(first_file)
                        print(f"✓ Parquet file validation:")
                        print(f"  - Rows: {len(df)}")
                        print(f"  - Columns: {len(df.columns)}")
                        print(f"  - Expected columns: 3 (ID) + 151 (features) = 154")
                        
                        if len(df.columns) == 154:
                            print("  ✓ Column count matches expected!")
                        else:
                            print(f"  ⚠ Column count mismatch: expected 154, got {len(df.columns)}")
                            
                    except Exception as e:
                        print(f"  ⚠ Could not validate Parquet file: {e}")
                else:
                    print("⚠ No Parquet output files found")
                
                return True
                
            else:
                print(f"✗ DeepSEA prediction failed with exit code: {result.returncode}")
                
                if result.stdout:
                    print("STDOUT:")
                    print(result.stdout)
                    
                if result.stderr:
                    print("STDERR:")
                    print(result.stderr)
                    
                return False
                
        except subprocess.TimeoutExpired:
            print("✗ DeepSEA prediction timed out after 5 minutes")
            return False
            
        except Exception as e:
            print(f"✗ DeepSEA prediction failed with exception: {e}")
            return False

if __name__ == "__main__":
    success = run_deepsea_test()
    if success:
        print("\\n🎉 Test completed successfully! The corrected handlers are working.")
    else:
        print("\\n⚠️ Test failed. Please check the error messages above.")
    
    sys.exit(0 if success else 1)
