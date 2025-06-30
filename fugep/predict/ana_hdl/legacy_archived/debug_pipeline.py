#!/usr/bin/env python3
"""
Debug Pipeline - Comprehensive test/validation script for FuGEP data flow analysis.

This script captures and analyzes the full data flow:
inputs → model → predictions → output handling

It helps debug column mismatches, memory calculations, and data structure issues
in the enhanced handler system.
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import traceback
from typing import Dict, List, Any, Tuple
import gc

# Add FuGEP to path
sys.path.insert(0, '/home/sdodl001/FuGEP')

from fugep.predict.analyzer import Analyzer
from fugep.predict.seq_ana.gve.peak import PeakGVarEvaluator
from fugep.predict.ana_hdl.enhanced_mean_gve_handler import EnhancedMeanGVEHandler
from fugep.predict.ana_hdl.enhanced_pval_handler import EnhancedPvalHandler
from fugep.predict.ana_hdl.output_backends import ParquetBackend, TSVBackend
from fugep.data import Genome
from fugep.utils import load_features_list

class PipelineDebugger:
    """Debug and validate the FuGEP prediction pipeline."""
    
    def __init__(self, config_file: str, debug_output_dir: str = None):
        """Initialize the debugger with a config file."""
        self.config_file = config_file
        self.debug_output_dir = debug_output_dir or "/tmp/fugep_debug"
        os.makedirs(self.debug_output_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.debug_output_dir}/debug.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("PipelineDebugger")
        
        self.analysis_results = {}
        
    def load_config(self) -> Dict[str, Any]:
        """Load and parse the YAML configuration."""
        self.logger.info(f"Loading config from: {self.config_file}")
        
        with open(self.config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        self.logger.debug(f"Config keys: {list(config.keys())}")
        self.analysis_results['config'] = config
        return config
    
    def analyze_input_data(self, vcf_file: str, features_file: str) -> Dict[str, Any]:
        """Analyze the input VCF and features data."""
        self.logger.info("Analyzing input data...")
        
        input_analysis = {}
        
        # Analyze VCF file
        try:
            self.logger.info(f"Reading VCF file: {vcf_file}")
            vcf_df = pd.read_csv(vcf_file, sep='\t', nrows=10)  # Sample first 10 rows
            input_analysis['vcf'] = {
                'file_path': vcf_file,
                'sample_shape': vcf_df.shape,
                'columns': list(vcf_df.columns),
                'sample_data': vcf_df.head(3).to_dict('records'),
                'dtypes': vcf_df.dtypes.to_dict()
            }
            self.logger.info(f"VCF columns: {vcf_df.columns.tolist()}")
            self.logger.info(f"VCF sample shape: {vcf_df.shape}")
        except Exception as e:
            self.logger.error(f"Error reading VCF: {e}")
            input_analysis['vcf'] = {'error': str(e)}
        
        # Analyze features file
        try:
            self.logger.info(f"Reading features file: {features_file}")
            with open(features_file, 'r') as f:
                features = [line.strip() for line in f.readlines()]
            input_analysis['features'] = {
                'file_path': features_file,
                'count': len(features),
                'sample_features': features[:10],
                'all_features': features
            }
            self.logger.info(f"Features count: {len(features)}")
            self.logger.info(f"Sample features: {features[:5]}")
        except Exception as e:
            self.logger.error(f"Error reading features: {e}")
            input_analysis['features'] = {'error': str(e)}
        
        self.analysis_results['input_data'] = input_analysis
        return input_analysis
    
    def create_mock_evaluator(self, config: Dict[str, Any]) -> PeakGVarEvaluator:
        """Create a PeakGVarEvaluator with limited data for testing."""
        self.logger.info("Creating mock evaluator...")
        
        analyzer_config = config['analyzer']
        
        # Create limited VCF file for testing (first 5 variants)
        original_vcf = analyzer_config['vcfFile']
        test_vcf = f"{self.debug_output_dir}/test_variants.tsv"
        
        # Copy first few lines for testing
        with open(original_vcf, 'r') as src, open(test_vcf, 'w') as dst:
            header = src.readline()
            dst.write(header)
            for i, line in enumerate(src):
                if i >= 5:  # Only first 5 variants
                    break
                dst.write(line)
        
        self.logger.info(f"Created test VCF with 5 variants: {test_vcf}")
        
        # Create evaluator
        evaluator = PeakGVarEvaluator(
            analysis=analyzer_config['analysis'],
            refSeq=Genome(input_path=analyzer_config['refSeq']['input_path']),
            vcfFile=test_vcf,
            strandIdx=analyzer_config['strandIdx'],
            seqLen=analyzer_config['seqLen'],
            batchSize=2,  # Small batch for debugging
            useCuda=False,
            dataParallel=False,
            loggingVerbosity=2,  # More verbose
            writeMemLimit=analyzer_config.get('writeMemLimit', 50),
            save_mult_pred=analyzer_config.get('save_mult_pred', True),
            outputFormat=analyzer_config.get('outputFormat', 'parquet'),
            dataTypeConfig=analyzer_config.get('dataTypeConfig', 'memory_optimized'),
            trainedModelPath=analyzer_config['trainedModelPath'],
            features=load_features_list(input_path=analyzer_config['features']['input_path'])
        )
        
        self.analysis_results['evaluator_config'] = {
            'test_vcf': test_vcf,
            'batch_size': 2,
            'features_count': len(evaluator.features),
            'analysis_types': evaluator.analysis
        }
        
        return evaluator
    
    def analyze_variant_loading(self, evaluator: PeakGVarEvaluator) -> Dict[str, Any]:
        """Analyze how variants are loaded and processed."""
        self.logger.info("Analyzing variant loading...")
        
        variant_analysis = {}
        
        try:
            # Get variant data
            evaluator._loadVcfFile()
            variant_analysis['variant_count'] = len(evaluator.vcfDf)
            variant_analysis['vcf_columns'] = list(evaluator.vcfDf.columns)
            variant_analysis['sample_variants'] = evaluator.vcfDf.head(3).to_dict('records')
            
            # Analyze variant ID generation
            if hasattr(evaluator, '_generate_variant_ids'):
                sample_ids = []
                for i in range(min(3, len(evaluator.vcfDf))):
                    row = evaluator.vcfDf.iloc[i]
                    var_id = evaluator._generate_variant_ids(row)
                    sample_ids.append(var_id)
                
                variant_analysis['sample_variant_ids'] = sample_ids
                variant_analysis['variant_id_type'] = type(sample_ids[0]) if sample_ids else None
            
            self.logger.info(f"Loaded {len(evaluator.vcfDf)} variants")
            self.logger.info(f"VCF columns: {evaluator.vcfDf.columns.tolist()}")
            
        except Exception as e:
            self.logger.error(f"Error in variant loading analysis: {e}")
            variant_analysis['error'] = str(e)
            variant_analysis['traceback'] = traceback.format_exc()
        
        self.analysis_results['variant_loading'] = variant_analysis
        return variant_analysis
    
    def analyze_handler_creation(self, evaluator: PeakGVarEvaluator) -> Dict[str, Any]:
        """Analyze handler creation and configuration."""
        self.logger.info("Analyzing handler creation...")
        
        handler_analysis = {}
        
        try:
            # Create handlers directly
            mean_gve_handler = EnhancedMeanGVEHandler(
                output_format='parquet',
                data_type_config='memory_optimized',
                features=evaluator.features
            )
            
            pval_handler = EnhancedPvalHandler(
                output_format='parquet', 
                data_type_config='memory_optimized',
                features=evaluator.features
            )
            
            handler_analysis['mean_gve'] = {
                'class': mean_gve_handler.__class__.__name__,
                'output_format': mean_gve_handler.output_format,
                'data_type_config': mean_gve_handler.data_type_config,
                'features_count': len(mean_gve_handler.features),
                'backend_type': type(mean_gve_handler.backend).__name__,
                'columns_for_ids': getattr(mean_gve_handler, 'columns_for_ids', None),
                'chunk_size': getattr(mean_gve_handler.backend, 'chunk_size', None)
            }
            
            handler_analysis['pval'] = {
                'class': pval_handler.__class__.__name__,
                'output_format': pval_handler.output_format,
                'data_type_config': pval_handler.data_type_config,
                'features_count': len(pval_handler.features),
                'backend_type': type(pval_handler.backend).__name__,
                'columns_for_ids': getattr(pval_handler, 'columns_for_ids', None),
                'chunk_size': getattr(pval_handler.backend, 'chunk_size', None)
            }
            
            self.logger.info(f"Created handlers - Mean GVE: {type(mean_gve_handler.backend).__name__}, Pval: {type(pval_handler.backend).__name__}")
            
        except Exception as e:
            self.logger.error(f"Error in handler creation analysis: {e}")
            handler_analysis['error'] = str(e)
            handler_analysis['traceback'] = traceback.format_exc()
        
        self.analysis_results['handler_creation'] = handler_analysis
        return handler_analysis
    
    def simulate_prediction_data(self, evaluator: PeakGVarEvaluator, num_variants: int = 3) -> Dict[str, Any]:
        """Simulate prediction data structure to test handlers."""
        self.logger.info("Simulating prediction data...")
        
        simulation = {}
        
        try:
            # Simulate variant IDs
            variant_ids = []
            for i in range(num_variants):
                var_id = f"chr1_100{i}_A_T"  # Simple variant ID format
                variant_ids.append(var_id)
            
            # Simulate predictions (features x variants)
            num_features = len(evaluator.features)
            predictions = np.random.rand(num_features, num_variants).astype(np.float32)
            
            simulation['variant_ids'] = variant_ids
            simulation['predictions_shape'] = predictions.shape
            simulation['predictions_dtype'] = str(predictions.dtype)
            simulation['sample_predictions'] = predictions[:3, :].tolist()  # First 3 features, all variants
            simulation['features_count'] = num_features
            
            self.logger.info(f"Simulated data - Variants: {len(variant_ids)}, Features: {num_features}, Predictions shape: {predictions.shape}")
            
            # Test handler processing
            self.test_handler_processing(variant_ids, predictions, evaluator.features)
            
        except Exception as e:
            self.logger.error(f"Error in prediction simulation: {e}")
            simulation['error'] = str(e)
            simulation['traceback'] = traceback.format_exc()
        
        self.analysis_results['prediction_simulation'] = simulation
        return simulation
    
    def test_handler_processing(self, variant_ids: List[str], predictions: np.ndarray, features: List[str]):
        """Test how handlers process the prediction data."""
        self.logger.info("Testing handler processing...")
        
        handler_test = {}
        
        try:
            # Create handlers
            mean_gve_handler = EnhancedMeanGVEHandler(
                output_format='parquet',
                data_type_config='memory_optimized',
                features=features
            )
            
            # Test mean GVE processing
            self.logger.info("Testing mean GVE handler...")
            
            # Simulate what happens in the handler
            mean_gve_results = []
            for i, var_id in enumerate(variant_ids):
                # Each variant gets mean prediction across features
                mean_pred = np.mean(predictions[:, i])
                
                # Create result row structure
                if isinstance(var_id, (list, tuple)):
                    result_row = list(var_id) + [mean_pred]
                else:
                    result_row = [var_id, mean_pred]
                
                mean_gve_results.append(result_row)
                self.logger.debug(f"Mean GVE result row {i}: {result_row}")
            
            # Test expected columns
            expected_columns = mean_gve_handler.columns_for_ids + ['mean_gve']
            self.logger.info(f"Expected columns: {expected_columns}")
            self.logger.info(f"Result row length: {len(mean_gve_results[0]) if mean_gve_results else 0}")
            
            handler_test['mean_gve'] = {
                'expected_columns': expected_columns,
                'expected_column_count': len(expected_columns),
                'result_rows': mean_gve_results,
                'result_row_lengths': [len(row) for row in mean_gve_results],
                'column_mismatch': len(expected_columns) != len(mean_gve_results[0]) if mean_gve_results else False
            }
            
            # Test with actual backend writing
            try:
                # Create temporary output directory
                temp_output = f"{self.debug_output_dir}/test_output"
                os.makedirs(temp_output, exist_ok=True)
                
                # Initialize backend with proper output path
                mean_gve_handler.backend.initialize_output(temp_output, "test_mean_gve")
                
                # Try to write the data
                for result_row in mean_gve_results:
                    mean_gve_handler.backend.add_result([result_row])
                
                # Force write
                mean_gve_handler.backend.write_current_chunk()
                
                handler_test['mean_gve']['write_test'] = "SUCCESS"
                self.logger.info("Mean GVE handler write test: SUCCESS")
                
            except Exception as write_error:
                handler_test['mean_gve']['write_test'] = f"FAILED: {write_error}"
                handler_test['mean_gve']['write_traceback'] = traceback.format_exc()
                self.logger.error(f"Mean GVE handler write test failed: {write_error}")
            
        except Exception as e:
            self.logger.error(f"Error in handler processing test: {e}")
            handler_test['error'] = str(e)
            handler_test['traceback'] = traceback.format_exc()
        
        self.analysis_results['handler_processing'] = handler_test
    
    def analyze_memory_calculation(self, predictions: np.ndarray, variant_ids: List[str]) -> Dict[str, Any]:
        """Analyze memory calculation logic."""
        self.logger.info("Analyzing memory calculation...")
        
        memory_analysis = {}
        
        try:
            import sys
            
            # Test different memory calculation approaches
            
            # 1. Traditional method (single result row)
            single_result_row = [variant_ids[0], np.mean(predictions[:, 0])]
            traditional_mem = sys.getsizeof(single_result_row)
            
            # 2. Enhanced method (all result rows)
            all_result_rows = []
            for i, var_id in enumerate(variant_ids):
                result_row = [var_id, np.mean(predictions[:, i])]
                all_result_rows.append(result_row)
            
            enhanced_mem = sys.getsizeof(all_result_rows)
            
            # 3. Per-variant calculation
            per_variant_mem = sum(sys.getsizeof(row) for row in all_result_rows)
            
            memory_analysis = {
                'variant_count': len(variant_ids),
                'predictions_memory_mb': predictions.nbytes / (1024 * 1024),
                'traditional_single_row_bytes': traditional_mem,
                'enhanced_all_rows_bytes': enhanced_mem,
                'per_variant_sum_bytes': per_variant_mem,
                'comparison': {
                    'traditional_vs_enhanced_ratio': enhanced_mem / traditional_mem if traditional_mem > 0 else 0,
                    'per_variant_vs_enhanced_ratio': per_variant_mem / enhanced_mem if enhanced_mem > 0 else 0
                }
            }
            
            self.logger.info(f"Memory analysis - Traditional: {traditional_mem}B, Enhanced: {enhanced_mem}B, Per-variant: {per_variant_mem}B")
            
        except Exception as e:
            self.logger.error(f"Error in memory calculation analysis: {e}")
            memory_analysis['error'] = str(e)
            memory_analysis['traceback'] = traceback.format_exc()
        
        self.analysis_results['memory_calculation'] = memory_analysis
        return memory_analysis
    
    def run_full_analysis(self) -> Dict[str, Any]:
        """Run the complete pipeline analysis."""
        self.logger.info("Starting full pipeline analysis...")
        
        try:
            # 1. Load configuration
            config = self.load_config()
            
            # 2. Analyze input data
            vcf_file = config['analyzer']['vcfFile']
            features_file = config['analyzer']['features']['input_path']
            self.analyze_input_data(vcf_file, features_file)
            
            # 3. Create mock evaluator
            evaluator = self.create_mock_evaluator(config)
            
            # 4. Analyze variant loading
            self.analyze_variant_loading(evaluator)
            
            # 5. Analyze handler creation
            self.analyze_handler_creation(evaluator)
            
            # 6. Simulate prediction data and test processing
            self.simulate_prediction_data(evaluator)
            
            # 7. Analyze memory calculation
            if 'prediction_simulation' in self.analysis_results:
                sim = self.analysis_results['prediction_simulation']
                if 'variant_ids' in sim and 'predictions_shape' in sim:
                    # Create dummy predictions for memory analysis
                    variant_ids = sim['variant_ids']
                    predictions = np.random.rand(*sim['predictions_shape']).astype(np.float32)
                    self.analyze_memory_calculation(predictions, variant_ids)
            
            self.logger.info("Full analysis completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Error in full analysis: {e}")
            self.analysis_results['full_analysis_error'] = {
                'error': str(e),
                'traceback': traceback.format_exc()
            }
        
        return self.analysis_results
    
    def save_analysis_report(self):
        """Save the analysis results to files."""
        self.logger.info("Saving analysis report...")
        
        # Save as YAML
        report_file = f"{self.debug_output_dir}/analysis_report.yml"
        with open(report_file, 'w') as f:
            yaml.dump(self.analysis_results, f, default_flow_style=False, indent=2)
        
        # Save as text summary
        summary_file = f"{self.debug_output_dir}/analysis_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("FuGEP Pipeline Debug Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            
            for section, data in self.analysis_results.items():
                f.write(f"{section.upper()}:\n")
                f.write("-" * 30 + "\n")
                
                if isinstance(data, dict):
                    for key, value in data.items():
                        f.write(f"  {key}: {value}\n")
                else:
                    f.write(f"  {data}\n")
                f.write("\n")
        
        self.logger.info(f"Analysis report saved to: {report_file}")
        self.logger.info(f"Analysis summary saved to: {summary_file}")


def main():
    """Main function to run the pipeline debugger."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug FuGEP prediction pipeline")
    parser.add_argument("config_file", help="Path to the YAML configuration file")
    parser.add_argument("--output-dir", default="/tmp/fugep_debug", 
                       help="Output directory for debug files (default: /tmp/fugep_debug)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config_file):
        print(f"Error: Config file not found: {args.config_file}")
        return 1
    
    # Run the analysis
    debugger = PipelineDebugger(args.config_file, args.output_dir)
    results = debugger.run_full_analysis()
    debugger.save_analysis_report()
    
    print(f"\nDebug analysis completed!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Check {args.output_dir}/debug.log for detailed logs")
    print(f"Check {args.output_dir}/analysis_report.yml for full results")
    print(f"Check {args.output_dir}/analysis_summary.txt for summary")
    
    # Print key findings
    if 'handler_processing' in results:
        handler_proc = results['handler_processing']
        if 'mean_gve' in handler_proc:
            mean_gve = handler_proc['mean_gve']
            print(f"\nKEY FINDINGS:")
            print(f"- Expected columns: {mean_gve.get('expected_columns', 'N/A')}")
            print(f"- Expected column count: {mean_gve.get('expected_column_count', 'N/A')}")
            print(f"- Result row lengths: {mean_gve.get('result_row_lengths', 'N/A')}")
            print(f"- Column mismatch: {mean_gve.get('column_mismatch', 'N/A')}")
            print(f"- Write test: {mean_gve.get('write_test', 'N/A')}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
