"""
Example usage of the new enhanced handler system.
Demonstrates mi        output_path_prefix='/path/to/output',
        mult_predictions=False,
        save_mult_pred=False,
        write_mem_limit=10000  # 10GB chunks
    )n from old separate handlers to unified system.
"""

# OLD WAY (before enhancement) - Required 4 separate files:
# from .mean_gve_handler import MeanGVEHandler  
# from .mean_gve_parquet import MeanGVEParquetHandler
# from .pval_handler import PvalHandler
# from .pval_parquet import PvalParquetHandler

# NEW WAY (after enhancement) - Single unified system:
from .handler_factory import create_mean_gve_handler, create_pval_handler


def example_old_usage_equivalent():
    """Show how old usage patterns work with new system."""
    
    # OLD: MeanGVEHandler(...) 
    # NEW: (exact same interface)
    mean_gve_handler = create_mean_gve_handler(
        output_format='tsv',
        features=['feat1', 'feat2', 'feat3'],
        columns_for_ids=['variant_id'],
        output_path_prefix='/path/to/output',
        mult_predictions=False,
        save_mult_pred=False,
        write_mem_limit=1500
    )
    
    # OLD: PvalHandler(...)
    # NEW: (exact same interface)  
    pval_handler = create_pval_handler(
        output_format='tsv',
        features=['feat1', 'feat2', 'feat3'],
        columns_for_ids=['variant_id'],
        output_path_prefix='/path/to/output',
        mult_predictions=True,
        save_mult_pred=True,
        write_mem_limit=1500
    )
    
    # Usage is exactly the same
    # mean_gve_handler.handle_batch_predictions(batch_preds, batch_ids, baseline_preds)
    # pval_handler.handle_batch_mult_predictions(batch_preds, batch_ids, baseline_preds)


def example_new_parquet_usage():
    """Show new parquet capabilities for large-scale genomics data."""
    
    # NEW: Basic parquet usage (uses memory_optimized by default)
    # This keeps chunks separate for efficient column-wise access
    # Perfect for TB-scale data (e.g., 2000 × 10GB chunks = 20TB total)
    mean_gve_parquet = create_mean_gve_handler(
        output_format='parquet',  # Uses memory_optimized by default (75% memory savings)
        features=['feat1', 'feat2', 'feat3'],
        columns_for_ids=['variant_id'],
        output_path_prefix='/path/to/output',
        mult_predictions=False,
        save_mult_pred=False,
        write_mem_limit=10000  # 10GB chunks (adjust based on memory)
    )
    
    # NEW: Override default for specific precision requirements
    production_handler = create_pval_handler(
        output_format='parquet',
        data_type_config='production',  # Override default - balanced precision/memory (50% reduction)
        features=['feat1', 'feat2', 'feat3'],
        columns_for_ids=['variant_id'],
        output_path_prefix='/path/to/output',
        mult_predictions=True,
        save_mult_pred=True,
        write_mem_limit=10000  # 10GB chunks
    )
    
    # NEW: High precision for research applications
    research_handler = create_mean_gve_handler(
        output_format='parquet',
        data_type_config='high_precision',  # Maximum precision, larger memory usage
        features=['feat1', 'feat2', 'feat3'],
        columns_for_ids=['variant_id'],
        output_path_prefix='/path/to/output',
        mult_predictions=False,
        save_mult_pred=False,
        write_mem_limit=10000
    )
    
    # Usage is exactly the same as before
    # handler.handle_batch_predictions(batch_preds, batch_ids, baseline_preds)
    # pval_parquet_chunked.handle_batch_mult_predictions(batch_preds, batch_ids, baseline_preds)


def example_migration_benefits():
    """Show the benefits of the new system."""
    
    print("✅ BENEFITS OF NEW SYSTEM:")
    print("1. Single codebase - no duplication between TSV/Parquet handlers")
    print("2. Pluggable backends - easy to add new formats (Arrow, Feather, etc.)")
    print("3. Consistent API - same interface for all formats") 
    print("4. Better memory management - configurable chunking")
    print("5. Column-wise access - parquet chunks enable efficient queries")
    print("6. Backward compatibility - existing code continues to work")
    print("7. Future-proof - easy to extend with new statistical methods")
    
    print("\n🚀 PERFORMANCE IMPROVEMENTS:")
    print("- Parquet: ~50% smaller files than TSV")
    print("- Chunking: Process TB-scale datasets without OOM")
    print("- Column access: Read only needed features from large files") 
    print("- Compression: Built-in compression with parquet format")
    
    print("\n🧬 GENOMICS-SCALE OPTIMIZATIONS:")
    print("- No merging: Keep 2000 × 10GB chunks separate (avoids 20TB single files)")
    print("- Memory efficient: Fixed 10-20GB RAM usage regardless of dataset size")
    print("- Parallel processing: Each chunk can be processed independently")
    print("- Column queries: Fast feature-specific analysis across all chunks")
    print("- Fault tolerance: Partial results preserved if interrupted")


def example_data_type_optimization():
    """Demonstrate data type optimization benefits for TB-scale genomics data."""
    from .optimal_data_types import OptimalDataTypes
    
    print("\n" + "="*60)
    print("DATA TYPE OPTIMIZATION FOR TB-SCALE GENOMICS DATA")
    print("="*60)
    
    # Simulate TB-scale dataset parameters
    num_variants = 100_000_000  # 100M variants
    num_features = 200          # 200 genomic features  
    
    print(f"\nDataset: {num_variants:,} variants × {num_features} features")
    print("-" * 50)
    
    # Show memory savings for different optimization levels
    for use_case in ['production', 'memory_optimized', 'high_precision']:
        savings = OptimalDataTypes.get_memory_savings(num_variants, num_features, use_case)
        
        print(f"\n{use_case.upper()} CONFIGURATION:")
        print(f"  Default memory:    {savings['default_memory_mb']:,.0f} MB ({savings['default_memory_mb']/1024:.1f} GB)")
        print(f"  Optimized memory:  {savings['optimized_memory_mb']:,.0f} MB ({savings['optimized_memory_mb']/1024:.1f} GB)")
        print(f"  Memory savings:    {savings['savings_mb']:,.0f} MB ({savings['savings_percent']:.1f}%)")
        print(f"  Schema: chromosome={savings['schema']['chromosome']}, position={savings['schema']['position']}")
        print(f"          gve_scores={savings['schema']['gve_scores']}, pval_scores={savings['schema']['pval_scores']}")
    
    print("\n" + "="*60)
    print("CONFIGURATION RECOMMENDATIONS:")
    print("="*60)
    print("• PRODUCTION: Best balance of precision and memory savings")
    print("  - Use for most genomics pipelines")
    print("  - 50% memory reduction with scientific precision")
    print()
    print("• MEMORY_OPTIMIZED: Maximum memory savings for TB-scale data")
    print("  - Use when memory/storage is critical constraint")  
    print("  - 75% memory reduction, acceptable precision loss")
    print()
    print("• HIGH_PRECISION: Maximum precision for research")
    print("  - Use for critical research requiring highest precision")
    print("  - Larger memory usage but no precision loss")
    print()
    
    # Example handler creation with different optimizations
    print("EXAMPLE HANDLER CREATION:")
    print("-" * 30)
    print("# Production (recommended)")
    print("handler = create_mean_gve_handler('parquet', 'production', ...)")
    print()
    print("# Memory optimized for TB-scale data")  
    print("handler = create_mean_gve_handler('parquet', 'memory_optimized', ...)")
    print()
    print("# High precision for research")
    print("handler = create_mean_gve_handler('parquet', 'high_precision', ...)")


def example_chromosome_encoding():
    """Show chromosome encoding options for memory efficiency."""
    import pandas as pd
    from .optimal_data_types import OptimalDataTypes
    
    print("\n" + "="*50)
    print("CHROMOSOME ENCODING EXAMPLES")
    print("="*50)
    
    # Sample chromosome data
    sample_chroms = ['chr1', 'chr2', 'chr22', 'chrX', 'chrY', 'chrM'] * 1000
    chr_series = pd.Series(sample_chroms)
    
    print(f"Original data: {chr_series.dtype}, memory: {chr_series.memory_usage(deep=True)} bytes")
    
    # Show different encoding options
    for encoding in ['minimal', 'compact', 'standard']:
        encoded = OptimalDataTypes.encode_chromosome(chr_series, encoding)
        memory_bytes = encoded.memory_usage(deep=True)
        memory_savings = ((chr_series.memory_usage(deep=True) - memory_bytes) / 
                         chr_series.memory_usage(deep=True)) * 100
        
        print(f"{encoding:10}: {str(encoded.dtype):8}, memory: {memory_bytes:6} bytes ({memory_savings:.1f}% savings)")
        if encoding == 'minimal':
            print(f"             Encoding: chr1=1, chr2=2, ..., chrX=23, chrY=24, chrM=25")
        elif encoding == 'compact':
            print(f"             Encoding: uint16 with flexible mapping for unknown chromosomes")
        else:
            print(f"             Encoding: pandas category type (efficient for repeated values)")
        print()


if __name__ == "__main__":
    print("Enhanced Handler System Examples")
    print("=" * 40)
    example_migration_benefits()
    example_data_type_optimization()
    example_chromosome_encoding()
