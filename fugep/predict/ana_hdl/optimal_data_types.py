"""
Optimal Data Type Configuration for TB-Scale Genomics Output
==========================================================

This module defines optimal data types for genomics data columns to minimize memory usage
and storage while maintaining precision requirements for scientific accuracy.
"""

import numpy as np
import pandas as pd


class OptimalDataTypes:
    """
    Defines optimal data types for different genomics data categories.
    
    Memory Savings Analysis:
    - float64 → float32: 50% memory reduction  
    - float64 → float16: 75% memory reduction (use cautiously for scores)
    - int64 → uint32: 50% reduction for positions (supports up to 4.3B positions)
    - int64 → uint8: 87.5% reduction for chromosomes (supports 1-255 chr names)
    """
    
    # Chromosome data types
    CHROMOSOME_TYPES = {
        # Option 1: Minimal memory (87.5% reduction) - for numeric chromosomes only
        'minimal': 'uint8',        # Range: 1-255 (1,2,3...22,X=23,Y=24,M=25)
        
        # Option 2: Compact (75% reduction) - supports chr prefixes  
        'compact': 'uint16',       # Range: 0-65535 (enough for any chromosome encoding)
        
        # Option 3: Standard (50% reduction) - for string chromosome names
        'standard': 'category'     # Pandas category type (efficient for repeated values)
    }
    
    # Genomic position data types  
    POSITION_TYPES = {
        # Option 1: Compact (50% reduction) - supports up to 4.3 billion positions
        'uint32': 'uint32',        # Range: 0 to 4,294,967,295 (covers human genome)
        
        # Option 2: Standard - if positions exceed 4.3B or negative values needed
        'int64': 'int64'           # Full range but larger memory footprint
    }
    
    # GVE and p-value score data types
    SCORE_TYPES = {
        # Option 1: High precision (50% reduction) - RECOMMENDED for GVE scores
        'float32': 'float32',      # Range: ±3.4e38, precision: ~7 decimal places
        
        # Option 2: Medium precision (75% reduction) - use with caution for p-values
        'float16': 'float16',      # Range: ±65504, precision: ~3 decimal places
        
        # Option 3: Full precision - only if extreme precision required
        'float64': 'float64'       # Maximum precision but largest memory usage
    }
    
    @classmethod
    def get_recommended_schema(cls, use_case='production'):
        """
        Get recommended data type schema based on use case.
        
        Parameters
        ----------
        use_case : str
            - 'production': Balanced precision/memory for production pipelines
            - 'memory_optimized': Maximum memory savings for large datasets
            - 'high_precision': Maximum precision for research
            
        Returns
        -------
        dict
            Dictionary mapping column types to pandas/numpy dtypes
        """
        if use_case == 'production':
            return {
                'chromosome': cls.CHROMOSOME_TYPES['compact'],     # uint16 
                'position': cls.POSITION_TYPES['uint32'],          # uint32
                'gve_scores': cls.SCORE_TYPES['float32'],          # float32
                'pval_scores': cls.SCORE_TYPES['float32'],         # float32 (safer than float16)
                'variant_id': 'string'                             # Efficient string type
            }
        elif use_case == 'memory_optimized':
            return {
                'chromosome': cls.CHROMOSOME_TYPES['minimal'],     # uint8 (encode chr1=1, chrX=23, etc.)
                'position': cls.POSITION_TYPES['uint32'],          # uint32  
                'gve_scores': cls.SCORE_TYPES['float16'],          # float16 (use with caution)
                'pval_scores': cls.SCORE_TYPES['float32'],         # float32 (p-values need more precision)
                'variant_id': 'string'                             # Efficient string type
            }
        elif use_case == 'high_precision':
            return {
                'chromosome': cls.CHROMOSOME_TYPES['standard'],    # category
                'position': cls.POSITION_TYPES['int64'],           # int64
                'gve_scores': cls.SCORE_TYPES['float64'],          # float64
                'pval_scores': cls.SCORE_TYPES['float64'],         # float64
                'variant_id': 'string'                             # Efficient string type
            }
        else:
            raise ValueError(f"Unknown use_case: {use_case}")
    
    @classmethod
    def get_memory_savings(cls, num_rows, num_features, use_case='production'):
        """
        Calculate estimated memory savings compared to default pandas types.
        
        Parameters
        ----------
        num_rows : int
            Number of variants/rows
        num_features : int
            Number of genomic features
        use_case : str
            Use case scenario
            
        Returns
        -------
        dict
            Memory usage comparison and savings
        """
        schema = cls.get_recommended_schema(use_case)
        
        # Default pandas dtypes (what it would use without optimization)
        default_bytes_per_row = (
            8 +                          # chromosome (object/string)
            8 +                          # position (int64)  
            8 * num_features +           # gve/pval scores (float64)
            16                           # variant_id (object/string estimate)
        )
        
        # Optimized dtypes
        optimized_bytes_per_row = 0
        if schema['chromosome'] == 'uint8':
            optimized_bytes_per_row += 1
        elif schema['chromosome'] == 'uint16':
            optimized_bytes_per_row += 2
        else:  # category
            optimized_bytes_per_row += 4  # category index
            
        if schema['position'] == 'uint32':
            optimized_bytes_per_row += 4
        else:  # int64
            optimized_bytes_per_row += 8
            
        if schema['gve_scores'] == 'float16':
            optimized_bytes_per_row += 2 * num_features
        elif schema['gve_scores'] == 'float32':
            optimized_bytes_per_row += 4 * num_features
        else:  # float64
            optimized_bytes_per_row += 8 * num_features
            
        optimized_bytes_per_row += 16  # variant_id string
        
        default_total_mb = (default_bytes_per_row * num_rows) / (1024 * 1024)
        optimized_total_mb = (optimized_bytes_per_row * num_rows) / (1024 * 1024)
        savings_mb = default_total_mb - optimized_total_mb
        savings_percent = (savings_mb / default_total_mb) * 100
        
        return {
            'default_memory_mb': default_total_mb,
            'optimized_memory_mb': optimized_total_mb,
            'savings_mb': savings_mb,
            'savings_percent': savings_percent,
            'use_case': use_case,
            'schema': schema
        }
    
    @classmethod
    def encode_chromosome(cls, chromosome_series, encoding_type='compact'):
        """
        Encode chromosome names to numeric types for memory efficiency.
        
        Parameters
        ----------
        chromosome_series : pandas.Series
            Series of chromosome names (e.g., 'chr1', 'chr2', 'chrX', 'chrY', 'chrM')
        encoding_type : str
            Type of encoding to use ('minimal', 'compact', 'standard')
            
        Returns
        -------
        pandas.Series
            Encoded chromosome series with appropriate dtype
        """
        if encoding_type == 'minimal':
            # Map chromosomes to integers: chr1=1, chr2=2, ..., chrX=23, chrY=24, chrM=25
            chr_map = {}
            for i in range(1, 23):  # chr1-chr22
                chr_map[f'chr{i}'] = i
                chr_map[str(i)] = i  # Also handle numeric strings
            chr_map.update({'chrX': 23, 'X': 23, 'chrY': 24, 'Y': 24, 
                           'chrM': 25, 'M': 25, 'chrMT': 25, 'MT': 25})
            
            encoded = chromosome_series.map(chr_map)
            # Handle unknown chromosomes
            encoded = encoded.fillna(255)  # Use 255 for unknown/unplaced contigs
            return encoded.astype('uint8')
            
        elif encoding_type == 'compact':
            # Use uint16 with more flexible encoding
            chr_map = {}
            for i in range(1, 23):  # chr1-chr22
                chr_map[f'chr{i}'] = i
                chr_map[str(i)] = i
            chr_map.update({'chrX': 23, 'X': 23, 'chrY': 24, 'Y': 24,
                           'chrM': 25, 'M': 25, 'chrMT': 25, 'MT': 25})
            
            # Assign remaining values to unknown chromosomes
            unique_chrs = chromosome_series.unique()
            next_value = 26
            for chr_name in unique_chrs:
                if chr_name not in chr_map:
                    chr_map[chr_name] = next_value
                    next_value += 1
            
            encoded = chromosome_series.map(chr_map)
            return encoded.astype('uint16')
            
        elif encoding_type == 'standard':
            # Use pandas category type
            return chromosome_series.astype('category')
        else:
            raise ValueError(f"Unknown encoding_type: {encoding_type}")


def get_optimal_types(use_case='production'):
    """
    Get optimal data types for the specified use case.
    Convenience function that wraps OptimalDataTypes.get_recommended_schema().
    
    Parameters
    ----------
    use_case : str
        Use case scenario ('production', 'memory_optimized', 'high_precision')
        
    Returns
    -------
    dict
        Recommended data type schema
    """
    return OptimalDataTypes.get_recommended_schema(use_case)


def apply_optimal_dtypes(df, columns_config, features):
    """
    Apply optimal data types to a genomics DataFrame.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with genomics data
    columns_config : dict
        Configuration dict from get_recommended_schema()
    features : list
        List of feature column names
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with optimized data types
    """
    df_optimized = df.copy()
    
    # Apply chromosome encoding if chromosome column exists
    if 'chrom' in df_optimized.columns or 'chromosome' in df_optimized.columns:
        chr_col = 'chrom' if 'chrom' in df_optimized.columns else 'chromosome'
        if columns_config['chromosome'] in ['uint8', 'uint16']:
            encoding_type = 'minimal' if columns_config['chromosome'] == 'uint8' else 'compact'
            df_optimized[chr_col] = OptimalDataTypes.encode_chromosome(
                df_optimized[chr_col], encoding_type)
        else:
            df_optimized[chr_col] = df_optimized[chr_col].astype(columns_config['chromosome'])
    
    # Apply position data type if position column exists
    if 'pos' in df_optimized.columns or 'position' in df_optimized.columns:
        pos_col = 'pos' if 'pos' in df_optimized.columns else 'position'
        df_optimized[pos_col] = df_optimized[pos_col].astype(columns_config['position'])
    
    # Apply score data types to feature columns
    for feature in features:
        if feature in df_optimized.columns:
            if 'gve' in feature.lower():
                df_optimized[feature] = df_optimized[feature].astype(columns_config['gve_scores'])
            elif 'pval' in feature.lower():
                df_optimized[feature] = df_optimized[feature].astype(columns_config['pval_scores'])
    
    # Apply variant ID type if exists
    if 'variant_id' in df_optimized.columns:
        df_optimized['variant_id'] = df_optimized['variant_id'].astype(columns_config['variant_id'])
    
    return df_optimized


# Example usage and benchmarking
if __name__ == "__main__":
    # Example: Calculate memory savings for a large genomics dataset
    num_variants = 100_000_000  # 100M variants
    num_features = 200          # 200 genomic features
    
    print("Memory Usage Analysis for 100M Variants × 200 Features")
    print("=" * 60)
    
    for use_case in ['production', 'memory_optimized', 'high_precision']:
        savings = OptimalDataTypes.get_memory_savings(num_variants, num_features, use_case)
        print(f"\n{use_case.upper()} USE CASE:")
        print(f"  Default memory:    {savings['default_memory_mb']:,.1f} MB ({savings['default_memory_mb']/1024:.1f} GB)")
        print(f"  Optimized memory:  {savings['optimized_memory_mb']:,.1f} MB ({savings['optimized_memory_mb']/1024:.1f} GB)")
        print(f"  Memory savings:    {savings['savings_mb']:,.1f} MB ({savings['savings_percent']:.1f}%)")
        print(f"  Schema: {savings['schema']}")
