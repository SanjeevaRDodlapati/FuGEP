"""
Factory functions for creating analysis handlers with different output formats.
Provides backward compatibility while enabling the new pluggable backend system.
"""

from .enhanced_mean_gve_handler import EnhancedMeanGVEHandler
from .enhanced_pval_handler import EnhancedPvalHandler


def create_mean_gve_handler(output_format='tsv', data_type_config='memory_optimized', **kwargs):
    """
    Factory function to create Mean GVE handlers.
    
    Parameters
    ----------
    output_format : str
        - 'tsv': Traditional TSV output (compatible with existing code)
        - 'hdf5': Traditional HDF5 output (compatible with existing code)  
        - 'parquet': Efficient parquet format with chunking support (recommended for large datasets)
    data_type_config : str
        Data type optimization level:
        - 'memory_optimized': Maximum memory savings for TB-scale data (default)
        - 'production': Balanced precision/memory
        - 'high_precision': Maximum precision for research
    **kwargs : dict
        All other parameters passed to handler constructor
    
    Returns
    -------
    EnhancedMeanGVEHandler
        Handler instance configured for the specified output format
    
    Examples
    --------
    # Traditional usage (unchanged)
    handler = create_mean_gve_handler('tsv', features=features, ...)
    
    # New parquet usage for large datasets (keeps chunks separate)
    handler = create_mean_gve_handler('parquet', features=features, ...)
    
    # Memory-optimized for TB-scale data  
    handler = create_mean_gve_handler('parquet', 'memory_optimized', features=features, ...)
    """
    # Set data_type_config if not already specified
    if 'data_type_config' not in kwargs:
        kwargs['data_type_config'] = data_type_config
    
    return EnhancedMeanGVEHandler(output_format=output_format, **kwargs)


def create_pval_handler(output_format='tsv', data_type_config='memory_optimized', **kwargs):
    """
    Factory function to create P-value handlers.
    
    Parameters
    ----------
    output_format : str
        - 'tsv': Traditional TSV output (compatible with existing code)
        - 'hdf5': Traditional HDF5 output (compatible with existing code)  
        - 'parquet': Efficient parquet format with chunking support (recommended for large datasets)
        - 'parquet_merged': Parquet format, merge chunks into single file (use only for small datasets)
    data_type_config : str
        Data type optimization level:
        - 'memory_optimized': Maximum memory savings for TB-scale data (default)
        - 'production': Balanced precision/memory
        - 'high_precision': Maximum precision for research
    **kwargs : dict
        All other parameters passed to handler constructor
    
    Returns
    -------
    EnhancedPvalHandler
        Handler instance configured for the specified output format
    
    Examples
    --------
    # Traditional usage (unchanged)
    handler = create_pval_handler('tsv', features=features, ...)
    
    # New parquet usage for large datasets (keeps chunks separate)
    handler = create_pval_handler('parquet', features=features, ...)
    
    # Memory-optimized for TB-scale data
    handler = create_pval_handler('parquet', 'memory_optimized', features=features, ...)
    """
    # Set data_type_config if not already specified
    if 'data_type_config' not in kwargs:
        kwargs['data_type_config'] = data_type_config
    
    return EnhancedPvalHandler(output_format=output_format, **kwargs)


# Backward compatibility aliases
def MeanGVEHandler(**kwargs):
    """Backward compatibility alias for traditional Mean GVE handler."""
    return create_mean_gve_handler('tsv', **kwargs)


def PvalHandler(**kwargs):
    """Backward compatibility alias for traditional P-value handler.""" 
    return create_pval_handler('tsv', **kwargs)


def MeanGVEParquetHandler(**kwargs):
    """Backward compatibility alias for parquet Mean GVE handler."""
    return create_mean_gve_handler('parquet', **kwargs)


def PvalParquetHandler(**kwargs):
    """Backward compatibility alias for parquet P-value handler."""
    return create_pval_handler('parquet', **kwargs)
