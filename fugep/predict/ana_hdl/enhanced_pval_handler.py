"""
Enhanced P-value Handler with pluggable output backends.
Supports TSV, HDF5, and Parquet formats through backend abstraction.

Important: In traditional mode (TSV/HDF5), this handler follows the exact same
memory calculation pattern as the original PvalHandler for consistency.
The backend system also uses the traditional memory calculation method.
"""
import numpy as np
from scipy import stats
from .handler import PredictionsHandler
from .output_backends import ParquetBackend, TSVBackend, HDF5Backend


class EnhancedPvalHandler(PredictionsHandler):
    """
    Enhanced P-value Handler that supports multiple output formats through pluggable backends.
    
    This replaces both PvalHandler and PvalParquetHandler with a single, flexible implementation.
    """
    
    def __init__(self,
                 features,
                 columns_for_ids,
                 output_path_prefix,
                 mult_predictions,
                 save_mult_pred,
                 output_format,
                 output_size=None,
                 write_mem_limit=1500,
                 write_labels=True,
                 data_type_config='memory_optimized'):
        """
        Parameters
        ----------
        output_format : {'tsv', 'hdf5', 'parquet'}
            Output format. Parquet is recommended for large datasets.
        data_type_config : str, default 'memory_optimized'
            Data type optimization level: 'production', 'memory_optimized', 'high_precision'
        data_type_config : str, default 'memory_optimized'
            Data type optimization level: 'memory_optimized', 'production', 'high_precision'
        """
        # Store configuration first (needed by _create_backend)
        self._features = features
        self._columns_for_ids = columns_for_ids
        self._output_format = output_format
        self._mult_predictions = mult_predictions
        self._data_type_config = data_type_config
        
        # Initialize parent class for TSV/HDF5 compatibility when needed
        if output_format in ['tsv', 'hdf5']:
            super(EnhancedPvalHandler, self).__init__(
                features, columns_for_ids, output_path_prefix, mult_predictions,
                save_mult_pred, output_format, output_size, write_mem_limit, write_labels)
            self._create_write_handler("pval")
            self._backend = None
        else:
            # Use backend system for parquet and other formats
            self._backend = self._create_backend(
                output_format, output_path_prefix, features, columns_for_ids,
                write_mem_limit)
            
            # If backend creation failed (e.g., missing parquet deps), fall back to TSV
            if self._backend is None:
                super(EnhancedPvalHandler, self).__init__(
                    features, columns_for_ids, output_path_prefix, mult_predictions,
                    save_mult_pred, 'tsv', output_size, write_mem_limit, write_labels)
                self._create_write_handler("pval")
        
        self.needs_base_pred = True
        self._results = []
        self._samples = []
        # Note: No longer tracking _all_ids to prevent memory leaks
    
    def _create_backend(self, output_format, output_path_prefix, features, 
                       columns_for_ids, write_mem_limit):
        """Factory method to create appropriate backend."""
        if output_format == 'parquet':
            # Check for parquet dependencies and fall back to TSV if missing
            try:
                import pyarrow
                return ParquetBackend(output_path_prefix, features, columns_for_ids,
                                    write_mem_limit, suffix="_pval",
                                    data_type_config=self._data_type_config)
            except ImportError:
                try:
                    import fastparquet
                    return ParquetBackend(output_path_prefix, features, columns_for_ids,
                                        write_mem_limit, suffix="_pval",
                                        data_type_config=self._data_type_config)
                except ImportError:
                    print("WARNING: Parquet dependencies not found. Falling back to TSV format.")
                    print("To use parquet format, install with: pip install pyarrow")
                    # Fall back to traditional TSV handler
                    return None  # Will use traditional handler
        elif output_format == 'tsv_chunked':
            return TSVBackend(output_path_prefix, features, columns_for_ids,
                            write_mem_limit)
        elif output_format == 'hdf5_chunked':
            return HDF5Backend(output_path_prefix, features, columns_for_ids,
                             write_mem_limit)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def handle_batch_predictions(self, batch_predictions, batch_ids, baseline_predictions):
        """
        Calculate p-values for a batch and store results.
        
        NOTE: This method is for single predictions per variant. 
        For proper p-value computation, you need multiple predictions (use handle_batch_mult_predictions).
        """
        # For single predictions, we can't compute meaningful p-values
        # Store placeholder values for consistency
        results = []
        for i, variant_id in enumerate(batch_ids):
            # Create result row with NaN p-values (can't do t-test with single prediction)
            pvals = [np.nan] * len(self._features)
            result_row = [variant_id] + pvals
            results.append(result_row)
        
        # Store results using appropriate method
        if self._backend:
            # Use backend system (parquet, chunked formats)
            self._backend.add_results(results, batch_ids)
            # No need to accumulate IDs in handler - backend handles this
        else:
            # Use traditional system (TSV, HDF5)
            # For traditional system, store absolute diffs (matching original behavior)
            absolute_diffs = np.abs(baseline_predictions[0] - batch_predictions[0])
            self._results.append(absolute_diffs)
            self._samples.append(batch_ids)
            if self._reached_mem_limit():
                self.write_to_file()
    
    def handle_batch_mult_predictions(self, batch_predictions, batch_ids, baseline_predictions):
        """
        Calculate p-values for multiple predictions per variant.
        
        This is the correct method for p-value computation following the original PvalHandler logic.
        """
        # Calculate differences: baseline - variant predictions
        diffs = baseline_predictions - batch_predictions
        
        # Perform t-test across the multiple predictions (axis=0)
        # This gives one p-value per feature
        _, pvals = stats.ttest_1samp(diffs, 0, axis=0)
        
        # Prepare results
        results = []
        for i, variant_id in enumerate(batch_ids):
            # All variants in this batch get the same p-values (computed across all predictions)
            pvals_list = pvals.tolist() if hasattr(pvals, 'tolist') else list(pvals)
            result_row = [variant_id] + pvals_list
            results.append(result_row)
        
        # Store results using appropriate method
        if self._backend:
            # Use backend system (parquet, chunked formats)
            self._backend.add_results(results, batch_ids)
            # No need to accumulate IDs in handler - backend handles this
        else:
            # Use traditional system (TSV, HDF5) - matches original PvalHandler
            self._results.append(pvals)
            self._samples.append(batch_ids)
            if self._reached_mem_limit():
                self.write_to_file()
    
    def write_to_file(self, close_filehandle=True):
        """
        Write results to file using appropriate backend.
        """
        if self._backend:
            # Use backend system
            output_path = self._backend.finalize()
            if isinstance(output_path, str):
                print(f"P-value results written to {output_path}")
            else:
                print(f"P-value results written to {len(output_path)} chunk files")
        else:
            # Use traditional system
            super().write_to_file()
