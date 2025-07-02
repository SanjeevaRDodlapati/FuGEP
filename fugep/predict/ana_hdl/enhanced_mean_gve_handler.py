"""
Enhanced Mean GVE Handler with pluggable output backends.
Supports TSV, HDF5, and Parquet formats through backend abstraction.

Important: In traditional mode (TSV/HDF5), this handler follows the exact same
memory calculation pattern as the original MeanGVEHandler for consistency.
The backend system also uses the traditional memory calculation method.
"""
import numpy as np
from .handler import PredictionsHandler
from .output_backends import ParquetBackend, TSVBackend, HDF5Backend


class EnhancedMeanGVEHandler(PredictionsHandler):
    """
    Enhanced Mean GVE Handler that supports multiple output formats through pluggable backends.
    
    This replaces both MeanGVEHandler and MeanGVEParquetHandler with a single, flexible implementation.
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
            super(EnhancedMeanGVEHandler, self).__init__(
                features, columns_for_ids, output_path_prefix, mult_predictions,
                save_mult_pred, output_format, output_size, write_mem_limit, write_labels)
            self._create_write_handler("mean_gve")
            self._backend = None
        else:
            # Use backend system for parquet and other formats
            self._backend = self._create_backend(
                output_format, output_path_prefix, features, columns_for_ids,
                write_mem_limit)
            
            # If backend creation failed (e.g., missing parquet deps), fall back to TSV
            if self._backend is None:
                super(EnhancedMeanGVEHandler, self).__init__(
                    features, columns_for_ids, output_path_prefix, mult_predictions,
                    save_mult_pred, 'tsv', output_size, write_mem_limit, write_labels)
                self._create_write_handler("mean_gve")
        
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
                                    write_mem_limit, suffix="_mean_gve",
                                    data_type_config=self._data_type_config)
            except ImportError:
                try:
                    import fastparquet
                    return ParquetBackend(output_path_prefix, features, columns_for_ids,
                                        write_mem_limit, suffix="_mean_gve",
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
        Calculate mean GVE for a batch and store results.
        """
        # Calculate absolute differences between variant and reference predictions
        absolute_diffs = np.abs(baseline_predictions[0] - batch_predictions[0])
        
        # Prepare results
        results = []
        for i, variant_id in enumerate(batch_ids):
            gve_values = absolute_diffs[i]
            
            # If multiple predictions, average them
            if len(gve_values.shape) > 1:
                gve_mean = np.mean(gve_values, axis=0)
            else:
                gve_mean = gve_values
            
            # Create result row
            result_row = list(variant_id) + gve_mean.tolist()
            results.append(result_row)
        
        # Store results using appropriate method
        if self._backend:
            # Use backend system (parquet, chunked formats)
            self._backend.add_results(results, batch_ids)
            # No need to accumulate IDs in handler - backend handles this
        else:
            # Use traditional system (TSV, HDF5) - follow original handler pattern
            # Traditional handlers expect numpy arrays, not lists of individual rows
            self._results.append(absolute_diffs)  # Append the numpy array directly
            self._samples.append(batch_ids)       # Append batch_ids as usual
            if self._reached_mem_limit():
                self.write_to_file()
    
    def handle_batch_mult_predictions(self, batch_predictions, batch_ids, baseline_predictions):
        """
        Calculate mean GVE for multiple predictions per variant.
        Follows the exact same pattern as the original MeanGVEHandler.
        """
        # Calculate differences: baseline - variant predictions (matches original)
        diffs = baseline_predictions - batch_predictions
        
        # Calculate mean GVE across multiple predictions (axis=0 - across mult_predictions)
        # diffs shape: (mult_predictions, batch_size, n_features) -> (5, 128, 151)
        # After mean: (batch_size, n_features) -> (128, 151)
        gve_mean = np.mean(diffs, axis=0)
        
        # Store results using appropriate method
        if self._backend:
            # Use backend system (parquet, chunked formats)
            
            # Prepare results: each variant gets its own GVE scores
            results = []
            for i, variant_id in enumerate(batch_ids):
                # Extract GVE scores for this variant (row i from gve_mean matrix)
                # gve_mean shape is (batch_size, n_features), so gve_mean[i, :] gives GVE scores for variant i
                variant_gve = gve_mean[i, :].tolist()
                
                # Check for first variant to update backend feature count if needed
                if i == 0:
                    # Check if we need to update backend feature count
                    if len(variant_gve) != len(self._features):
                        # Update backend to handle actual feature count
                        actual_feature_names = [f"gve_feature_{j}" for j in range(len(variant_gve))]
                        self._backend.features = actual_feature_names
                
                # Create result row: [chrom, pos, name] + [gve1, gve2, ..., gve151]
                result_row = list(variant_id) + variant_gve
                results.append(result_row)
            
            if results:
                self._backend.add_results(results, batch_ids)
            # No need to accumulate IDs in handler - backend handles this
        else:
            # Use traditional system (TSV, HDF5) - matches original MeanGVEHandler exactly
            self._results.append(gve_mean)
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
        else:
            # Use traditional system
            super().write_to_file()
