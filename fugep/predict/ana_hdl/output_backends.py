"""
Output backend abstraction for different file formats.
Separates computation logic from storage format.
Includes optimal data type configuration for memory efficiency.

Note: The backend system uses the traditional memory calculation method
(same as original handlers) for consistent memory limit checking.
This includes: self._results[0].nbytes * len(self._results) + 
getsizeof(self._samples[0]) * len(self._samples), divided by 10**6.
"""
import os
import warnings
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from .optimal_data_types import OptimalDataTypes, apply_optimal_dtypes


class OutputBackend(ABC):
    """Abstract base class for output backends with optimal data type support."""
    
    def __init__(self, output_path_prefix, features, columns_for_ids, 
                 write_mem_limit=1500, data_type_config='production'):
        self.output_path_prefix = output_path_prefix
        self.features = features
        self.columns_for_ids = columns_for_ids
        self.write_mem_limit = write_mem_limit
        
        # Configure optimal data types for memory efficiency
        self.data_type_config = data_type_config
        self.dtype_schema = OptimalDataTypes.get_recommended_schema(data_type_config)
        
        self._results = []
        self._current_chunk_ids = []  # Track IDs only for current chunk
        self._chunk_files = []
        self._chunk_counter = 0
    
    @abstractmethod
    def write_chunk(self, results, chunk_id):
        """Write a chunk of results to storage."""
        pass
    
    @abstractmethod
    def get_final_output_path(self, suffix=""):
        """Get the final output file path."""
        pass
    
    def add_results(self, results, ids):
        """Add results and check if chunk writing is needed."""
        self._results.extend(results)
        self._current_chunk_ids.extend(ids)
        
        # Check memory limit using actual memory calculation
        if self._reached_mem_limit():
            self._write_current_chunk()
    
    def _reached_mem_limit(self):
        """
        Calculate memory usage using traditional handler method.
        Uses the exact same calculation as original handlers for consistency.
        """
        if not self._results or not self._current_chunk_ids:
            return False
            
        try:
            from sys import getsizeof
            
            # Traditional memory calculation (from handler.py):
            # mem_used = (self._results[0].nbytes * len(self._results) +
            #             getsizeof(self._samples[0]) * len(self._samples))
            # return mem_used / 10**6 >= self._write_mem_limit
            
            # Calculate memory for results 
            results_memory = 0
            if self._results:
                if hasattr(self._results[0], 'nbytes'):
                    # Traditional numpy array approach
                    results_memory = self._results[0].nbytes * len(self._results)
                else:
                    # Enhanced backend approach - self._results is a list of individual result rows
                    # Each row is [variant_id, feature1, feature2, ...]
                    num_rows = len(self._results)
                    if num_rows > 0:
                        # Estimate bytes per row: variant_id (string ~50 bytes) + features (float ~8 bytes each)
                        estimated_bytes_per_row = 50 + len(self.features) * 8
                        results_memory = num_rows * estimated_bytes_per_row
            
            # Calculate memory for ID storage using getsizeof
            ids_memory = getsizeof(self._current_chunk_ids[0]) * len(self._current_chunk_ids)
            
            # Total memory using traditional conversion (10**6, not 1000*1000)
            mem_used = results_memory + ids_memory
            total_memory_mb = mem_used / 10**6
            
            return total_memory_mb >= self.write_mem_limit
            
        except Exception as e:
            # Fallback to estimation if actual calculation fails
            print(f"Warning: Could not calculate memory usage using traditional method ({e}), using estimation")
            return len(self._results) * len(self.features) * 8 >= self.write_mem_limit * 1024 * 1024
    
    def _write_current_chunk(self):
        """Write current results as a chunk and clear memory."""
        if not self._results:
            return
            
        chunk_path = self.write_chunk(self._results, self._chunk_counter)
        self._chunk_files.append(chunk_path)
        self._chunk_counter += 1
        
        # Clear memory
        result_count = len(self._results)
        
        # Log memory usage info for debugging using traditional calculation
        if hasattr(self, '_reached_mem_limit'):
            try:
                from sys import getsizeof
                if self._results and self._current_chunk_ids:
                    # Use traditional memory calculation for logging
                    if hasattr(self._results[0], 'nbytes'):
                        results_mem = self._results[0].nbytes * len(self._results)
                    else:
                        # Enhanced backend - estimate memory for result rows
                        num_rows = len(self._results)
                        estimated_bytes_per_row = 50 + len(self.features) * 8
                        results_mem = num_rows * estimated_bytes_per_row
                    ids_mem = getsizeof(self._current_chunk_ids[0]) * len(self._current_chunk_ids)
                    total_mb = (results_mem + ids_mem) / 10**6  # Traditional conversion
                    print(f"Written chunk {self._chunk_counter} with {result_count} rows "
                          f"(~{total_mb:.1f}MB using traditional calculation)")
                else:
                    print(f"Written chunk {self._chunk_counter} with {result_count} rows")
            except:
                print(f"Written chunk {self._chunk_counter} with {result_count} rows")
        else:
            print(f"Written chunk {self._chunk_counter} with {result_count} rows")
        
        self._results = []
        self._current_chunk_ids = []  # Clear current chunk IDs
    
    def finalize(self):
        """Write remaining results as final chunk."""
        # Write any remaining results
        if self._results:
            self._write_current_chunk()
            
        if not self._chunk_files:
            print("No data to write")
            return None
            
        print(f"Written {len(self._chunk_files)} chunk files:")
        for chunk_file in self._chunk_files:
            print(f"  {chunk_file}")
        return self._chunk_files
    
    def _apply_optimal_dtypes(self, df):
        """Apply optimal data types to DataFrame for memory efficiency."""
        try:
            return apply_optimal_dtypes(df, self.dtype_schema, self.features)
        except Exception as e:
            # Fallback to original DataFrame if type conversion fails
            print(f"Warning: Could not apply optimal data types: {e}")
            return df
    
    def get_memory_info(self):
        """Get information about memory usage and data type optimization."""
        if not self._results:
            return {"status": "no_data"}
            
        # Estimate memory savings for current dataset
        num_rows = len(self._results)
        num_features = len(self.features)
        
        savings = OptimalDataTypes.get_memory_savings(
            num_rows, num_features, self.data_type_config)
        
        return {
            "current_rows": num_rows,
            "features": num_features,
            "data_type_config": self.data_type_config,
            "estimated_memory_mb": savings['optimized_memory_mb'],
            "estimated_savings_percent": savings['savings_percent'],
            "dtype_schema": self.dtype_schema
        }


class ParquetBackend(OutputBackend):
    """Parquet output backend using pandas/pyarrow."""
    
    def __init__(self, output_path_prefix, features, columns_for_ids, 
                 write_mem_limit=1500, suffix="", data_type_config='production'):
        super().__init__(output_path_prefix, features, columns_for_ids, 
                        write_mem_limit, data_type_config)
        self.suffix = suffix
        
        # Check for parquet dependencies
        try:
            import pyarrow
        except ImportError:
            try:
                import fastparquet
            except ImportError:
                raise ImportError("Either pyarrow or fastparquet is required for Parquet support")
    
    def write_chunk(self, results, chunk_id):
        """Write results chunk to parquet file with optimal data types."""
        # Use dynamic columns based on actual data
        columns = self.columns_for_ids + list(self.features)
        
        # Verify column count matches data
        if results and len(results[0]) != len(columns):
            actual_data_cols = len(results[0])
            expected_cols = len(columns)
            print(f"WARNING: Column count mismatch in ParquetBackend: expected {expected_cols}, got {actual_data_cols}")
            
            # Adjust columns to match actual data
            if actual_data_cols < expected_cols:
                # Fewer data columns than expected - truncate column names
                columns = columns[:actual_data_cols]
                print(f"DEBUG: Truncated columns to {len(columns)}")
            elif actual_data_cols > expected_cols:
                # More data columns than expected - add generic column names
                extra_cols = actual_data_cols - expected_cols
                extra_names = [f"extra_feature_{i}" for i in range(extra_cols)]
                columns.extend(extra_names)
                print(f"DEBUG: Added {extra_cols} extra columns: {extra_names}")
        
        df = pd.DataFrame(results, columns=columns)
        
        # Apply optimal data types for memory efficiency  
        df_optimized = self._apply_optimal_dtypes(df)
        
        chunk_path = f"{self.output_path_prefix}{self.suffix}_chunk_{chunk_id:06d}.parquet"
        df_optimized.to_parquet(chunk_path, index=False)
        
        # Log memory savings information
        if chunk_id == 0:  # Log only for first chunk to avoid spam
            memory_info = self.get_memory_info()
            print(f"Applied {self.data_type_config} data type optimization:")
            print(f"  - Estimated memory savings: {memory_info.get('estimated_savings_percent', 0):.1f}%")
            print(f"  - Data type schema: {memory_info.get('dtype_schema', {})}")
        
        return chunk_path
    
    def get_final_output_path(self, suffix=""):
        """Get final output path for merged file."""
        return f"{self.output_path_prefix}{self.suffix}.parquet"


class TSVBackend(OutputBackend):
    """TSV output backend (can be added for compatibility)."""
    
    def write_chunk(self, results, chunk_id):
        """Write results chunk to TSV file."""
        chunk_path = f"{self.output_path_prefix}_chunk_{chunk_id:06d}.tsv"
        
        columns = self.columns_for_ids + list(self.features)
        df = pd.DataFrame(results, columns=columns)
        df.to_csv(chunk_path, sep='\t', index=False)
        
        return chunk_path
    
    def get_final_output_path(self):
        return f"{self.output_path_prefix}.tsv"


class HDF5Backend(OutputBackend):
    """HDF5 output backend (can be added for compatibility)."""
    
    def write_chunk(self, results, chunk_id):
        """Write results chunk to HDF5 file."""
        # Implementation for HDF5 chunked writing
        pass
    
    def get_final_output_path(self):
        return f"{self.output_path_prefix}.h5"
