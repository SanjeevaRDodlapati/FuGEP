import os
import warnings
import numpy as np
import pandas as pd

class MeanGVEParquetHandler(object):
    """
    Parquet handler for mean GVE, matching the interface of MeanGVEHandler.
    Supports chunked writing and optional merging for large-scale datasets.
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
                 merge_chunks=False):
        """
        Parameters
        ----------
        merge_chunks : bool, default False
            If True, merge all chunk files into a single output file at the end.
            If False, keep chunk files as separate outputs for efficient column-wise access.
            For very large datasets (>TB), keeping chunks is often preferred.
        """
        self.needs_base_pred = True
        self._results = []
        self._all_ids = []
        
        self._features = features
        self._columns_for_ids = columns_for_ids
        self._output_path_prefix = output_path_prefix
        self._mult_predictions = mult_predictions
        self._save_mult_pred = save_mult_pred
        self._output_format = output_format
        self._output_size = output_size
        self._write_mem_limit = write_mem_limit
        self._write_labels = write_labels
        self._merge_chunks = merge_chunks
        
        self._chunk_files = []
        self._chunk_counter = 0
        self._baseline_predictions = None

    def handle_batch_predictions(self, 
                               batch_predictions, 
                               batch_ids, 
                               baseline_predictions):
        """
        Calculate mean GVE for a batch and store results.
        """
        if self._baseline_predictions is None:
            self._baseline_predictions = baseline_predictions

        # Calculate absolute differences between variant and reference predictions
        absolute_diffs = np.abs(baseline_predictions[0] - batch_predictions[0])
        
        # Prepare results similar to MeanGVEHandler
        results = []
        for i, variant_id in enumerate(batch_ids):
            gve_values = absolute_diffs[i]
            # If multiple predictions, average them
            if len(gve_values.shape) > 1:
                gve_mean = np.mean(gve_values, axis=0)
            else:
                gve_mean = gve_values
            
            # Create result row
            result_row = [variant_id] + gve_mean.tolist()
            results.append(result_row)

        # Store results for later writing
        self._results.extend(results)
        self._all_ids.extend(batch_ids)
        
        # Check if we should write a chunk
        if len(self._results) * len(self._features) * 8 >= self._write_mem_limit * 1024 * 1024:
            self._write_chunk()

    def _write_chunk(self):
        """Write current results to a chunk file."""
        if not self._results:
            return
            
        # Check for pyarrow availability
        try:
            import pyarrow
        except ImportError:
            try:
                import fastparquet
            except ImportError:
                raise ImportError("Either pyarrow or fastparquet is required for Parquet support")
        
        # Create column names
        columns = self._columns_for_ids + [f for f in self._features]
        
        # Create DataFrame
        df = pd.DataFrame(self._results, columns=columns)
        
        # Write chunk file
        chunk_path = f"{self._output_path_prefix}_chunk_{self._chunk_counter:06d}.parquet"
        df.to_parquet(chunk_path, index=False)
        
        self._chunk_files.append(chunk_path)
        self._chunk_counter += 1
        
        # Clear memory
        self._results = []
        
        print(f"Written chunk {self._chunk_counter} with {len(df)} rows to {chunk_path}")

    def write_to_file(self, close_filehandle=True):
        """
        Write remaining results and optionally merge chunks.
        """
        # Write any remaining results
        if self._results:
            self._write_chunk()
            
        if not self._chunk_files:
            print("No data to write")
            return
            
        if self._merge_chunks:
            self._merge_all_chunks()
        else:
            print(f"Kept {len(self._chunk_files)} chunk files for efficient column-wise access:")
            for chunk_file in self._chunk_files:
                print(f"  {chunk_file}")
            print(f"For column-wise operations, use: pd.read_parquet('{self._output_path_prefix}_chunk_*.parquet', columns=['col1', 'col2'])")

    def _merge_all_chunks(self):
        """
        Merge all chunk files into a single output file.
        Uses streaming approach for memory efficiency.
        """
        print(f"Merging {len(self._chunk_files)} chunk files...")
        
        output_path = f"{self._output_path_prefix}.parquet"
        
        # Use streaming merge for large datasets
        if len(self._chunk_files) > 100:  # Arbitrary threshold for "many chunks"
            self._streaming_merge(output_path)
        else:
            self._batch_merge(output_path)
        
        # Clean up chunk files after successful merge
        for chunk_file in self._chunk_files:
            try:
                os.remove(chunk_file)
            except OSError:
                warnings.warn(f"Could not remove chunk file: {chunk_file}")
        
        print(f"Merged results written to {output_path}")

    def _batch_merge(self, output_path):
        """Merge chunks by loading all into memory (for smaller datasets)."""
        dfs = []
        for chunk_file in self._chunk_files:
            df = pd.read_parquet(chunk_file)
            dfs.append(df)
        
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df.to_parquet(output_path, index=False)

    def _streaming_merge(self, output_path):
        """Merge chunks using streaming approach (for larger datasets)."""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            # Fallback to batch merge if pyarrow not available
            self._batch_merge(output_path)
            return
        
        # Read schema from first chunk
        first_table = pq.read_table(self._chunk_files[0])
        schema = first_table.schema
        
        # Create writer
        with pq.ParquetWriter(output_path, schema) as writer:
            for chunk_file in self._chunk_files:
                table = pq.read_table(chunk_file)
                writer.write_table(table)

    def get_ids(self):
        """Return all processed IDs."""
        return self._all_ids
