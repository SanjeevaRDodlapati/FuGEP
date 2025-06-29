import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats

class PvalParquetHandler(object):
    """
    Parquet handler for p-value computation, matching the interface of PvalHandler.
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
        self._samples = []
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
        self._output_filepath = output_path_prefix + "_pval.parquet"

    def handle_batch_predictions(self, batch_predictions, batch_ids, baseline_predictions):
        absolute_diffs = np.abs(baseline_predictions[0] - batch_predictions[0])
        self._results.append(absolute_diffs)
        self._samples.append(batch_ids)
        
        # Check memory limit
        if len(self._results) > self._write_mem_limit:
            self._write_chunk()

    def handle_batch_mult_predictions(self, batch_predictions, batch_ids, baseline_predictions):
        diffs = baseline_predictions - batch_predictions
        _, pval = stats.ttest_1samp(diffs, 0, axis=0)
        self._results.append(pval)
        self._samples.append(batch_ids)
        
        # Check memory limit
        if len(self._results) > self._write_mem_limit:
            self._write_chunk()

    def _write_chunk(self):
        """Write current batch to temporary file and clear memory"""
        if not hasattr(self, '_chunk_files'):
            self._chunk_files = []
        
        if not self._results:
            return
            
        chunk_file = f"{self._output_filepath}.chunk_{len(self._chunk_files)}"
        
        # Create rows for DataFrame
        rows = []
        for sample, result in zip(self._samples, self._results):
            if isinstance(sample, (list, tuple)):
                row = list(sample) + [result]
            else:
                row = [sample] + [result]
            rows.append(row)
        
        columns = self._columns_for_ids + ['pval']
        df = pd.DataFrame(rows, columns=columns)
        df.to_parquet(chunk_file, index=False)
        self._chunk_files.append(chunk_file)
        
        # Clear memory
        self._results = []
        self._samples = []

    def write_to_file(self):
        """Write all accumulated results to final Parquet file"""
        # First write any remaining data as a chunk
        if self._results:
            self._write_chunk()
        
        # If we have chunk files, merge them
        if hasattr(self, '_chunk_files') and self._chunk_files:
            dfs = []
            for chunk_file in self._chunk_files:
                df = pd.read_parquet(chunk_file)
                dfs.append(df)
            
            # Combine all chunks
            final_df = pd.concat(dfs, ignore_index=True)
            final_df.to_parquet(self._output_filepath, index=False)
            
            # Clean up chunk files
            import os
            for chunk_file in self._chunk_files:
                try:
                    os.remove(chunk_file)
                except OSError:
                    pass
            
            print(f"P-value results written to {self._output_filepath} ({len(final_df)} rows)")
        
        elif self._results:
            # Direct write if no chunks (small dataset)
            rows = []
            for sample, result in zip(self._samples, self._results):
                if isinstance(sample, (list, tuple)):
                    row = list(sample) + [result]
                else:
                    row = [sample] + [result]
                rows.append(row)
            
            columns = self._columns_for_ids + ['pval']
            df = pd.DataFrame(rows, columns=columns)
            df.to_parquet(self._output_filepath, index=False)
            print(f"P-value results written to {self._output_filepath} ({len(df)} rows)")
        
        # Clear final memory
        self._results = []
        self._samples = []
        for ids, vals in zip(self._samples, self._results):
            for id_row, val_row in zip(ids, np.atleast_2d(vals)):
                row = list(id_row) + list(val_row)
                rows.append(row)
        columns = self._columns_for_ids + self._features
        df = pd.DataFrame(rows, columns=columns)
        df.to_parquet(self._output_filepath, index=False)
        self._results = []
        self._samples = []
