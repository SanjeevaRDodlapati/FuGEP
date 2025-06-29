import numpy as np
import pandas as pd

class MeanGVEParquetHandler(object):
    """
    Parquet handler for mean GVE, matching the interface of MeanGVEHandler.
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
                 write_labels=True):
        self.needs_base_pred = True
        self._results = []
        self._samples = []
        self._features = features
        self._columns_for_ids = columns_for_ids
        self._output_path_prefix = output_path_prefix
        self._mult_predictions = mult_predictions
        self._save_mult_pred = save_mult_pred
        self._output_format = output_format
        self._output_size = output_size
        self._write_mem_limit = write_mem_limit
        self._write_labels = write_labels
        self._output_filepath = output_path_prefix + "_mean_gve.parquet"

    def handle_batch_predictions(self, batch_predictions, batch_ids, baseline_predictions=None):
        absolute_diffs = np.abs(baseline_predictions[0] - batch_predictions[0])
        self._results.extend(absolute_diffs)
        self._samples.extend(batch_ids)
        
        # Check memory limit
        if len(self._results) > self._write_mem_limit:
            self._write_chunk()

    def handle_batch_mult_predictions(self, batch_predictions, batch_ids, baseline_predictions):
        diffs = baseline_predictions - batch_predictions
        gve_mean = np.mean(diffs, axis=0)
        self._results.append(gve_mean)
        self._samples.append(batch_ids)
        # Optionally, implement memory limit logic if needed

    def _write_chunk(self):
        """Write current batch to temporary file and clear memory"""
        if not hasattr(self, '_chunk_files'):
            self._chunk_files = []
        
        chunk_file = f"{self._output_filepath}.chunk_{len(self._chunk_files)}"
        rows = [dict(zip(self._columns_for_ids + self._features, 
                        list(sample) + [result]))
               for sample, result in zip(self._samples, self._results)]
        
        df = pd.DataFrame(rows)
        df.to_parquet(chunk_file, index=False)
        self._chunk_files.append(chunk_file)
        
        # Clear memory
        self._results = []
        self._samples = []

    def write_to_file(self):
        if not self._results:
            return None
        # Flatten samples and results
        rows = []
        for ids, vals in zip(self._samples, self._results):
            for id_row, val_row in zip(ids, np.atleast_2d(vals)):
                row = list(id_row) + list(val_row)
                rows.append(row)
        columns = self._columns_for_ids + self._features
        df = pd.DataFrame(rows, columns=columns)
        df.to_parquet(self._output_filepath, index=False)
        self._results = []
        self._samples = []
