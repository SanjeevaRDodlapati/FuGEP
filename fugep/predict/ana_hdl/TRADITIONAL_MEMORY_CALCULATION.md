## Traditional Memory Calculation Implementation

### Overview
The enhanced handlers now use the traditional memory calculation method for consistency with the original handlers.

### Implementation Details

#### Memory Calculation Method
The enhanced handlers and output backends now use the exact same memory calculation logic as the original handlers:

```python
# Traditional method (from handler.py)
mem_used = (self._results[0].nbytes * len(self._results) +
            getsizeof(self._samples[0]) * len(self._samples))
return mem_used / 10**6 >= self._write_mem_limit
```

#### Key Components:
1. **Results Memory**: `self._results[0].nbytes * len(self._results)`
   - Uses `.nbytes` attribute for numpy arrays
   - Falls back to `getsizeof()` for non-numpy arrays

2. **Samples Memory**: `getsizeof(self._samples[0]) * len(self._samples)`
   - Uses `getsizeof()` for sample/ID storage

3. **Unit Conversion**: Divides by `10**6` (not `1000*1000`)
   - Maintains exact compatibility with original system

#### Enhanced Handler Behavior:

**Backend Mode (Parquet/Chunked):**
- Uses the backend's `_reached_mem_limit()` method
- Processes individual result rows for chunked output
- Memory calculated on processed data

**Traditional Mode (TSV/HDF5):**
- Uses inherited `_reached_mem_limit()` from PredictionsHandler
- Stores numpy arrays directly (like original handlers)
- Exact same memory calculation pattern

#### Files Modified:
- `output_backends.py`: Traditional memory calculation method
- `enhanced_mean_gve_handler.py`: Traditional pattern in legacy mode
- `enhanced_pval_handler.py`: Traditional pattern in legacy mode

### Benefits:
1. **Consistency**: Same memory behavior as original handlers
2. **Compatibility**: Seamless transition for existing workflows
3. **Accuracy**: Uses actual numpy array sizes for precise calculation
4. **Fallback**: Graceful degradation if calculation fails

### Future Considerations:
- The parallel memory estimation method remains available for future use
- Decision on which method to use can be made based on performance testing
- Current implementation prioritizes compatibility and consistency
