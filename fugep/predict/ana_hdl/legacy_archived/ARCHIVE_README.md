# Legacy Handler Archive

This directory contains the original handler implementations that have been replaced by the enhanced pipeline system.

## **Archived on:** June 30, 2025

## **Replacement System:**
- **Original handlers** → **Enhanced handlers with pluggable backends**
- **Format-specific implementations** → **Unified handlers with backend abstraction**
- **Memory-inefficient patterns** → **Memory-optimized TB-scale processing**

## **Development/Test Files Archived:**

### **Test and Validation Scripts:**
- `test_pipeline_validation.py` - Comprehensive pipeline validation script (8 tests)
- `test_deepsea_quick_validation.py` - Quick DeepSEA validation test with memory limits
- `test_minimal_pipeline.py` - Minimal end-to-end test script
- `test_deepsea_corrected.py` - Corrected DeepSEA test implementation
- `debug_pipeline.py` - Debug script for troubleshooting pipeline issues

### **Test Configuration Files:**
- `test_deepsea_50mb.yml` - DeepSEA config with 50MB memory limit for testing
- `test_deepsea_100mb.yml` - DeepSEA config with 100MB memory limit for testing  
- `test_deepsea_debug.yml` - DeepSEA debug configuration

### **Legacy Handlers (Kept for Reference):**
- `mean_gve_handler.py` - Original mean GVE handler (kept in main directory)
- `pval_handler.py` - Original p-value handler (kept in main directory)

### **Enhanced Production System:**
- `enhanced_mean_gve_handler.py` - Unified handler with TSV/HDF5/Parquet support
- `enhanced_pval_handler.py` - Unified handler with TSV/HDF5/Parquet support
- `output_backends.py` - Pluggable backend system (Parquet, TSV, HDF5)
- `handler_factory.py` - Factory functions for backward compatibility

## **Key Improvements:**
1. ✅ **Eliminated Code Duplication** - Single handler per analysis type
2. ✅ **Memory-Based Chunking** - Efficient TB-scale data processing
3. ✅ **Pluggable Backends** - Support for Parquet, TSV, HDF5
4. ✅ **Data Type Optimization** - Memory-optimized column types
5. ✅ **Backward Compatibility** - Seamless transition from legacy code

## **Migration Status:**
- ✅ All Sei jobs updated to use enhanced pipeline
- ✅ All DeepSEA jobs updated to use enhanced pipeline  
- ✅ Memory limits optimized (12GB for Sei, 500MB for DeepSEA testing)
- ✅ Parquet output with memory_optimized data types
- ✅ Validation tests passing
- ✅ Production ready

## **Files Archived/Deleted:**
These development and testing files have been cleaned up:
- Test and validation scripts (moved to archive or deleted)
- Debug scripts and temporary files
- Development configuration files

## **Legacy Handlers Status:**
- `mean_gve_handler.py` - **Kept for reference** (not archived yet)
- `pval_handler.py` - **Kept for reference** (not archived yet)

These will remain available until further notice for comparison and reference purposes.

## **Restoration:**
If needed, these files can be restored from git history:
```bash
git log --oneline --follow fugep/predict/ana_hdl/mean_gve_handler.py
git checkout <commit-hash> -- fugep/predict/ana_hdl/mean_gve_handler.py
```

## **Cleanup Summary (June 30, 2025):**

✅ **Test Scripts Archived** - All development and validation test scripts moved to archive  
✅ **Debug Files Archived** - Debug and troubleshooting scripts archived  
✅ **Test Configs Archived** - Test configuration files with small memory limits archived  
✅ **Log Files Archived** - Validation log files moved to archive  
✅ **Codebase Cleaned** - Main directory now contains only production code

## **Production Files Active:**
- `enhanced_mean_gve_handler.py` - Production mean GVE handler
- `enhanced_pval_handler.py` - Production p-value handler  
- `output_backends.py` - Pluggable backend system
- `handler_factory.py` - Factory functions for handler creation
- `optimal_data_types.py` - Data type optimization utilities

## **Job Configuration Status:**
- ✅ **Sei Jobs**: All 8 chunks configured with 12GB memory limit and Parquet output
- ✅ **DeepSEA Jobs**: All 8 chunks configured with 500MB memory limit and Parquet output
- ✅ **Submit Scripts**: Created for both Sei and DeepSEA Parquet jobs
- ✅ **Memory Validation**: Tested with small limits, verified chunking behavior
