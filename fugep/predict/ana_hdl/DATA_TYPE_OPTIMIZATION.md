# Data Type Optimization for TB-Scale Genomics Data

## Overview

The FuGEP pipeline now includes optimal data type configuration to dramatically reduce memory usage and storage requirements for TB-scale genomics datasets. The system defaults to **memory-optimized** configuration for maximum efficiency, with easy override options for specific requirements.

## 🎯 **Quick Start**

### Basic Usage (Default: Memory-Optimized)
```python
# Default: memory_optimized (75% memory savings)
mean_gve_handler = create_mean_gve_handler(
    output_format='parquet',
    # data_type_config defaults to 'memory_optimized'
    features=features,
    columns_for_ids=['variant_id'],
    output_path_prefix='/path/to/output'
)
```

### Override for Specific Requirements
```python
# Override to production settings (50% memory savings)
mean_gve_handler = create_mean_gve_handler(
    output_format='parquet',
    data_type_config='production',  # <- Override default
    features=features,
    columns_for_ids=['variant_id'],
    output_path_prefix='/path/to/output'
)
```

## 📊 **Data Type Configurations**

### Production Configuration (Recommended)
- **Memory Savings**: ~50%
- **Use Case**: Most genomics pipelines
- **Precision**: Scientific accuracy maintained

| Data Type | Original | Optimized | Range | Precision |
|-----------|----------|-----------|-------|-----------|
| Chromosome | object (8 bytes) | uint16 (2 bytes) | 0-65535 | Perfect |
| Position | int64 (8 bytes) | uint32 (4 bytes) | 0-4.3B | Covers human genome |
| GVE Scores | float64 (8 bytes) | float32 (4 bytes) | ±3.4e38 | 7 decimal places |
| P-values | float64 (8 bytes) | float32 (4 bytes) | ±3.4e38 | 7 decimal places |

### Memory-Optimized Configuration
- **Memory Savings**: ~75%
- **Use Case**: TB-scale data, storage-constrained environments
- **Precision**: Acceptable loss for large-scale analysis

| Data Type | Original | Optimized | Range | Precision |
|-----------|----------|-----------|-------|-----------|
| Chromosome | object (8 bytes) | uint8 (1 byte) | 1-255 | chr1=1, chrX=23, etc. |
| Position | int64 (8 bytes) | uint32 (4 bytes) | 0-4.3B | Covers human genome |
| GVE Scores | float64 (8 bytes) | float16 (2 bytes) | ±65504 | 3 decimal places |
| P-values | float64 (8 bytes) | float32 (4 bytes) | ±3.4e38 | 7 decimal places* |

*P-values use float32 even in memory-optimized mode for statistical accuracy

### High-Precision Configuration
- **Memory Savings**: Minimal
- **Use Case**: Research requiring maximum precision
- **Precision**: Maximum available

| Data Type | Original | Optimized | Notes |
|-----------|----------|-----------|-------|
| All types | Same | Same | No optimization, maximum precision |

## 🔧 **Configuration Options**

### Handler Factory Configuration
```python
from fugep.predict.ana_hdl.handler_factory import create_mean_gve_handler

# Method 1: Pass data_type_config parameter
handler = create_mean_gve_handler(
    output_format='parquet',
    data_type_config='production',  # or 'memory_optimized', 'high_precision'
    features=features,
    # ... other parameters
)

# Method 2: Use format shortcuts for different configurations
production_handler = create_mean_gve_handler('parquet', 'production', ...)
memory_handler = create_mean_gve_handler('parquet', 'memory_optimized', ...)
precision_handler = create_mean_gve_handler('parquet', 'high_precision', ...)
```

### Analyzer Configuration
```python
# In analyzer.py _initializeReporters method
enhanced_args = {
    'features': self._features,
    'columns_for_ids': colNamesOfIds,
    'output_path_prefix': outputPath,
    'output_format': 'parquet',
    'data_type_config': 'production'  # <- Configure here
}

if "mean_gve" == s:
    reporters.append(create_mean_gve_handler(**enhanced_args))
```

### YAML Configuration
```yaml
# config.yml
analyzer:
  class_name: "PeakGVarEvaluator"
  analysis: ["mean_gve", "pval"]
  output_format: "parquet"
  data_type_config: "production"  # or "memory_optimized", "high_precision"
  writeMemLimit: 10000
```

## 📈 **Memory Savings Analysis**

### Example: 100M Variants × 200 Features

| Configuration | Memory Usage | Savings | File Size (per chunk) |
|---------------|--------------|---------|----------------------|
| Default (pandas) | 160 GB | 0% | ~16 GB |
| Production | 80 GB | 50% | ~8 GB |
| Memory Optimized | 40 GB | 75% | ~4 GB |
| High Precision | 160 GB | 0% | ~16 GB |

### Real-World TB-Scale Impact

For your 2000 chunk × 10GB scenario:
- **Default**: 2000 × 16GB = 32TB total
- **Production**: 2000 × 8GB = 16TB total (50% savings = 16TB saved)
- **Memory Optimized**: 2000 × 4GB = 8TB total (75% savings = 24TB saved)

## ⚠️ **Important Considerations**

### When to Use Each Configuration

**Production (Recommended)**
- ✅ Most genomics pipelines
- ✅ Balance of memory savings and precision
- ✅ Standard scientific accuracy maintained
- ✅ Compatible with downstream analysis tools

**Memory Optimized**
- ✅ TB-scale datasets where storage is critical
- ✅ Exploratory analysis with large datasets
- ⚠️ Slightly reduced precision for GVE scores (float16)
- ✅ P-values maintain full precision (float32)

**High Precision**
- ✅ Critical research requiring maximum precision
- ✅ Small to medium datasets
- ❌ Not recommended for TB-scale data (memory constraints)

### Chromosome Encoding

The system automatically handles chromosome encoding:

```python
# Input chromosomes (any format)
['chr1', 'chr2', 'chr22', 'chrX', 'chrY', 'chrM']

# Minimal encoding (uint8): chr1=1, chr2=2, ..., chrX=23, chrY=24, chrM=25
# Compact encoding (uint16): Flexible mapping for unknown chromosomes
# Standard encoding (category): Pandas category type
```

### Precision Requirements

**GVE Scores**: 
- float32: Scientific accuracy for most applications (7 decimal places)
- float16: Use only for exploratory analysis (3 decimal places)

**P-values**:
- Always use float32 or higher for statistical significance testing
- float16 not recommended for p-values

## 🚀 **Migration Guide**

### From Existing Code
```python
# OLD: No data type optimization
handler = create_mean_gve_handler('parquet', features=features, ...)

# NEW: With optimization (backward compatible)
handler = create_mean_gve_handler('parquet', 'production', features=features, ...)
```

### Performance Testing
```python
# Test memory usage
from fugep.predict.ana_hdl.optimal_data_types import OptimalDataTypes

savings = OptimalDataTypes.get_memory_savings(
    num_rows=1000000,
    num_features=200, 
    use_case='production'
)
print(f"Memory savings: {savings['savings_percent']:.1f}%")
```

## 🔍 **Troubleshooting**

### Common Issues

**"Could not apply optimal data types" Warning**
- System falls back to original types if conversion fails
- Data still written correctly, just without optimization
- Check for unusual chromosome names or data ranges

**Precision Loss Concerns**
- Use 'high_precision' configuration for critical research
- Test with representative data subset first
- Compare results between configurations

**Memory Constraints**
- Use 'memory_optimized' for maximum savings
- Adjust write_mem_limit for available system memory
- Consider processing data in smaller batches

The optimized data type system provides significant memory and storage savings while maintaining the scientific accuracy required for genomics research. Choose the configuration that best fits your use case and computational constraints.
