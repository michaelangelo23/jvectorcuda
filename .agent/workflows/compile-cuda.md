---
description: How to compile CUDA kernels (.cu) to PTX files for JVectorCUDA
---

# CUDA Kernel Compilation Workflow

## Prerequisites
- Must run from **Visual Studio 2019 Developer Command Prompt**
- nvcc and cl.exe must be in PATH

## Compilation Steps

// turbo-all

1. Navigate to the jvectorcuda project directory:
```
cd C:\Users\Michael Angelo\Documents\Vindex\jvectorcuda
```

2. Compile all CUDA kernels to PTX:
```
nvcc -ptx -o src\main\resources\kernels\euclidean_distance.ptx src\main\resources\kernels\euclidean_distance.cu
nvcc -ptx -o src\main\resources\kernels\cosine_similarity.ptx src\main\resources\kernels\cosine_similarity.cu
nvcc -ptx -o src\main\resources\kernels\inner_product.ptx src\main\resources\kernels\inner_product.cu
```

3. Verify the PTX files were created:
```
dir src\main\resources\kernels\*.ptx
```

## Notes
- PTX files are forward-compatible (CUDA 11.8+ runtime)
- The batch kernels (`*_batch.ptx`) are already pre-compiled
- If you modify any `.cu` file, you must recompile to `.ptx`
