import cuml
import cupy
# runtimeGetVersion returns e.g. 12080 for CUDA 12.8
ver = cupy.cuda.runtime.runtimeGetVersion()
print(f"cuML {cuml.__version__}, CUDA runtime {ver//1000}.{(ver%1000)//10}")
# Test GPU support via cuML
from cuml.manifold import UMAP
print("UMAP imported, GPU ready")