# Differentiable Gaussian Rasterization

CUDA rasterization sources used by the hybrid 3D/4D Gaussian renderer in this repository. The implementation builds on the rasterizer from *3D Gaussian Splatting for Real-Time Radiance Field Rendering*.

## Integration

[gaussian_renderer/diff_gaussian_rasterization.py](../gaussian_renderer/diff_gaussian_rasterization.py) loads these sources with torch.utils.cpp_extension.load. Importing that wrapper compiles the local extension and exposes the forward/backward operations used by [gaussian_renderer/__init__.py](../gaussian_renderer/__init__.py).

A compatible CUDA toolkit with nvcc, a C++ compiler, Ninja, and CUDA-enabled PyTorch are required. Training uses this JIT path; a separate pip installation of this directory is not required by the training entry point. See the [repository README](../README.md) for environment setup.

## Citation

Please cite the original rasterization work when using it in research:

~~~bibtex
@Article{kerbl3Dgaussians,
  author = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  title = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  journal = {ACM Transactions on Graphics},
  number = {4},
  volume = {42},
  month = {July},
  year = {2023},
  url = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
~~~

Original source notices and bundled third-party licenses are retained.
