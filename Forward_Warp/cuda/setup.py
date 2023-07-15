from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

setup(
    name='forward_warp_cuda',
    ext_modules=[
        CUDAExtension('forward_warp_cuda', [
            'forward_warp_cuda.cpp',
            'forward_warp_cuda_kernel.cu',
        ], extra_compile_args={'cxx': [], 'nvcc': ['-arch=sm_75']}),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
