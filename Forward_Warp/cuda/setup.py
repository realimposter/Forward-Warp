from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='forward_warp_cuda',
    ext_modules=[
        CUDAExtension('forward_warp_cuda', [
            'forward_warp_cuda.cpp',
            'forward_warp_cuda_kernel.cu',
        ], extra_compile_args={'cxx': [], 'nvcc': ['-arch=sm_80',
                                                   '-gencode', 'arch=compute_80,code=sm_80',
                                                   '-gencode', 'arch=compute_86,code=sm_86',
                                                   '-gencode', 'arch=compute_89,code=sm_89',
                                                   '-gencode', 'arch=compute_89,code=compute_89']}),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
