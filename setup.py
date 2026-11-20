import sys
from setuptools import setup, Extension

extra_compile_args = ["/O2", "/arch:AVX2", "/openmp"] if sys.platform == "win32" else [
    "-O3", "-march=native", "-fopenmp", "-Wall"
]
extra_link_args = [] if sys.platform == "win32" else ["-fopenmp"]

ext = Extension(
    "ttb2._ttb2_core",
    sources=["ttb2/_ttb2_core.c"],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

setup(name="ttb2", version="0.2.0", packages=["ttb2"], ext_modules=[ext])
