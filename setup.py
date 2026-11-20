import sys
from setuptools import setup, Extension

if sys.platform == "win32":
    compile_args = ["/O2", "/openmp"]
    link_args = []
else:
    compile_args = ["-O3", "-march=native", "-fopenmp", "-fPIC"]
    link_args = ["-fopenmp"]

c_ext = Extension(
    name="ttb1._ttb1_core",
    sources=["ttb1/_ttb1_core.c"],
    extra_compile_args=compile_args,
    extra_link_args=link_args,
)

setup(
    name="ttb1",
    version="0.1.0",
    packages=["ttb1"],
    ext_modules=[c_ext],
)
