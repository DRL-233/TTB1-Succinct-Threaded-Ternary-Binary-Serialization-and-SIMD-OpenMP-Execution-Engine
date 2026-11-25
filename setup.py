from setuptools import setup, Extension

ext = Extension(
    "_ttb2_core",
    sources=["ttb2/_ttb2_core.c", "ttb2/ttb2_lib.c"],
    include_dirs=["ttb2"],
    extra_compile_args=["-O3", "-march=native", "-fopenmp"],
    extra_link_args=["-fopenmp"],
)

setup(
    name="ttb2",
    version="0.3.0",
    description="TTB3-Bridge: Cycloidal Hub-Entangled Succinct Ternary Execution Engine & OZIP Z3-Triality Runtime",
    author="Matthias Asare-Darko",
    packages=["ttb2"],
    ext_modules=[ext],
)
