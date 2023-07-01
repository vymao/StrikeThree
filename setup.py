from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension("trajectory_module", sources=["strikethree/envs/mujoco/trajectory_solver.pyx"], language="c++")
]


setup(
    name="strikethree",
    version="0.0.1",
    packages = find_packages(),
    install_requires=[
        "pygame==2.1.0",
    ],
    ext_modules=cythonize(extensions),
)
