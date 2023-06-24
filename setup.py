from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension("trajectory_module", sources=["envs/mujoco/trajectory_solver.pyx"], language="c++")
]


setup(
    name="strikethree",
    version="0.0.1",
    install_requires=["gym==0.26.0", "pygame==2.1.0"],
    ext_modules=cythonize(extensions),
)
