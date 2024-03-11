import multiprocessing
import os
import platform
import re
import subprocess
import sys
from distutils.version import LooseVersion

from setuptools import Extension
from setuptools import find_packages
from setuptools import setup
from setuptools.command.build_ext import build_ext

# Extension based on the example below (MIT License).
# Modified slightly to support the extra CMake argument.
# https://github.com/benjaminjack/python_cpp_example


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r"version\s*([\d.]+)", out.decode()).group(1))
            if cmake_version < "3.1.0":
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        # This dependency may be installed after this script is executed, so we do not import
        # at the top-level of this script and instead import after any installation has completed.
        import torch

        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            "-DPYTHON_EXECUTABLE=" + sys.executable,
            "-DCMAKE_PREFIX_PATH=" + torch.__path__[0],
            "-DCMAKE_INSTALL_PREFIX=" + extdir,  # using install prefix makes it easier to control RPATH if necessary
        ]

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
        # use several cpus if we can
        build_args += ["--", "-j{}".format(multiprocessing.cpu_count() // 4 + 1)]

        env = os.environ.copy()
        env["CXXFLAGS"] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get("CXXFLAGS", ""), self.distribution.get_version())

        print(cmake_args)
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(
            ["cmake", "--build", ".", "--target", "install"] + build_args,
            cwd=self.build_temp,
        )
        print()  # Add an empty line for cleaner output


setup(
    name="torchscript",
    version="1.0.0",
    description="Custom operator registry for TorchScript at Pinterest.",
    # long_description=readme(),
    author="Pinterest Engineering",
    install_requires=["torch>1.6.0", "cmake>=3.17"],
    # tell setuptools to look for any packages under 'python'
    packages=find_packages("python"),
    # tell setuptools that all packages will be under the 'python' directory
    package_dir={"": "python"},
    # build extensions 'operators under the package 'torchscript'
    ext_modules=[CMakeExtension("torchscript/")],
    # add custom build_ext command
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
