try:
    from setuptools import setup, find_packages
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()
    from setuptools import setup, find_packages

setup(
      name="DML",
      version="Alpha",
      description="D's Machine Learning is a machine learning toolkit for python,focus on rightness but efficiency",
      author="DarkScope",
      author_email="just.dark@qq.com",
      url="https://github.com/justdark/dml",
      license="WTFPL",
      packages= find_packages(exclude=["tests","*.npz","*.csv","\.{0}"])
      )