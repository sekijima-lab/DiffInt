import sys
from setuptools import setup,Extension
flags=[] if sys.platform=='win32' else ['-O2','-fno-fast-math','-ffp-contract=off']
setup(name='diffint-runtime',version='0.1.0',packages=['diffint_runtime'],
      ext_modules=[Extension('diffint_runtime._legacy_math',['diffint_runtime/_legacy_math.c'],extra_compile_args=flags)],
      python_requires='>=3.12')
