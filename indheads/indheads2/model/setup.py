from setuptools import setup
# Adapated from https://github.com/karpathy/minGPT

setup(name='minGPT',
      version='0.0.1',
      author='Andrej Karpathy',
      packages=['mingpt'],
      description='A PyTorch re-implementation of GPT',
      license='MIT',
      install_requires=[
            'torch',
            'bidict',
            'numpy'
      ],
)
