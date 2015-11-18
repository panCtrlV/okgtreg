from setuptools import setup

setup(
    name='okgtreg',
    version='0.1',
    description='Implementation of Optimal Kernel Group Optimization for regression',
    classifiers=[
      'Development Status :: 3 - Alpha',
      'Programming Language :: Python :: 2.7',
    ],
    keywords='OKGT regression exploratory data analysis',
    url='http://www.stat.purdue.edu/~panc',
    author='Pan Chao',
    authro_email='panc@purdue.edu',
    packages=['okgtreg'],
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn',
        'pickle'
    ],
    include_package_data=True,
    zip_safe=False,
    test_suite='nose.collector',
    tests_require=['nose']
)