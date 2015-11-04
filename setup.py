__author__ = 'panc'


from setuptools import setup

setup(
    name='okgtreg_primitive',
    version='0.1',
    description='Implementation of Optimal Kernel Group Optimization for regression',
    classifiers=[
      'Development Status :: 3 - Alpha',
      'Programming Language :: Python :: 2.7',
    ],
    keywords='OKGT regression exploratory data analysis',
    url='http://github.com/...',
    author='panc',
    authro_email='panc@purdue.edu',
    packages=['okgtreg_primitive'],
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn'
    ],
    include_package_data=True,
    zip_safe=False,
    test_suite='nose.collector',
    tests_require=['nose']
)