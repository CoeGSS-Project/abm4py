# ABM4py

Agent-based modeling framework for python

0.7.3 stable version

## Getting Started

This framework allows to develop parallel prototypes of agent-based models.
It allows for quick development of ABM models that scale up to a few hundeds 
of parallel cores, supports parallel IO and aggregation.

The current version only support limited partitioning methods, but we aim for
extensions.

### Prerequisites

```
python > 3.x
numpy >= 1.14
pandas 
h5py
matplotlib

OPTIONAL:
mpi4py for parallel execution
h5py compiled in parallel mode (see: http://docs.h5py.org/en/latest/build.html#building-against-parallel-hdf5)
numba for optinal compiled function execution
```

### Installing

git clone https://abm4py@bitbucket.org/abm4py/abm4py.git

## Running the tests

sh tests/run_test.sh

## Deployment

This Framework is tested using Anaconda 3. Thus, follow the standart installation
procedure fo Andaconda and add missing packages using: "conda install PACKAGE_NAME"

## Authors

- Andreas Geiges 
- Steffen Fuerst
- Sarah Wolf
- Gesine Steudle

## Contributing

- Jette von Postel


## License

This project is licensed under the GNU Lesser General Public License as published 
by the Free Software Foundation, version 3 only. - see the file headers for details.

## Acknowledgments

This framework was supported and partly funded by the EU Horizon 2020 framework programme project 
CoeGSS (No. 676547), which is gratefully acknowledged.
