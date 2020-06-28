#!/bin/bash
rm -rf /io/build/**
cd /io
/opt/python/cp36-cp36m/bin/python3.6 setup.py sdist bdist_wheel
auditwheel repair --plat manylinux2014_x86_64 ./dist/*.whl
rm -rf /io/*.egg-info
mv /io/dist /io/build/
mv /io/wheelhouse /io/build/
