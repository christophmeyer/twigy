#!/bin/bash
docker run -t -v "$PWD:/io" cmeyr/manylinux2014_x86_64_boost166:latest /app/build_wheel_docker.sh 
