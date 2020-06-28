FROM quay.io/pypa/manylinux2014_x86_64

RUN /opt/python/cp36-cp36m/bin/pip3.6 install cmake
RUN ln -s /opt/python/cp36-cp36m/bin/cmake /usr/bin/cmake

RUN yum install -y http://repo.okay.com.mx/centos/7/x86_64/release/okay-release-1-1.noarch.rpm
RUN yum install -y boost166-devel

COPY ./build_wheel_docker.sh /app/build_wheel_docker.sh