#!/usr/bin/env bash



# Usage (run from ./):
# ./run-backend-api-tests.sh
# ./run-backend-api-tests.sh [tests]...
#
# Example:
# ./run-backend-api-tests.sh tasks.mgr.dashboard.test_pool.DashboardTest
#
# Or source this script. Allows to re-run tests faster:
# $ source run-backend-api-tests.sh
# $ run_teuthology_tests [tests]...
# $ cleanup_teuthology

# creating temp directory to store virtualenv and teuthology

get_cmake_variable() {
    local variable=$1
    grep "$variable" CMakeCache.txt | cut -d "=" -f 2
}

setup_teuthology() {
    TEMP_DIR=`mktemp -d`

    CURR_DIR=`pwd`
    BUILD_DIR="$CURR_DIR/../../../../build"

    read -r -d '' TEUTHOLOFY_PY_REQS <<EOF
apache-libcloud==2.2.1 \
asn1crypto==0.22.0 \
bcrypt==3.1.4 \
certifi==2018.1.18 \
cffi==1.10.0 \
chardet==3.0.4 \
configobj==5.0.6 \
cryptography==2.1.4 \
enum34==1.1.6 \
gevent==1.2.2 \
greenlet==0.4.13 \
idna==2.5 \
ipaddress==1.0.18 \
Jinja2==2.9.6 \
manhole==1.5.0 \
MarkupSafe==1.0 \
netaddr==0.7.19 \
packaging==16.8 \
paramiko==2.4.0 \
pexpect==4.4.0 \
psutil==5.4.3 \
ptyprocess==0.5.2 \
pyasn1==0.2.3 \
pycparser==2.17 \
PyNaCl==1.2.1 \
pyparsing==2.2.0 \
python-dateutil==2.6.1 \
PyYAML==3.12 \
requests==2.18.4 \
six==1.10.0 \
urllib3==1.22
EOF



    cd $TEMP_DIR

    virtualenv --python=/usr/bin/python venv
    source venv/bin/activate
    eval pip install $TEUTHOLOFY_PY_REQS
    pip install -r $CURR_DIR/requirements.txt
    deactivate

    git clone https://github.com/ceph/teuthology.git

    cd $BUILD_DIR

    CEPH_MGR_PY_VERSION_MAJOR=$(get_cmake_variable MGR_PYTHON_VERSION | cut -d '.' -f1)
    if [ -n "$CEPH_MGR_PY_VERSION_MAJOR" ]; then
        CEPH_PY_VERSION_MAJOR=${CEPH_MGR_PY_VERSION_MAJOR}
    else
        if [ $(get_cmake_variable WITH_PYTHON2) = ON ]; then
            CEPH_PY_VERSION_MAJOR=2
        else
            CEPH_PY_VERSION_MAJOR=3
        fi
    fi

    export COVERAGE_ENABLED=true
    export COVERAGE_FILE=.coverage.mgr.dashboard

    MGR=2 RGW=1 ../src/vstart.sh -n -d
    sleep 10
    cd $CURR_DIR
}

run_teuthology_tests() {
    cd "$BUILD_DIR"
    source $TEMP_DIR/venv/bin/activate


    if [ "$#" -gt 0 ]; then
      TEST_CASES=""
      for t in "$@"; do
        TEST_CASES="$TEST_CASES $t"
      done
    else
      TEST_CASES=`for i in \`ls $BUILD_DIR/../qa/tasks/mgr/dashboard/test_*\`; do F=$(basename $i); M="${F%.*}"; echo -n " tasks.mgr.dashboard.$M"; done`
      TEST_CASES="tasks.mgr.test_dashboard $TEST_CASES"
    fi

    export PATH=$BUILD_DIR/bin:$PATH
    export LD_LIBRARY_PATH=$BUILD_DIR/lib/cython_modules/lib.${CEPH_PY_VERSION_MAJOR}/:$BUILD_DIR/lib
    export PYTHONPATH=$TEMP_DIR/teuthology:$BUILD_DIR/../qa:$BUILD_DIR/lib/cython_modules/lib.${CEPH_PY_VERSION_MAJOR}/
    eval python ../qa/tasks/vstart_runner.py $TEST_CASES

    deactivate
    cd $CURR_DIR
}

cleanup_teuthology() {
    cd "$BUILD_DIR"
    killall ceph-mgr
    sleep 10
    ../src/stop.sh
    sleep 5

    cd $CURR_DIR
    rm -rf $TEMP_DIR

    unset TEMP_DIR
    unset CURR_DIR
    unset BUILD_DIR
    unset setup_teuthology
    unset run_teuthology_tests
    unset cleanup_teuthology
}

setup_teuthology

# End sourced section
return 2> /dev/null

run_teuthology_tests "$@"
cleanup_teuthology
