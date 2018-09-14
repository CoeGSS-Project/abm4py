#!/bin/bash

echo "#####Function tests:####"
TEST="00_mpi4py_test.py"
echo "Test of $TEST"
if mpirun -np 4 python function/$TEST > funcTest.test; then
echo "Success"
else
echo "Error in $TEST"
fi

TEST="01_h5py_test.py"
echo "Test of $TEST"
if python function/$TEST >> funcTest.test; then
echo "Success"
else
echo "Error in $TEST"
fi


TEST="02_h5py_parallel_test.py"
echo "Test of $TEST"
if python function/$TEST >> funcTest.test; then
echo "Success"
else
echo "Error in $TEST"
fi
mpirun -np 4 python  scaling_test.py

TEST="03_framework_test.py"
echo "Test of $TEST"
if mpirun -np 2 python function/$TEST >> funcTest.test; then
echo "Success"
else
echo "Error in $TEST"
fi

echo "##### Speed tests:####"
TEST="01_move_test.py"
echo "Test of $TEST"
if python speed/$TEST > speedTest.test; then
echo "Success"
else
echo "Error in $TEST"
fi

TEST="02_id_referencing_test.py"
echo "Test of $TEST"
if python speed/$TEST > speedTest.test; then
echo "Success"
else
echo "Error in $TEST"
fi

TEST="03_delete_test.py"
echo "Test of $TEST"
if python speed/$TEST > speedTest.test; then
echo "Success"
else
echo "Error in $TEST"
fi

TEST="04_agent_filter_test.py"
echo "Test of $TEST"
if python speed/$TEST > speedTest.test; then
echo "Success"
else
echo "Error in $TEST"
fi

TEST="05_peer_aggregate_test.py"
echo "Test of $TEST"
if python speed/$TEST > speedTest.test; then
echo "Success"
else
echo "Error in $TEST"
fi


TEST="06_attribute_assignment_test.py"
echo "Test of $TEST"
if python speed/$TEST > speedTest.test; then
echo "Success"
else
echo "Error in $TEST"
fi

TEST="07_world_getAttr_test.py"
echo "Test of $TEST"
if python speed/$TEST > speedTest.test; then
echo "Success"
else
echo "Error in $TEST"
fi

TEST="08_spatial_iteration_test.py"
echo "Test of $TEST"
if python speed/$TEST > speedTest.test; then
echo "Success"
else
echo "Error in $TEST"
fi

