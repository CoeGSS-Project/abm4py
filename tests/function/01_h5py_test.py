import h5py

f = h5py.File('test.hdf5', 'w')
dset = f.create_dataset('test', (100,), dtype='i')
for i in range(100):
    dset[i] = i
    
f.close()
