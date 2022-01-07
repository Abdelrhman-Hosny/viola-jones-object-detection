from numba import cuda,vectorize
import numpy as np
# to measure exec time
from timeit import default_timer as timer  
 
# normal function to run on cpu
def func(a):                               
    for i in range(10000000):
        a[i]+= 1     
 
# function optimized to run on gpu
@cuda.jit                   
def func2(a):
    for i in range(10000000):
        a[i]+= 1




@cuda.jit('void(float32[:], float32[:], float32[:])')
def cu_add2(a, b, c):
    """This kernel function will be executed by a thread."""
    i  = cuda.grid(1)
    if i > c.shape[0]:
        return

    c[i] = a[i]+b[i]



@vectorize(['int64(int64, int64)',
            'float32(float32, float32)',
            'float64(float64, float64)'])
def cu_add(a, b):
    return a + b


@vectorize(['int64(int64, int64)',
            'float32(float32, float32)',
            'float64(float64, float64)'])
def cu_vec_add_2d(a, b):
    return a + b

if __name__=="__main__":

    
    n = 10000000                           
    a = np.ones(n, dtype = np.float64)
    b = np.ones(n, dtype = np.float32)
     
    
     
    start = timer()
   # threadsperblock = 1
   # blockspergrid = (a.size + (threadsperblock - 1)) // threadsperblock
    #func2[blockspergrid, threadsperblock](a)
   # print("with GPU:", timer()-start)
    device = cuda.get_current_device()
    n = 100
    a = np.arange(n, dtype=np.float32)
    b = np.arange(n, dtype=np.float32)
    c = np.empty_like(a)

    tpb = device.WARP_SIZE
    bpg = int(np.ceil(float(n)/tpb))
    print('Blocks per grid:', bpg)
    print('Threads per block', tpb)

    cu_add2[bpg, tpb](a, b, c)
    print(c)
    print("GPU: jit", timer()-start)  


    n = 10000000                           
    a = np.ones(n, dtype = np.float64)
    start = timer()
    #func(a)
    print("without GPU:", timer()-start)   

    a = [[1,2,3],[3,1,4]]
    start = timer()
    da = cuda.to_device(a)
    bpg = int(np.ceil(float(n)/tpb))
    #func2[bpg, tpb](da)
    print("with GPU:", timer()-start)

    start = timer()
    n = 100
    a = np.arange(n, dtype=np.float32)
    b = np.arange(n, dtype=np.float32)
    c = cu_add(a, b)
    print(c)

    print("GPU: vectorise", timer()-start)  

    start = timer()
    n = 480
    p = 320
    a = np.random.random((n, p)).astype(np.float32)
    b = np.ones((n, p)).astype(np.float32)
    c= cu_vec_add_2d(a, b)
    print (a[-5:, -5:])
    print (b[-5:, -5:])
    print (c[-5:, -5:])
    print("GPU: vectorise", timer()-start) 

    gpu = cuda.get_current_device()
    print(gpu)