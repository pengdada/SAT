Efficient Algorithms for the Summed Area Tables Primitive on GPUs
======================
Register Cache based SAT computation


## Description
Summed-Area-Tables (SAT), namely Integral Image, computation on CUDA-enabled GPUs

Functions:

1, ScanRow-BRLT

(another name BlockScan in source code)

2, BRLT-ScanRow

(another name SerialScan in source code)

3, ScanRowColumn

(another name IncreamentScan in source code)

## Installation

requirements : Linux , CUDA 9.0 (or above) Toolkit, cmake (2.8 or above)

Note : 

       (1) using CUDA 9.2, our three algorithms become faster than using CUDA 9.0, but OpenCV performance do not change. 
       (2) The latest DGX-1 server only support CUDA 9.0.

(step 1) go to folder SAT/integralimage/

(step 2) make

(step 3) exeucution file is "./release/sat"

## Test scripts
all of the algorithms are implemented by c++ template function, so any data type can be used, e.g. u8, u32, f32, double......

- [Test for ScanRow-BRLT algorithm]  
  ./test0.sh 
        
- [Test for BRLT-ScanRow algorithm]  
  ./test1.sh 
        
- [Test for ScanRowColumn algorithm]  
  ./test2.sh
     
## Arguments
  
  please refer to "SAT/para.png"
  
  ![Alt text](https://github.com/pengdada/SAT/blob/master/para.png?raw=true "arguments")
  
