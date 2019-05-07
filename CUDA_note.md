## Note of CUDA

+ initial learning

reference: https://blog.csdn.net/breaksoftware/article/details/79302590

函数修饰： __host__, __global__, __device__

![image](https://github.com/Samaritan-Infi/CUDA_KNN/blob/master/image/function_decorate.png)

__global__ 异步执行

grid_block 示意图

对于线程设置的初始值还是不是很清楚

![image](https://github.com/Samaritan-Infi/CUDA_KNN/blob/master/image/grid_block.png)

内存修饰：__global__, __shared__, __constant__, __texture__

寄存器：在核函数内 int i即表示寄存器变量。

__global__：     全局内存，在主机函数中开辟和释放。

__shared__：     共享存储，每个block内的线程共享这个存储。

__constant__：   常量存储，只读。定义在所有函数之外，作用范围整个文件。

__texture__：    纹理存储，只读。内存不连续。

+ GPU 矩阵乘法 

reference: https://blog.csdn.net/lishuiwang/article/details/49073389

Naive Implementation On GPUs: 见 matrixMul.cu

+ 注意点

1、二维矩阵内存分配: CUDA为了让内存对齐（存储单元数的倍数），提高访问效率，不足一行的补齐，计算时使用pitch为行单位

reference: https://blog.csdn.net/susu0203/article/details/83111221

2、dim3，内置结构体，分配线程用

3、__synctheads()，当前block中的所有线程执行到此时，进行同步。

reference: https://blog.csdn.net/yu132563/article/details/52555434

4、原子操作，ATOM，atomicAdd(&i, 1);

构建锁控制共享内存

5、(void **) &， 对于二级指针，如 &a，如果初始指向int类型，即 &a === (int **)&a， (void **)&a相当于转换初始类型为通用类型

reference: https://blog.csdn.net/wcybrain/article/details/78300445

6、矩阵乘法，二维矩阵的内存存储用一维来实现，注意运算前后的复制

implementation: src/matrixMul.cu
