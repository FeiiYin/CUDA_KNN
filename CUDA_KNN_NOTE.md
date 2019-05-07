## CUDA_KNN 编写的一些注意点

+ 理论

reference: https://blog.csdn.net/time_runner/article/details/39120285

GPU环境下，不能使用递归，不能使用快排进行线性选择

k小的时候使用插入排序（只排序前k个元素），k大的时候使用梳排序

![image](https://github.com/Samaritan-Infi/CUDA_KNN/blob/master/image/sort_cmp.png)

+ 梳排序

时间复杂度 O(n*logn)，不稳定的排序算法

reference: https://www.cnblogs.com/kkun/archive/2011/11/23/2260293.html

每次交换固定距离长度两端的数值，固定长度初始为数组长度，固定长度=固定长度/1.3，遍历枚举至长度为1

思想是使逆序的元素尽可能快地移动到最终的位置，而不是像冒泡排序那样每次交换只移动一个位置。

+ 代码

reference: https://github.com/vincentfpgarcia/kNN-CUDA

reference: https://blog.csdn.net/j98355/article/details/52786104

src/knncuda.cu

line 84 用子矩阵算距离的时候，每算一个需要同步

line 159 这是初始偏离，相当于是第一行的偏离作为起点，后面算点就不用a[i*pitch+w]就可以把w省略，减少计算

line 173 使用大小为k的插入排序，选出前k个元素，这个不用同步

line 203 compute_sort 排序排所有的平方的数，距离算的根号只对前k个算即可
