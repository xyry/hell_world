# Learn_libtorch

## 官方例子
example-app 是官方的例子,现在已经跑通.官方例子地址:https://pytorch.org/cppdocs/installing.html#minimal-example
### 遇到的问题
example-app.cpp 自己手敲代码,make一直报错.但是把官方的代码复制粘贴上去,make就成功了,不明白哪里出了问题.

编译指令
```shell
# 最后有两个点,不要漏掉
# cpu libtorch 路径
/home/ypl/Desktop/learn_libpytorch/libtorch
# gpu cuda9.0 路径
/home/ypl/Downloads/libtorch_cu90
cmake -DCMAKE_PREFIX_PATH=/home/ypl/Desktop/learn_libpytorch/libtorch ..
make
#cmake --build . --config Release

#运行指令,后面是 deeplabv3+ 模型路径
./deeplab /home/ypl/dataset/pytorch_model/deeplabv3+.pt
```
##　libtorch版本问题
下载的libtorch版本的时候，向下兼容，比如下载　cuda10.1对应的版本,https://download.pytorch.org/libtorch/cu101/libtorch-shared-with-deps-1.5.**1**%2Bcu101.zip
在Cuda10.0上是可以运行的.但是如果你下的是cuda9.0 本机是cuda10.0 就会报错.

## libtorch 调用cuda
RuntimeError: Input type (CUDAFloatType) and weight type (CPUFloatType) should be the same
这个报错告诉我们,将输入和模型都应该转为CUDA类型
```c++
// 设置device为cuda
torch::Device device = torch::kCPU;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Training on GPU." << std::endl;
        device = torch::kCUDA;
    }
    std::cout<<device<<std::endl;
//模型以cuda形式加载
module = torch::jit::load(argv[1],device);
//输入数据以cuda形式创建

```
## 出现的问题

1. OpenCV Error: Assertion failed (ssize.width > 0 && ssize.height > 0) in resize, file /tmp/binarydeb/ros-kinetic-opencv3-3.3.1/modules/imgproc/src/resize.cpp, line 3939
terminate called after throwing an instance of 'cv::Exception'
  what():  /tmp/binarydeb/ros-kinetic-opencv3-3.3.1/modules/imgproc/src/resize.cpp:3939: error: (-215) ssize.width > 0 && ssize.height > 0 in function resize
  > 解决办法: 检查传入的图片路径是否写对,这个报错是因为没有读到图片,所以没有办法resize
2. torch::Double can’t be casted to double directly, you should use output_tensor[i][j][k].item().to<double>(). It first converts a Tensor with a single value to a ScalarType then to a double.

3.CUDA运算的数据不能转为为Eigen,要先将Tensor转为cpu类型,才可以转 


 # 参考资料
 https://pytorch.org/tutorials/advanced/cpp_frontend.html
 https://blog.csdn.net/u010397980/article/details/89437628
 https://www.jianshu.com/p/9e8eb211df62
 
 1. 关于libtorch的问题,可以在下面这个网址提问或者搜索答案
 https://discuss.pytorch.org/t/type-conversion-in-libtorch-for-c/45139
 
 5. mat->opencv可视化
 https://blog.csdn.net/qq_34917728/article/details/84502004?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.nonecase&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.nonecase
 
 6. Data transfer between LibTorch C++ and Eigen
 https://discuss.pytorch.org/t/data-transfer-between-libtorch-c-and-eigen/54156
