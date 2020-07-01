#include<torch/script.h>
#include <torch/torch.h>
#include<iostream>
#include<memory>
#include <ATen/ATen.h>

#include <string>
#include <chrono>
 
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"
#include <fstream>
#include<Eigen/Eigen>
#include <unsupported/Eigen/CXX11/Tensor>
using namespace std;
/*
argc 参数数量.
argv 参数列表,以字符串存储.
*/
int main(int argc,const char* argv[]){
    //读取参数
    if(argc!=2){
	    std::cerr<<"usage: deeplab <path-to-.pt module>\n";
	    return -1;
    }
    //选择设备
    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Training on GPU." << std::endl;
        device = torch::kCUDA;
    }
    // std::cout<<device<<std::endl;
    //调用gpu成功

    //创建一个模型
    torch::jit::script::Module module;
    try {
        //加载模型
        module = torch::jit::load(argv[1],device);
    }catch (const c10::Error& e){
        std::cerr<<"error loading the module\n";
        return -1;
    }
    

    //opencv读取图片
    std::cout<<"module load success\n";
    std::string s="../1341845948.747856.png";
    cv::Mat src,image,float_image;
    src=cv::imread(s);
    //test cv::Size usage
    // std::cout<<(int)src.cols<<(int)src.rows<<std::endl;
    //处理图片
    int img_size = 224;
    resize(src,image,cv::Size(img_size,img_size));
    cv::imshow("image3",image);
    cv::cvtColor(image,image,CV_BGR2RGB);
    image.convertTo(float_image,CV_32F,1.0/255);

    auto img_tensor = torch::from_blob(float_image.data, {1, img_size, img_size, 3});
    //将cv::Mat转成tensor,大小为1,224,224,3
    img_tensor = img_tensor.permute({0, 3, 1, 2});  //调换顺序变为torch输入的格式 1,3,224,224
        //img_tensor[0][0] = img_tensor[0][0].sub_(0.485).div_(0.229);  //减去均值,除以标准差
        //img_tensor[0][1] = img_tensor[0][1].sub_(0.456).div_(0.224);
        //img_tensor[0][2] = img_tensor[0][2].sub_(0.406).div_(0.225);
        
    auto img_var = torch::autograd::make_variable(img_tensor, false);  //不需要梯度
    std::vector<torch::jit::IValue> inputs;
	inputs.emplace_back(img_var.to(at::kCUDA));  // 把预处理后的图像放入gpu
    torch::Tensor result = module.forward(inputs).toTensor();  //前向传播获取结果
    inputs.pop_back();
    // std::cout << "result:" << result << std::endl;
    //对21个224x224 取最大值,每个像素点上的值为预测概率最大的类别,类别0-20,0为背景,15为人,9为椅子
    auto pred = result.argmax(1);
    std::cout<<result.size(0)<<" "<<result.size(1)<<" "<<result.size(2)<<" "<<result.size(3)<<std::endl;
    

    //Eigen
    using Tensor3d = Eigen::Tensor<double,3,Eigen::RowMajor>;
    using Tensor4d = Eigen::Tensor<double,4,Eigen::RowMajor>;
    using Tensor3f = Eigen::Tensor<float,3,Eigen::RowMajor>;
    using Tensor4f = Eigen::Tensor<float,4,Eigen::RowMajor>;
    result = result.to(torch::kCPU);
    auto output_data =
      Eigen::TensorMap<Tensor4f>(result.data_ptr<float>(),
                                 result.size(0),
                                 result.size(1),
                                 result.size(2),
                                 result.size(3));
    Tensor4d output = output_data.cast<double>();

    // Extract mean confidence over the Monte Carlo trials
    Tensor3d mean_confidence = output.mean(Eigen::array<int, 1>({0}));
    cout<<"mean_confidence="<<mean_confidence<<endl;

    // float* data = T.data_ptr<float>();
    // Eigen::Map<Eigen::MatrixXf> E(data, T.size(0), T.size(1));

    

    //tensor 转 opencv::matrix
    // cv::Mat vis_result=cv::Mat(img_size,img_size,CV_8UC1);
    // for(int i=0;i<img_size;i++){
    //     for(int j=0;j<img_size;j++){
    //         //这种赋值成功
    //         vis_result.data[i*img_size+j]=pred[0][i][j].item().to<int>()*12;
    //     }
    // }
    // cv::imshow("image1",vis_result);
    // cv::imshow("image2",src);
    // cv::waitKey();

    // cv::Mat label_colours;
    // cv::Mat mImgSegment_color;
    // cv::Mat mImgSegment_color_final;
    // cv::Mat mImgSegment=pred[0].item().to<int>();
    // std::string LUT_file = "../pascal_png";
    // label_colours = cv::imread(LUT_file,1);
    // cv::cvtColor(label_colours, label_colours, CV_RGB2BGR);

    // cv::cvtColor(mImgSegment,mImgSegment_color, CV_GRAY2BGR);

    // LUT(mImgSegment_color, label_colours, mImgSegment_color_final);
    // cv::imshow("image3",mImgSegment_color_final);
    


    // //创建一个输入图片数组
    // std::vector<torch::jit::IValue> inputs;
    // //创建一个输入向量
    // torch::jit::IValue input=torch::ones({1,3,224,224},device);
    // //将输入向量放入输入数组
    // inputs.push_back(input);
    // //模型开始预测
    // at::Tensor output=module.forward(inputs).toTensor();
    // // 1x21x224x224
    // //输出处理
    // std::cout<<output[0][0][1][1]<<"\n";
    // //代表图片数量
    // std::cout<<output.size(0)<<"\n";
    // //代表分类的数量,有多少个类
    // std::cout<<output.size(1)<<"\n";
    // //代表图片的高和宽
    // std::cout<<output.size(2)<<"\n";
    // std::cout<<output.size(3)<<"\n";
    
}

//处理成文件
    // std::ofstream of;
    // of.open("out.txt");
    // of<<pred<<std::endl;
    // of.close();

    /*
    std::cout << "Testing LibTorch to Eigen:" << std::endl;
    // LibTorch
    // torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    torch::Tensor T = torch::rand({3, 3},device);
    
    std::cout << "LibTorch:" << std::endl;
    std::cout << T.to(torch::kCPU)<< std::endl;
    T=T.to(torch::kCPU);
    // Eigen
    float* data = T.data_ptr<float>();
    Eigen::Map<Eigen::MatrixXf> E(data, T.size(0), T.size(1));
    std::cout << "EigenMat:\n" << E << std::endl;
    // re-check after changes
    E(0,0) = 0;
    std::cout << "EigenMat:\n" << E << std::endl;
    std::cout << "LibTorch:" << std::endl;
    std::cout << T << std::endl;
    */