#define COMPILER_MSVC
#define NOMINMAX
#define PLATFORM_WINDOWS

#include <fstream>

#include <utility>
#include<vector>
#include<string>

#include<opencv2/opencv.hpp>
#include"tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/cc/ops/standard_ops.h"

using namespace tensorflow;
using tensorflow::Tensor;
using std::cout;
using std::endl;

int main() {
	// ��������ͼ��
	cv::Mat img = cv::imread("C:\\Users\\admin\\Pictures\\3.png");
	cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
	int height = img.rows;
	int width = img.cols;
	int depth = img.channels();
	cv::Mat img_transpose = img.t();

	// ȡͼ�����ݣ�����tensorflow֧�ֵ�Tensor������
	const float* source_data = (float*)img.data;
	cout<<source_data[1]<<endl;
	tensorflow::Tensor input_tensor(DT_FLOAT, TensorShape({ 1, height, width ,1 })); //����ֻ����һ��ͼƬ���ο�tensorflow�����ݸ�ʽNHWC
	auto input_tensor_mapped = input_tensor.tensor<float,4>(); // input_tensor_mapped�൱��input_tensor�����ݽӿڣ���4����ʾ������4ά�ġ�����ȡ�����ս��ʱҲ�ܿ��������÷�                                                                                                      

	// �����ݸ��Ƶ�input_tensor_mapped�У�ʵ���Ͼ��Ǳ���opencv��Mat����
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			input_tensor_mapped(0, i, j ,0) = (255- img.at<uchar>(i, j))/255.0;
			
		}
	}

	// ��ʼ��tensorflow session
	Session* session;
	Status status = NewSession(SessionOptions(), &session);
	if (!status.ok()) {
		std::cerr << status.ToString() << endl;
		return -1;
	}
	else {
		cout << "Session created successfully" << endl;
	}


	// ��ȡ�����Ƶ�ģ���ļ���graph��
	GraphDef graph_def;
	status = ReadBinaryProto(Env::Default(), "e:\\model.pb", &graph_def);
	if (!status.ok()) {
		std::cerr << status.ToString() << endl;
		return -1;
	}
	else {
		cout << "Load graph protobuf successfully" << endl;
	}


	// ��graph���ص�session
	status = session->Create(graph_def);
	if (!status.ok()) {
		std::cerr << status.ToString() << endl;
		return -1;
	}
	else {
		cout << "Add graph to session successfully" << endl;
	}

	tensorflow::Tensor keep_prob(DT_FLOAT, TensorShape());
	keep_prob.scalar<float>()() = 1.0;
	std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
		{ "x", input_tensor },
		{ "keep_prob", keep_prob },
	};

	// ���outputs
	std::vector<tensorflow::Tensor> outputs;

	// ���лỰ���������"x_predict"��������ģ���ж��������������ƣ����ս��������outputs��
	auto input_shape = input_tensor.shape();
	status = session->Run(inputs, { "y" }, {}, &outputs);
	if (!status.ok()) {
		std::cerr << status.ToString() << endl;
		system("pause");
		return -1;
	}
	else {
		cout << "Run session successfully" << endl;
	}
	// ��������������Ŀ��ӻ�
	tensorflow::Tensor output = std::move(outputs.at(0)); // ģ��ֻ���һ��������������Ȱѽ���Ƴ�����ҲΪ�˸��õ�չʾ��
	auto out_shape = output.shape(); // �����������Ϊ1x4x16
	auto out_val = output.tensor<float, 2>(); // �뿪ͷ���÷���Ӧ��3��������ά��
	// cout << out_val.argmax(2) << " "; // Ԥ��������pythonһ�£���������ֵ�в��죬�²��Ǽ��㾫�Ȳ�ͬ��ɵ�
	cout <<"Output tensor shape:"<< output.shape()<<endl;
	for (int j = 0; j < out_shape.dim_size(1); j++) {
		cout << out_val(0, j) << " ";
	}
	cout << "size:" << out_shape.dim_size(1)<<endl;
	cout <<"Prediction value:"<< out_val.argmax(1)<<endl;
	system("pause");
}