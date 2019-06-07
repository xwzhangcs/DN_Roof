#pragma once
#include <vector>
#include <map>
#include <tuple>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/filewritestream.h"

namespace util {
	double readNumber(const rapidjson::Value& node, const char* key, double default_value);
	std::vector<double> read1DArray(const rapidjson::Value& node, const char* key);
	bool readBoolValue(const rapidjson::Value& node, const char* key, bool default_value);
	std::string readStringValue(const rapidjson::Value& node, const char* key);
	bool AddGaussianNoise_Opencv(const cv::Mat mSrc, cv::Mat &mDst, double Mean, double StdDev);
}