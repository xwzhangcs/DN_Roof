#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <memory>
#include <Windows.h>

/**** helper functions *****/
std::vector<std::string> get_all_files_names_within_folder(std::string folder);
/**** steps *****/
std::vector<double> feedDnn(std::string img_filename, std::string modeljson, bool bDebug);
