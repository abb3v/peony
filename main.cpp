#include <iostream>
#include <opencv2/opencv.hpp>
#include <unistd.h>
#include <vector>

int usage(char *argv[]) {
    std::cerr << "usage: " << argv[0]
              << " [-t threshold_factor] <input> [output=dithered.ext]"
              << std::endl;
    return 1;
}

int main(int argc, char *argv[]) {
    int opt;
    float threshold_factor = 1.0;

    while ((opt = getopt(argc, argv, "t:")) != -1) {
        switch (opt) {
        case 't':
            threshold_factor = std::atof(optarg);
            break;
        default:
            return usage(argv);
        }
    }

    if (optind >= argc) {
        return usage(argv);
    }

    std::string input_path = argv[optind];
    std::string output_path = (optind + 1 < argc) ? argv[optind + 1] : "dithered." + input_path.substr(input_path.find_last_of(".") + 1);

    cv::Mat image = cv::imread(input_path);
    if (image.empty()) {
        std::cerr << "error: could not open image file" << std::endl;
        return 1;
    }

    cv::Mat matrix = (cv::Mat_<float>(4, 4) << 0, 8, 2, 10,
                                                12, 4, 14, 6,
                                                 3, 11, 1, 9,
                                                15, 7, 13, 5);
    matrix /= 16.0;

    int rows = image.rows;
    int cols = image.cols;
    std::vector<uchar> thresholds(16);
    for (int i = 0; i < 16; ++i) {
        thresholds[i] = static_cast<uchar>(matrix.at<float>(i / 4, i % 4) * 255 * threshold_factor);
    }

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
            float grayscale = 0.2126f * pixel[2] + 0.7152f * pixel[1] + 0.0722f * pixel[0];
            uchar threshold = thresholds[(y % 4) * 4 + (x % 4)];
            uchar dithered = (grayscale > threshold) ? 255 : 0;
            image.at<cv::Vec3b>(y, x) = cv::Vec3b(dithered, dithered, dithered);
        }
    }

    cv::imwrite(output_path, image);
    image.release();

    return 0;
}
