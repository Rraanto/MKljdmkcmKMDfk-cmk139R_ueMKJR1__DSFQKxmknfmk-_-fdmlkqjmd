#ifndef TESTS_H
#define TESTS_H

#include <opencv2/core.hpp>

void calcul_distance_gauche_droite(const cv::Mat &img_input,
                                   const cv::Point &top_left,
                                   const cv::Point &bottom_center,
                                   const cv::Point &middle_top,
                                   const cv::Point &bottom_right);

void test_blocs_nb_et_distance(
    cv::Mat &img_input, const cv::Mat &frozen, const cv::Rect &roiGH,
    const cv::Rect &roiGB, const cv::Rect &roiDH, const cv::Rect &roiDB,
    const cv::Mat &noir, const cv::Mat &blanc, bool &config_1,
    const cv::Point &top_left, const cv::Point &bottom_center,
    const cv::Point &middle_top, const cv::Point &bottom_right);

#endif
