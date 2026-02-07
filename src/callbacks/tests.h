#ifndef TESTS_H
#define TESTS_H

#include <opencv2/core.hpp>

void calcul_distance_gauche_droite(const cv::Mat &img_input);

void test_blocs_nb_et_distance(cv::Mat &img_input, const cv::Mat &frozen,
                               bool &config_1);

#endif
