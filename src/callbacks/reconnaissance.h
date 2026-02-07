#ifndef RECONNAISSANCE_H
#define RECONNAISSANCE_H

#include <opencv2/core.hpp>
#include <vector>

#include "color_distribution.h"

void calcul_histogrammes_fond(std::vector<ColorDistribution> &col_hists,
                              const cv::Mat &img_input);

void ajout_histogramme_objet(std::vector<ColorDistribution> &col_hists_object,
                             const cv::Mat &img_input,
                             const cv::Point &haut_gauche,
                             const cv::Point &bas_droite);

cv::Mat
recoObject(cv::Mat input,
           const std::vector<std::vector<ColorDistribution>> &all_col_hists,
           const std::vector<cv::Vec3b> &colors, const int bloc);

#endif
