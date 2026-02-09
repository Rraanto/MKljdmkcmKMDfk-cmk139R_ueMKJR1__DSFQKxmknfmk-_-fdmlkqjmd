#include "tests.h"

#include <iostream>

#include "color_distribution.h"

using namespace cv;
using namespace std;

void calcul_distance_gauche_droite(const Mat &img_input) {
  /*
   * Calcule la distance entre couleurs d'une partie "gauche" et "droite"
   * définies par des points d'encrages bottom_center, middle_top,
   * bottom_right, top_left
   */
  const int width = img_input.cols;
  const int height = img_input.rows;
  const Point top_left(0, 0);
  const Point bottom_center(width / 2, height - 1);
  const Point middle_top(width / 2, 0);
  const Point bottom_right(width - 1, height - 1);

  ColorDistribution left_hist;
  left_hist =
      left_hist.getColorDistribution(img_input, top_left, bottom_center);
  left_hist.finished();

  ColorDistribution right_hist;
  right_hist =
      right_hist.getColorDistribution(img_input, middle_top, bottom_right);
  right_hist.finished();

  float d = left_hist.distance(right_hist);
  cout << "Distance de couleurs: " << d << endl;
}

void test_blocs_nb_et_distance(Mat &img_input, const Mat &frozen,
                               bool &config_1) {
  /*
   * effectue le test sur l'image partiulière de blocs noir et blancs
   */
  if (frozen.empty())
    return;

  img_input = frozen.clone();

  const int width = img_input.cols;
  const int height = img_input.rows;
  const Rect roiGH(0, 0, width / 2, height / 2);                  // gauche haut
  const Rect roiGB(0, height / 2, width / 2, height / 2);         // gauche bas
  const Rect roiDH(width / 2, 0, width / 2, height / 2);          // droite haut
  const Rect roiDB(width / 2, height / 2, width / 2, height / 2); // droite bas
  const Mat noir(height / 2, width / 2, CV_8UC3, Vec3b(0, 0, 0));
  const Mat blanc(height / 2, width / 2, CV_8UC3, Vec3b(255, 255, 255));

  blanc.copyTo(img_input(roiGH));
  blanc.copyTo(img_input((config_1) ? roiDB : roiDH));

  noir.copyTo(img_input(roiGB));
  noir.copyTo(img_input((config_1) ? roiDH : roiDB));

  config_1 = !config_1;

  calcul_distance_gauche_droite(img_input);
}
