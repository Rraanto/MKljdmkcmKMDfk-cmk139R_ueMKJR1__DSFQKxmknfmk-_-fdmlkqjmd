/*
 * Set up initial avec mini application de capture vidéo
 */

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <limits>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "callbacks/reconnaissance.h"
#include "callbacks/tests.h"
#include "color_distribution.h"
#include <vector>

#define FPS 60

using namespace cv;
using namespace std;

int main(int ragc, char **argv) {
  Mat img_input, img_seg, img_d_bgr, img_d_hsv, img_d_lab;

  VideoCapture *pCap = nullptr;
  const int width = 640;
  const int height = 480;
  const int size = 50;

  // ouvre la cam
  pCap = new VideoCapture(0);
  if (!pCap->isOpened()) {
    cout << "Ne peux pas ouvrir la caméera";
    return 1;
  }

  /*
   * Setup de l'image sur lequel afficher le flux vidéo
   */

  pCap->set(CAP_PROP_FRAME_WIDTH, 640);
  pCap->set(CAP_PROP_FRAME_HEIGHT, 480);

  (*pCap) >> img_input;
  if (img_input.empty()) {
    cout << "Problème avec la caméra (ouverte)";
    return 1;
  }

  Point pt1(width / 2 - size / 2, height / 2 - size / 2);
  Point pt2(width / 2 + size / 2, height / 2 + size / 2);

  namedWindow("input", 1);
  imshow("input", img_input);

  bool freeze = false;
  Mat frozen; // pour restaurer l'image précédente
              // quand on teste les rectangles NB

  /* points d'encrages pour détecter gauche droite */
  const Point top_left(0, 0);
  const Point bottom_center(width / 2, height - 1);
  const Point middle_top(width / 2, 0);
  const Point bottom_right(width - 1, height - 1);

  /* Pour stocker histogrammes fond et objets */
  std::vector<ColorDistribution> col_hists;
  std::vector<ColorDistribution> col_hists_object;

  /* Pour le test particulier de blocs NB */
  const Rect roiGH(0, 0, width / 2, height / 2);          // gauche haut
  const Rect roiGB(0, height / 2, width / 2, height / 2); // gauche bas
  const Rect roiDH(width / 2, 0, width / 2, height / 2);  // droite haut
  const Rect roiDB(width / 2, height / 2, width / 2,
                   height / 2); // droite bas
  const Mat noir(height / 2, width / 2, CV_8UC3, Vec3b(0, 0, 0));
  const Mat blanc(height / 2, width / 2, CV_8UC3, Vec3b(255, 255, 255));

  /*
   * Interrupteurs pour switcher entre différents "modes"
   */
  bool config_1 = true; // pour alterner entre les configurations

  /* pour la mode reconnaissance */
  bool reco = false;
  std::vector<Vec3b> colors;
  colors.push_back(Vec3b(0, 0, 0));
  colors.push_back(Vec3b(0, 0, 255));

  /*
   * Boucle principale
   */
  while (true) {
    // 1000 ms -> FPS images
    // ? ms    <- 1 image
    // => 1000 / FPS
    char c = (char)waitKey(1000 / FPS); // attends 50ms -> 20 images/s

    if (pCap != nullptr && !freeze) {
      (*pCap) >> img_input; // recupere le flux
      cv::flip(img_input, img_input, 1);
    }

    if (c == 27 || c == 'q')
      break;

    if (c == 'f') { // geler l'image
      freeze = !freeze;
      if (freeze)
        frozen = img_input.clone();
    }

    if (c == 'v') {
      /*
       * Calcule la différence de couleurs entre région gauche et droite
       * points d'ancrage top_left, bottom_center définies en dehors de la
       * boucle
       */
      calcul_distance_gauche_droite(img_input, top_left, bottom_center,
                                    middle_top, bottom_right);
    }

    // Histogrammes du fond
    if (c == 'b') {
      calcul_histogrammes_fond(col_hists, img_input);
    }

    // calcul de l'histogramme de l'objet
    if (c == 'a') {
      ajout_histogramme_objet(col_hists_object, img_input, pt1, pt2);
    }

    if (c == 'x' && freeze) {
      /*
       * pour test particuliers d'images décomposées en noire et blanc
       * UNIQUEMENT QUAND FREEZE
       * Calcule la différence de couleurs région droite/région gauche
       */
      test_blocs_nb_et_distance(img_input, frozen, roiGH, roiGB, roiDH, roiDB,
                                noir, blanc, config_1, top_left, bottom_center,
                                middle_top, bottom_right);
    }

    if (c == 'r') {
      if (col_hists.empty())
        cout << "Pas d'histogrammes fond" << endl;

      else if (col_hists_object.empty())
        cout << "Pas d'histogrammes objets" << endl;
      else {
        reco = !reco;
      }
    }

    Mat out = img_input;

    // choix d'image à afficher selon mode reco ou non
    if (reco) {

      Mat gray;
      cvtColor(img_input, gray, COLOR_BGR2GRAY);

      Mat reco_img =
          recoObject(img_input, col_hists, col_hists_object, colors, 8);

      cvtColor(gray, img_input, COLOR_GRAY2BGR);
      out = 0.5 * reco_img + 0.5 * img_input;
    } else {
      cv::rectangle(img_input, pt1, pt2, Scalar({255.0, 255.0, 255.0}), 1);
    }
    imshow("input", out);
  }
}
