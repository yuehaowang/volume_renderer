#pragma once
#include <string>
#include "camera.hpp"
#include "implicit_geom.hpp"
#include "light.hpp"
#include "classifier.hpp"
#include "config.hpp"


/**
 * Scene settings for rendering
 */

struct MainScene
{
    Camera** main_camera;
    ImplicitGeometry** geometry;
    Light** lights;
    int count_lights;
    Classifier** classifiers;
    int count_classifiers;
    Classifier** using_classifier;
    bool scene_opened;
    bool save_next_frame;
    std::string result_save_path;

    MainScene();
    ~MainScene();
    void updateConfiguration(RenderingConfig* c);
    void openScene(const std::string& name);
    void closeScene();
    void saveRenderingResult(const std::string& save_path);
    void processRenderingResult(unsigned char* bytes);
};