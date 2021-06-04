#pragma once
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
    Classifier** classifier;

    MainScene();
    ~MainScene();
    void updateConfiguration(RenderingConfig* c);
};