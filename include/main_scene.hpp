#pragma once
#include "camera.hpp"
#include "implicit_geom.hpp"
#include "light.hpp"
#include "classifier.hpp"


struct MainScene
{
    Camera** main_camera;
    ImplicitGeometry** geometry;
    Light** lights;
    Classifier** classifier;

    MainScene();
};