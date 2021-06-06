#pragma once
#include <glad/glad.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include "config.hpp"
#include "main_scene.hpp"


/**
 * ImGui-based use interface
 */

class GUI
{
protected:
    GLuint tex_ID;
    RenderingConfig* render_settings;
    MainScene* scene;

    void drawVisualizer();
    void drawControls();
    void drawMenuBar();
    void drawCameraPanel();
    void drawLightPanel();
    void drawRaycastingPanel();
    void drawTransferFunctionPanel();
    void drawPerformancePanel();

public:
    GUI(MainScene* scn, GLuint tex, RenderingConfig* c);
    void draw();
};