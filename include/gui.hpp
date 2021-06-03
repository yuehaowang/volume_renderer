#pragma once
#include <glad/glad.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include "config.hpp"


/**
 * ImGui-based use interface
 */

class GUI
{
private:
    GLuint tex_ID;
    RenderingConfig* render_settings;

    void drawVisualizer();
    void drawControls();
    void drawMenuBar();
    void drawCameraPanel();
    void drawLightPanel();

public:
    GUI(GLuint tex, RenderingConfig* c);
    void draw();
};