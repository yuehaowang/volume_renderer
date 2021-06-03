#include "gui.hpp"


/**
 * GUI class
 */

GUI::GUI(GLuint tex, RenderingConfig* c)
    : tex_ID(tex)
    , render_settings(c)
{
}

void GUI::draw()
{
    /* Start the Dear ImGui frame */
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    /* Setup visualizer */
    drawVisualizer();
    /* Setup controls */
    drawControls();

    ImGui::Render();

}

void GUI::drawVisualizer()
{
    ImGui::SetWindowPos(UI_VISUALIZATION_NAME, UI_VISUALIZATION_POS);
    ImGui::SetWindowSize(UI_VISUALIZATION_NAME, UI_VISUALIZATION_SIZE);
    ImGui::Begin(UI_VISUALIZATION_NAME, nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);
    ImGui::Image((void*)(intptr_t)tex_ID, ImVec2(RESOLUTION_X, RESOLUTION_Y));
    ImGui::End();
}

void GUI::drawControls()
{
    ImGui::SetWindowPos(UI_CONTROLS_NAME, UI_CONTROLS_POS);
    ImGui::SetWindowSize(UI_CONTROLS_NAME, UI_CONTROLS_SIZE);
    ImGui::Begin(UI_CONTROLS_NAME, nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_MenuBar);

    drawMenuBar();
    drawCameraPanel();
    drawLightPanel();
    drawRaycastingPanel();
    
    ImGui::End();
}

void GUI::drawMenuBar()
{
    if (ImGui::BeginMenuBar())
    {
        if (ImGui::BeginMenu("Open"))
        {
            if (ImGui::MenuItem("Genus 2")) {}
            if (ImGui::MenuItem("Wineglass")) {}
            if (ImGui::MenuItem("Porous Surface")) {}
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Help"))
        {
            if (ImGui::MenuItem("About")) {}
            ImGui::EndMenu();
        }

        ImGui::EndMenuBar();
    }
}

void GUI::drawCameraPanel()
{
    if (ImGui::CollapsingHeader("Camera"))
    {
        ImGui::Text("Change view-point in spherical coordinates.");
        ImGui::SliderFloat("Distance", &render_settings->camera_pos_r, CAMERA_RADIUS_RANGE[0], CAMERA_RADIUS_RANGE[1]);
        ImGui::SliderFloat("Azimuth angle (rad)", &render_settings->camera_pos_azimuth, CAMERA_AZIMUTH_RANGE[0], CAMERA_AZIMUTH_RANGE[1]);
        ImGui::SliderFloat("Polar angle (rad)", &render_settings->camera_pos_polar, CAMERA_POLAR_RANGE[0], CAMERA_POLAR_RANGE[1]);
    }
}

void GUI::drawLightPanel()
{
    if (ImGui::CollapsingHeader("Light"))
    {
        ImGui::Text("Change light properties.");
        ImGui::SliderFloat("Light power", &render_settings->light_power, LIGHT_POWER_RANGE[0], LIGHT_POWER_RANGE[1]);
        ImGui::ColorEdit3("Light color (RGB)", render_settings->light_rgb.data());
    }
}

void GUI::drawRaycastingPanel()
{
    if (ImGui::CollapsingHeader("Ray Casting"))
    {
        ImGui::Text("Change parameters for ray casting.");
        ImGui::SliderFloat("Sampling step length", &render_settings->sampling_step_len, SAMPLING_STEP_LEN_RANGE[0], SAMPLING_STEP_LEN_RANGE[1]);
        // ImGui::ColorEdit3("Light color (RGB)", render_settings->light_rgb.data());
    }
}