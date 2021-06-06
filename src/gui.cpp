#include <string>
#include "gui.hpp"


/**
 * GUI class
 */

GUI::GUI(MainScene* scn, GLuint tex, RenderingConfig* c)
    : scene(scn)
    , tex_ID(tex)
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
    ImGui::Image((void*)(intptr_t)tex_ID, ImVec2(OUTPUT_RESOLUTION_X, OUTPUT_RESOLUTION_Y));
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
    drawTransferFunctionPanel();
    drawPerformancePanel();

    // ImGui::ShowDemoWindow();
    
    ImGui::End();
}

void GUI::drawMenuBar()
{
    bool to_open_about_popup = false;
    bool to_open_save_result_popup = false;

    if (ImGui::BeginMenuBar())
    {
        if (ImGui::BeginMenu("Open"))
        {
            if (ImGui::MenuItem("Genus 2"))
            {
                scene->openScene(BUILTIN_GEOM_GENUS_TWO);
            }
            if (ImGui::MenuItem("Wineglass"))
            {
                scene->openScene(BUILTIN_GEOM_WINEGLASS);
            }
            if (ImGui::MenuItem("Porous Surface"))
            {
                scene->openScene(BUILTIN_GEOM_POROUS_SURFACE);
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Flame Heat Field"))
            {
                scene->openScene(BUILTIN_GEOM_FLAME_HEAT_FIELD);
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Tool"))
        {
            if (ImGui::MenuItem("Save Result"))
            {
                to_open_save_result_popup = true;
            }
            
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Help"))
        {
            if (ImGui::MenuItem("About"))
            {
                to_open_about_popup = true;
            }
            
            ImGui::EndMenu();
        }

        ImGui::EndMenuBar();
    }

    if (to_open_about_popup)
    {
        ImGui::OpenPopup("About");
        ImVec2 center = ImGui::GetMainViewport()->GetCenter();
        ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
        to_open_about_popup = false;
    }
    if (ImGui::BeginPopupModal("About", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
    {
        ImGui::Text("Volume Renderer\n\nAn interactive volume visualization tool.\n\nCopyright 2021 Yuehao Wang\n\n\n");
        ImGui::Separator();

        if (ImGui::Button("OK", ImVec2(120, 0)))
        {
            ImGui::CloseCurrentPopup();
        }
        ImGui::SetItemDefaultFocus();
        ImGui::EndPopup();
    }

    if (to_open_save_result_popup)
    {
        ImGui::OpenPopup("Save Result");
        ImVec2 center = ImGui::GetMainViewport()->GetCenter();
        ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
        to_open_about_popup = false;
    }
    if (ImGui::BeginPopupModal("Save Result", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
    {
        static char file_path[512] = "./output.png";
        ImGui::Text("Please input the path to saving the result");
        ImGui::InputText("Saving Path", file_path, IM_ARRAYSIZE(file_path));
        ImGui::Separator();

        if (ImGui::Button("Save", ImVec2(120, 0)))
        {
            scene->saveRenderingResult(std::string(file_path));
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(120, 0)))
        {
            ImGui::CloseCurrentPopup();
        }
        ImGui::SetItemDefaultFocus();
        ImGui::EndPopup();
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
        ImGui::SliderFloat("Sampling step", &render_settings->sampling_step_len, SAMPLING_STEP_LEN_RANGE[0], SAMPLING_STEP_LEN_RANGE[1]);
    }
}

void GUI::drawTransferFunctionPanel()
{
    if (ImGui::CollapsingHeader("Transfer Function"))
    {
        const char* items[] = {"Isosurface", "Entire Volume"};
        ImGui::Text("Tune the transfer function for visualization.");
        ImGui::Combo("Classifier Type", (int*)&render_settings->classifier_type, items, IM_ARRAYSIZE(items)); //("Sampling step", &render_settings->sampling_step_len, SAMPLING_STEP_LEN_RANGE[0], SAMPLING_STEP_LEN_RANGE[1]);
    }
}

void GUI::drawPerformancePanel()
{
    if (ImGui::CollapsingHeader("Performance"))
    {
        ImGui::Text("Options and information for improving performance.");
        ImGui::Text("Performance: %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
        ImGui::Spacing(); ImGui::Spacing();
        int input_uint_step = 1;
        ImGui::InputScalar("Rendering interval", ImGuiDataType_U8, &render_settings->rendering_interval, &input_uint_step);
    }
}