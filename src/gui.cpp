#include <string>
#include "gui.hpp"
#include "utils.hpp"


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
            if (ImGui::MenuItem("CT Head"))
            {
                scene->openScene(BUILTIN_GEOM_CT_HEAD);
            }
            if (ImGui::MenuItem("MR Brain"))
            {
                scene->openScene(BUILTIN_GEOM_MR_BRAIN);
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
        ImGui::Text("Volume Renderer (v0.1.0)\n\nAn interactive volume visualization tool.\n\nCopyright 2021 Yuehao Wang\n\n\n");
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
    if (ImGui::CollapsingHeader("Lighting"))
    {
        ImGui::Text("Change light and material properties.");
        ImGui::SliderFloat("Light power", &render_settings->light_power, LIGHT_POWER_RANGE[0], LIGHT_POWER_RANGE[1]);
        ImGui::ColorEdit3("Light color (RGB)", render_settings->light_rgb.data());
        ImGui::SliderFloat("Ambient magnitude", &render_settings->ambient_magnitude, AMBIENT_MAGNITUDE_RANGE[0], AMBIENT_MAGNITUDE_RANGE[1]);
        ImGui::SliderFloat("Specular shininess", &render_settings->specular_shininess, SPECULAR_SHININESS_RANGE[0], SPECULAR_SHININESS_RANGE[1]);
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

static float imguiPlotGaussian(void* params, int i)
{
    float mu = ((float*)params)[0];
    float sigma = ((float*)params)[1];
    float half_n_val = ((float*)params)[2] / 2;
    return MathUtils::Gaussian(mu * half_n_val, sigma * half_n_val, (i - half_n_val));
}

void GUI::drawTransferFunctionPanel()
{
    if (ImGui::CollapsingHeader("Transfer Function"))
    {
        ImGui::Text("Tune the transfer function for visualization.");

        const char* cls_items[] = {"Isosurface", "Entire Volume"};
        ImGui::Combo("Classifier type", (int*)&render_settings->classifier_type, cls_items, IM_ARRAYSIZE(cls_items));

        const char* cm_items[] = {"Parula", "Heat", "Jet", "Turbo", "Hot", "Gray", "Magma", "Inferno", "Plasma", "Viridis", "Cividis", "Github"};
        ImGui::Combo("Colormap", (int*)&render_settings->colormap_type, cm_items, IM_ARRAYSIZE(cm_items));

        ImGui::AlignTextToFramePadding(); ImGui::Text("Visualization Target:");
        ImGui::RadioButton("Normal-x", (int*)&render_settings->visualize_target, Classifier::VisualizationTarget::NORMAL_X);
        ImGui::SameLine(); ImGui::RadioButton("Normal-y", (int*)&render_settings->visualize_target, Classifier::VisualizationTarget::NORMAL_Y);
        ImGui::SameLine(); ImGui::RadioButton("Normal-z", (int*)&render_settings->visualize_target, Classifier::VisualizationTarget::NORMAL_Z);
        ImGui::SameLine(); ImGui::RadioButton("Value", (int*)&render_settings->visualize_target, Classifier::VisualizationTarget::VALUE);
        ImGui::Spacing();

        ImGui::Separator(); ImGui::Spacing();

        if (render_settings->classifier_type == Classifier::ClassifierType::ISOSURFACE)
        {
            ImGui::Text("Specific settings for Isosurface classifier.");
            ImGui::InputFloat("Isovalue", &render_settings->isosurface_classifier_isovalue, 0.01);
            ImGui::SliderFloat("Sigma", &render_settings->isosurface_classifier_sigma, ISOSURFACE_CLASSIFIER_SIGMA_RANGE[0], ISOSURFACE_CLASSIFIER_SIGMA_RANGE[1]);
            
            /* Limit the range of isovalue */
            render_settings->isosurface_classifier_isovalue = MathUtils::clamp(
                render_settings->isosurface_classifier_isovalue,
                ISOSURFACE_CLASSIFIER_ISOVALUE_RANGE[0], ISOSURFACE_CLASSIFIER_ISOVALUE_RANGE[1]);

            int values_count = 400;
            float params[3] = {render_settings->isosurface_classifier_isovalue, render_settings->isosurface_classifier_sigma, (float)values_count};
            ImGui::PlotLines("Transfer\nfunction", &imguiPlotGaussian, (void*)params, values_count, 0, NULL, 0.0f, 1.0f, ImVec2(0, 60));
        }
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