#include <iostream>
#include <string>
#include <stdio.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include "utils.hpp"
#include "config.hpp"
#include "main_scene.hpp"
#include "volume_renderer.hpp"
#include "res_manager.hpp"
#include "gui.hpp"
#include "render_output.hpp"


/* Manager of resources */
ResourceManager resource_manager;


static void glfwErrorCallback(int error, const char* description)
{
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

GLFWwindow* initWindow()
{
    /* Initialize GLFW */
    glfwSetErrorCallback(glfwErrorCallback);
    if (!glfwInit())
    {
        return nullptr;
    }
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    /* Create window with graphics context */
    GLFWwindow* window = glfwCreateWindow(UI_WINDOW_SIZE_W, UI_WINDOW_SIZE_H, UI_WINDOW_TITLE, NULL, NULL);
    if (window == nullptr)
    {
        return nullptr;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    return window;
}

void initImGui(GLFWwindow* window)
{
    /* Setup Dear ImGui context */
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    /* Setup Dear ImGui style */
    ImGui::StyleColorsClassic();

    /* Setup Platform/Renderer backends */
    const char* glsl_version = "#version 100";
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    /* Load Fonts */
    io.Fonts->AddFontFromFileTTF((resource_manager.getResource("fonts/Roboto-Medium.ttf")).c_str(), 16.0f);
    io.Fonts->AddFontDefault();
}

void volumeRender(VolumeRenderer* renderer, MainScene* scene, RenderingOutput* out, RenderingConfig* render_settings)
{
    int res_x = out->width;
    int res_y = out->height;

    if (!scene->scene_opened)
    {
        return;
    }

    /* Update the scene according to the given rendering settings */
    scene->updateConfiguration(render_settings);

    /* Send the scene to the renderer */
    renderer->setVolume(scene->geometry);
    renderer->setCamera(scene->main_camera);
    renderer->setLights(scene->lights, scene->count_lights);
    renderer->setClassifier(scene->using_classifier);

    /* Render to a pixel array */
    Eigen::Vector3f* pixel_array;
    checkCudaErrors(cudaMallocManaged((void **)&pixel_array, res_x * res_y * sizeof(Eigen::Vector3f)));
    
    renderer->renderFrontToBack(pixel_array, res_x, res_y, render_settings->sampling_step_len);

    /* Store the rendering result in bytes */
    int num_bytes = res_x * res_y * RGBA_CHANNELS * sizeof(unsigned char);
    unsigned char* img_data;
    checkCudaErrors(cudaMallocManaged((void **)&img_data, num_bytes));

    int tx = CUDA_BLOCK_THREADS_X, ty = CUDA_BLOCK_THREADS_Y;
    dim3 blocks(res_x / tx + 1, res_y / ty + 1);
    dim3 threads(tx, ty);
    ImageUtils::pixelArrayToBytesRGBA<<<blocks, threads>>>(pixel_array, img_data, res_x, res_y);
    checkCudaErrors(cudaDeviceSynchronize());

    scene->processRenderingResult(img_data);

    /* Map CUDA device memory to a texture resource */
    cudaArray *texture_ptr;
	checkCudaErrors(cudaGraphicsMapResources(1, &(out->cuda_tex_resource), 0));
	checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, out->cuda_tex_resource, 0, 0));
	checkCudaErrors(cudaMemcpyToArray(texture_ptr, 0, 0, img_data, num_bytes, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGraphicsUnmapResources(1, &(out->cuda_tex_resource), 0));

    cudaFree(pixel_array);
    cudaFree(img_data);
}

int main(int, char**)
{
    /* Set CUDA device */
    cudaSetDevice(CUDA_DEVICE_ID);

    /* Setup main scene */
    MainScene* scene = new MainScene();
    /* Setup volume renderer */
    VolumeRenderer* renderer = new VolumeRenderer();

    /* Initialize window */
    GLFWwindow* window = initWindow();
    if (!window)
    {
        fprintf(stderr, "Failed to create GLFW window!\n");
        return 1;
    }
    /* Initialize OpenGL loader */
    if (!gladLoadGL())
    {
        fprintf(stderr, "Failed to initialize OpenGL loader!\n");
        return 1;
    }
    /* Initialize ImGui */
    initImGui(window);

    /* Rendering output */
    RenderingOutput* render_output = new RenderingOutput(OUTPUT_RESOLUTION_X, OUTPUT_RESOLUTION_Y);

    /* Initialize rendering settings */
    RenderingConfig* render_settings;
    checkCudaErrors(cudaMallocManaged((void**)&render_settings, sizeof(RenderingConfig)));
    initRenderingConfig<<<1, 1>>>(render_settings);
    checkCudaErrors(cudaDeviceSynchronize());
    
    /* Initialize GUI */
    GUI window_ui(scene, render_output->gl_tex_ID, render_settings);

    int curr_render_interval_idx = 0;

    /* Main loop */
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        /* Control the refreshing rate according to the rendering interval */
        if (curr_render_interval_idx++ >= render_settings->rendering_interval)
        {
            /* Render the scene */
            volumeRender(renderer, scene, render_output, render_settings);

            curr_render_interval_idx = 0;
        }

        /* Render UI */
        window_ui.draw();

        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0, 0, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    /* Cleanup */
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    cudaFree(render_settings);
    delete render_output;
    delete scene;
    delete renderer;

    return 0;
}