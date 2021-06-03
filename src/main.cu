#include <iostream>
#include <string>
#include <chrono>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
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


#define clock_t std::chrono::time_point<std::chrono::system_clock>


/**
 * A data structure for storing rendering output as texture
 */
struct RenderingOutput
{
    struct cudaGraphicsResource* cuda_tex_resource;
    GLuint gl_tex_ID;

    RenderingOutput(int res_x, int res_y)
    {
        /* Create an OpenGL texture */
        glGenTextures(1, &gl_tex_ID);
        glBindTexture(GL_TEXTURE_2D, gl_tex_ID);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, res_x, res_y, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glGenerateMipmap(GL_TEXTURE_2D);
        /* Set texturing parameters */
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        /* Register the texture with CUDA */
        checkCudaErrors(cudaGraphicsGLRegisterImage(&cuda_tex_resource, gl_tex_ID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
    }
};


ResourceManager resource_manager;


void vol_render(VolumeRenderer& renderer, MainScene& scene, RenderingOutput* out, RenderingConfig* render_settings)
{
    int res_x = RESOLUTION_X;
    int res_y = RESOLUTION_Y;

    /* Update the scene according to the given rendering settings */
    scene.updateConfiguration(render_settings);

    /* Send the scene to the renderer */
    renderer.setVolume(scene.geometry);
    renderer.setCamera(scene.main_camera);
    renderer.setLights(scene.lights, scene.count_lights);
    renderer.setClassifier(scene.classifier);

    /* Render to a pixel array */
    Eigen::Vector3f* pixel_array;
    checkCudaErrors(cudaMallocManaged((void **)&pixel_array, res_x * res_y * sizeof(Eigen::Vector3f)));

    // std::cout << "\n\nRendering..." << std::endl;

	// clock_t start_time = std::chrono::system_clock::now();
    
    renderer.renderFrontToBack(pixel_array, res_x, res_y, render_settings->sampling_step_len);

    // clock_t end_time = std::chrono::system_clock::now();
	// double time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
	// std::cout << "\nTime elapsed: " << time_elapsed << " ms" << std::endl;

    /* Store the rendering result in bytes */
    int num_bytes = res_x * res_y * 4 * sizeof(unsigned char);
    unsigned char* img_data;
    checkCudaErrors(cudaMallocManaged((void **)&img_data, num_bytes));

    int tx = 8, ty = 8;
    dim3 blocks(res_x / tx + 1, res_y / ty + 1);
    dim3 threads(tx, ty);
    pixelArrayToBytes<<<blocks, threads>>>(pixel_array, img_data, res_x, res_y);
    checkCudaErrors(cudaDeviceSynchronize());

    //stbi_write_png(OUTPUT_FILE, res_x, res_y, 4, img_data, 0);

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


static void glfwErrorCallback(int error, const char* description)
{
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}


GLFWwindow* init_window()
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


void init_imgui(GLFWwindow* window)
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


int main(int, char**)
{
    /* Setup scene */
    MainScene scene;

    /* Setup volume renderer */
    VolumeRenderer renderer;

    /* Initialize window */
    GLFWwindow* window = init_window();
    if (!window)
    {
        return 1;
    }
    /* Initialize OpenGL loader */
    if (!gladLoadGL())
    {
        fprintf(stderr, "Failed to initialize OpenGL loader!\n");
        return 1;
    }
    /* Initialize ImGui */
    init_imgui(window);

    /* Rendering output */    
    RenderingOutput render_output(RESOLUTION_X, RESOLUTION_Y);

    /* Rendering settings */
    RenderingConfig* render_settings;
    checkCudaErrors(cudaMallocManaged((void**)&render_settings, sizeof(RenderingConfig)));
    init_rendering_config<<<1, 1>>>(render_settings);
    checkCudaErrors(cudaDeviceSynchronize());

    /* Initialize GUI */
    GUI window_ui(render_output.gl_tex_ID, render_settings);

    /* Main loop */
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        /* Render the scene */
        vol_render(renderer, scene, &render_output, render_settings);

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

    return 0;
}