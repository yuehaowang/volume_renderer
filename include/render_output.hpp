#pragma once
#include <glad/glad.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <mutex>
#include "utils.hpp"


/**
 * A data structure with mutex lock for storing rendering output as texture
 */

struct RenderingOutput
{
    struct cudaGraphicsResource* cuda_tex_resource;
    GLuint gl_tex_ID;
    int width;
    int height;

    RenderingOutput(int res_x, int res_y)
        : width(res_x)
        , height(res_y)
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
