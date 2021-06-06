cmake_minimum_required(VERSION 3.8.0)
project(volrend VERSION 0.1.0 LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS  "${CMAKE_CUDA_FLAGS} -w -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --disable-warnings --expt-relaxed-constexpr")


# Find CUDA
find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

# Find OpenGL
find_package(OpenGL REQUIRED)
if(OpenGL_FOUND)
	message(STATUS "OpenGL found.")
	include_directories(${OPENGL_INCLUDE_DIR})
endif()

# Find GLFW3
find_package(glfw3 3.2 REQUIRED)
if(glfw3_FOUND)
	message(STATUS "GLFW3 found.")
endif()

# Add GLAD
include_directories(3rdLibs/glad/include)
add_library(glad 3rdLibs/glad/src/glad.c)
message(STATUS "Added GLAD.")

# Add ImGui
file(GLOB_RECURSE ImGui_SOURCE_FILES 
	3rdLibs/imgui/*.c
	3rdLibs/imgui/*.cc
	3rdLibs/imgui/*.cpp
	3rdLibs/imgui/backends/*.c
	3rdLibs/imgui/backends/*.cc
	3rdLibs/imgui/backends/*.cpp)
include_directories(3rdLibs/imgui/ 3rdLibs/imgui/backends)
add_library(imgui ${ImGui_SOURCE_FILES})
target_link_libraries(imgui glad glfw)
add_definitions(-DIMGUI_IMPL_OPENGL_LOADER_GLAD)
message(STATUS "Added ImGui.")

# Add include path
include_directories(
	src
	3rdLibs
	include)

# Find source files
file(GLOB_RECURSE SOURCE_FILES 
    src/*.c
	src/*.cc
	src/*.cpp
	src/*.cu)
message(STATUS "Source files: " ${SOURCE_FILES})

# Find and copy resources
file(GLOB_RECURSE RESOURCE_IMG_FILES 
    resources/*.png,
	resources/*.jpg)
file(GLOB_RECURSE RESOURCE_BIN_FILES 
    resources/*.bin)
file(GLOB_RECURSE RESOURCE_FONT_FILES 
	3rdLibs/imgui/misc/fonts/*.ttf)
file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/resources ${PROJECT_BINARY_DIR}/resources/images ${PROJECT_BINARY_DIR}/resources/fonts)
foreach(IMG_FILE ${RESOURCE_IMG_FILES})
	file(COPY ${IMG_FILE} DESTINATION ${PROJECT_BINARY_DIR}/resources/images/)
endforeach()
foreach(BIN_FILE ${RESOURCE_BIN_FILES})
	file(COPY ${BIN_FILE} DESTINATION ${PROJECT_BINARY_DIR}/resources/bin/)
endforeach()
foreach(FONT_FILE ${RESOURCE_FONT_FILES})
	file(COPY ${FONT_FILE} DESTINATION ${PROJECT_BINARY_DIR}/resources/fonts/)
endforeach()
add_definitions(-DRESOUCES_DIRECTORY="${PROJECT_BINARY_DIR}/resources")
message(STATUS "Resource files: " ${RESOURCE_IMG_FILES} ${RESOURCE_BIN_FILES} ${RESOURCE_FONT_FILES})


# Specific flags
if(WIN32)
	add_definitions(-D_AMD64_)
endif()

# Executables
add_executable(volrend ${SOURCE_FILES})
# Set CUDA compiling properties
set_target_properties(volrend PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# Link GLAD, glfw3, OpenGL
target_link_libraries(volrend glad glfw imgui OpenGL::GL ${CMAKE_DL_LIBS})

