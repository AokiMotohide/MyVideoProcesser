cmake_minimum_required(VERSION 3.16)
project(FlickerRemovalTool VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# ---------- 1. OpenCV ----------
# Find OpenCV package using the provided path
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# ---------- 2. FetchContent for GLFW and ImGui ----------
include(FetchContent)

# Fetch GLFW
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.3.8
)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# Fetch ImGui
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        v1.90.1
)
FetchContent_MakeAvailable(imgui)

# Create an ImGui library target manually since imgui doesn't provide a CMakeLists that builds a library by default
set(IMGUI_DIR ${imgui_SOURCE_DIR})
add_library(imgui STATIC
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui PUBLIC 
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
)
target_link_libraries(imgui PUBLIC glfw)
# Link OpenGL
find_package(OpenGL REQUIRED)
target_link_libraries(imgui PUBLIC OpenGL::GL)


# ---------- 3. Project Configuration ----------

# Add all source files
file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADERS "src/*.hpp")

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE src)
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE 
    ${OpenCV_LIBS}
    imgui
    glfw
    OpenGL::GL
)

# Copy OpenCV DLLs to output directory on build (Windows only)
if(WIN32)
    # This is a basic approach. Better approach is to use CMake's GET_TARGET_PROPERTY for exact dll paths,
    # but since OpenCV structure varies, this is a catch-all that copies dlls from bin folder.
    string(REGEX REPLACE "/lib/?$" "/bin/" OPENCV_BIN_DIR "${OpenCV_DIR}/x64/vc16")
    if (EXISTS "${OPENCV_BIN_DIR}")
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OPENCV_BIN_DIR}/opencv_world480d.dll" # We assume 4.x here. It might fail if version differs, safe to ignore
            "${OPENCV_BIN_DIR}/opencv_world480.dll"
            $<TARGET_FILE_DIR:${PROJECT_NAME}>
            || ${CMAKE_COMMAND} -E echo "DLL copy failed (might just mean specific versions weren't found, make sure OpenCV bin is in PATH)"
        )
    endif()
endif()
