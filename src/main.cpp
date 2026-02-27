#include <atomic>
#include <iostream>
#include <memory>
#include <thread>

// OpenCV
#include <opencv2/opencv.hpp>

// ImGui & GLFW
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include "imgui.h"
#include <GLFW/glfw3.h> // Will drag system OpenGL headers

// Core Logic
#include "core/VideoProcessor.hpp"
#include "filters/FlickerRemovalFilter.hpp"

// Utility to upload OpenCV Mat to OpenGL Texture
GLuint matToTexture(const cv::Mat &mat, GLuint imageTexture) {
  if (mat.empty())
    return imageTexture;

  if (imageTexture == 0) {
    glGenTextures(1, &imageTexture);
    glBindTexture(GL_TEXTURE_2D, imageTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
  } else {
    glBindTexture(GL_TEXTURE_2D, imageTexture);
  }

  cv::Mat image;
  // OpenGL needs RGBA or RGB. OpenCV defaults to BGR
  if (mat.channels() == 3) {
    cv::cvtColor(mat, image, cv::COLOR_BGR2RGB);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.cols, image.rows, 0, GL_RGB,
                 GL_UNSIGNED_BYTE, image.ptr());
  } else if (mat.channels() == 4) {
    cv::cvtColor(mat, image, cv::COLOR_BGRA2RGBA);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.cols, image.rows, 0, GL_RGBA,
                 GL_UNSIGNED_BYTE, image.ptr());
  } else if (mat.channels() == 1) {
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, mat.cols, mat.rows, 0, GL_RED,
                 GL_UNSIGNED_BYTE, mat.ptr());
  }

  return imageTexture;
}

int main(int, char **) {
  // 1. Setup GLFW
  if (!glfwInit())
    return 1;

  const char *glsl_version = "#version 130";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

  GLFWwindow *window =
      glfwCreateWindow(1280, 720, "Flicker Removal Tool", nullptr, nullptr);
  if (window == nullptr)
    return 1;

  glfwMakeContextCurrent(window);
  glfwSwapInterval(1); // Enable vsync

  // 2. Setup Dear ImGui context
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
  (void)io;
  ImGui::StyleColorsDark();

  // 3. Setup Platform/Renderer backends
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init(glsl_version);

  // 4. App State and Logic
  VideoProcessor processor;
  auto flickerFilter = std::make_shared<FlickerRemovalFilter>();
  processor.addFilter(flickerFilter);

  char inputPathBuf[256] = "input.mp4";
  char outputPathBuf[256] = "output.mp4";

  int windowSize = flickerFilter->getWindowSize();
  float strength = 0.8f;

  int previewFrameIndex = 0;
  GLuint previewTexture = 0;
  int previewWidth = 0, previewHeight = 0;

  std::atomic<bool> isProcessing(false);
  std::thread processingThread;

  // 5. Main Loop
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    // Start the ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // ImGui UI Construction
    ImGui::Begin("Flicker Removal Settings");

    ImGui::InputText("Input Video", inputPathBuf, IM_ARRAYSIZE(inputPathBuf));
    if (ImGui::Button("Load Video")) {
      if (processor.loadInput(inputPathBuf)) {
        previewFrameIndex = 0;
        cv::Mat preview = processor.processPreviewFrame(previewFrameIndex);
        if (!preview.empty()) {
          previewTexture = matToTexture(preview, previewTexture);
          previewWidth = preview.cols;
          previewHeight = preview.rows;
        }
      }
    }
    ImGui::Separator();

    // Parameter adjustment
    ImGui::Text("Flicker Removal Parameters");
    if (ImGui::SliderInt("Window Size (Frames)", &windowSize, 1, 30)) {
      flickerFilter->setWindowSize(windowSize);
    }
    if (ImGui::SliderFloat("Blend Strength", &strength, 0.0f, 1.0f)) {
      flickerFilter->setStrength(strength);
    }

    // Preview controls
    if (processor.getTotalFrames() > 0 && !isProcessing) {
      ImGui::Separator();
      ImGui::Text("Preview");
      if (ImGui::SliderInt("Preview Frame", &previewFrameIndex, 0,
                           processor.getTotalFrames() - 1)) {
        cv::Mat preview = processor.processPreviewFrame(previewFrameIndex);
        previewTexture = matToTexture(preview, previewTexture);
      }
      if (previewTexture != 0) {
        // Show preview image scaled down
        float scale = 400.0f / previewWidth;
        ImGui::Image((void *)(intptr_t)previewTexture,
                     ImGui::Vec2(previewWidth * scale, previewHeight * scale));
      }
    }
    ImGui::Separator();

    // Output and Processing start
    ImGui::InputText("Output Video", outputPathBuf,
                     IM_ARRAYSIZE(outputPathBuf));

    if (!isProcessing) {
      if (ImGui::Button("Start Processing", ImGui::Vec2(150, 40))) {
        if (processor.setOutput(outputPathBuf)) {
          isProcessing = true;
          if (processingThread.joinable())
            processingThread.join();

          // Run process in separate thread to not block GUI
          processingThread = std::thread([&processor, &isProcessing]() {
            processor.process();
            isProcessing = false;
          });
        }
      }
    } else {
      if (ImGui::Button("Stop Early", ImGui::Vec2(150, 40))) {
        processor.stop();
      }
      ImGui::ProgressBar(processor.getProgress(), ImGui::Vec2(-1.0f, 0.0f));
      ImGui::Text("Processing frame %d / %d", processor.getCurrentFrame(),
                  processor.getTotalFrames());
    }

    ImGui::End();

    // Rendering
    ImGui::Render();
    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(window);
  }

  // Cleanup
  if (processingThread.joinable()) {
    processor.stop();
    processingThread.join();
  }

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
