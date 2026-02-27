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

#include "core/VideoProcessor.hpp"
#include "filters/FlickerRemovalFilter.hpp"
#include "filters/NormalMapBlendFilter.hpp"
#include "filters/StructuralBlendFilter.hpp"
#include "portable-file-dialogs.h"

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
  io.FontGlobalScale = 1.2f; // Default font size adjusted
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
  int passes = 1;
  int flickerMode = 0; // 0 = Lighting, 1 = AI Structural

  int previewFrameIndex = 0;
  GLuint previewTexture = 0;
  int previewWidth = 0, previewHeight = 0;

  std::atomic<bool> isProcessing(false);
  std::thread processingThread;

  std::string statusMessage = "";
  ImVec4 statusColor = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);

  // 5. Main Loop
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    // Start the ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // ImGui UI Construction
    ImGui::Begin("UI Settings");
    ImGui::SliderFloat("Text Size", &io.FontGlobalScale, 0.5f, 3.0f, "%.2f");
    ImGui::End();

    // Main Unified Window
    ImGui::Begin("Flicker Removal Tool");

    ImGui::InputText("Input Video", inputPathBuf, IM_ARRAYSIZE(inputPathBuf));
    ImGui::SameLine();
    auto doLoadVideo = [&]() {
      if (processor.loadInput(inputPathBuf)) {
        previewFrameIndex = 0;
        cv::Mat preview = processor.processPreviewFrame(previewFrameIndex);
        if (!preview.empty()) {
          previewTexture = matToTexture(preview, previewTexture);
          previewWidth = preview.cols;
          previewHeight = preview.rows;
        }
        statusMessage = "Video loaded successfully!";
        statusColor = ImVec4(0.2f, 1.0f, 0.2f, 1.0f);
      } else {
        statusMessage = "Failed to load video. Check the path.";
        statusColor = ImVec4(1.0f, 0.2f, 0.2f, 1.0f);
        previewTexture = 0;
      }
    };

    if (ImGui::Button("Browse...##Input")) {
      auto f = pfd::open_file(
                   "Choose Video", ".",
                   {"Video Files", "*.mp4 *.mkv *.avi *.mov", "All Files", "*"})
                   .result();
      if (!f.empty()) {
        snprintf(inputPathBuf, sizeof(inputPathBuf), "%s", f[0].c_str());
        doLoadVideo();
      }
    }
    ImGui::SameLine();
    if (ImGui::Button("Load Video")) {
      doLoadVideo();
    }

    if (!statusMessage.empty()) {
      ImGui::TextColored(statusColor, "%s", statusMessage.c_str());
    }
    ImGui::Separator();

    // Parameter adjustment
    ImGui::Text("Flicker Removal Type");
    ImGui::RadioButton("Lighting Flicker (輝度・照明)", &flickerMode, 0);
    ImGui::SameLine();
    ImGui::RadioButton("AI Generation Flicker (形状・絵柄)", &flickerMode, 1);
    ImGui::SameLine();
    ImGui::RadioButton("Normal Map Smoothing (法線マップ用)", &flickerMode, 2);
    ImGui::Separator();

    ImGui::Text("Parameters");

    ImGui::Text("Presets: ");
    ImGui::SameLine();
    if (ImGui::Button("Low (弱)")) {
      windowSize = 5;
      strength = 0.5f;
      passes = 1;
    }
    ImGui::SameLine();
    if (ImGui::Button("Medium (中)")) {
      windowSize = 15;
      strength = 0.8f;
      passes = 2;
    }
    ImGui::SameLine();
    if (ImGui::Button("High (強)")) {
      windowSize = 30;
      strength = 1.0f;
      passes = 3;
    }

    ImGui::SliderInt("Window Size (Frames)", &windowSize, 1, 30);
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("How many surrounding frames to average. Larger = "
                        "smoother but slower.");

    ImGui::SliderFloat("Blend Strength", &strength, 0.0f, 1.0f);
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("How much of the smoothed luminance to blend in. 1.0 = "
                        "Max smoothing.");

    ImGui::SliderInt("Passes (Repeats)", &passes, 1, 5);
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("How many times to apply the filter. Higher passes = "
                        "stronger flicker reduction.");

    ImGui::Separator();

    // Preview
    ImGui::Text("Preview");
    // Preview controls
    if (processor.getTotalFrames() > 0 && !isProcessing) {
      if (ImGui::SliderInt("Preview Frame", &previewFrameIndex, 0,
                           processor.getTotalFrames() - 1)) {
        processor.clearFilters();
        for (int p = 0; p < passes; ++p) {
          std::shared_ptr<VideoFilter> filter;
          if (flickerMode == 0) {
            auto lfilter = std::make_shared<FlickerRemovalFilter>();
            lfilter->setWindowSize(windowSize);
            lfilter->setStrength(strength);
            filter = lfilter;
          } else if (flickerMode == 1) {
            auto sfilter = std::make_shared<StructuralBlendFilter>();
            sfilter->setWindowSize(windowSize);
            sfilter->setStrength(strength);
            filter = sfilter;
          } else {
            auto nfilter = std::make_shared<NormalMapBlendFilter>();
            nfilter->setWindowSize(windowSize);
            nfilter->setStrength(strength);
            filter = nfilter;
          }
          processor.addFilter(filter);
        }
        cv::Mat preview = processor.processPreviewFrame(previewFrameIndex);
        previewTexture = matToTexture(preview, previewTexture);
      }
      if (previewTexture != 0) {
        // Show preview image scaled down
        float scale = ImGui::GetContentRegionAvail().x / previewWidth;
        if (scale > 1.0f)
          scale = 1.0f; // Don't upscale
        ImGui::Image((void *)(intptr_t)previewTexture,
                     ImVec2(previewWidth * scale, previewHeight * scale));
      }
    } else if (isProcessing) {
      ImGui::Text("Processing... No preview available.");
    } else {
      ImGui::Text("Load a video to see the preview here.");
    }
    ImGui::Separator();

    // Export Settings
    ImGui::Text("Export Settings");

    static int outputMode = 0; // 0=Video, 1=Image Sequence
    ImGui::RadioButton("Video File Output", &outputMode, 0);
    ImGui::SameLine();
    ImGui::RadioButton("Image Sequence Folder", &outputMode, 1);

    if (outputMode == 0) {
      ImGui::InputText("Output Video", outputPathBuf,
                       IM_ARRAYSIZE(outputPathBuf));
      ImGui::SameLine();
      if (ImGui::Button("Browse...##Output")) {
        auto f = pfd::save_file("Choose Output Destination", ".",
                                {"Video Files", "*.mp4 *.mkv *.avi *.mov",
                                 "All Files", "*"})
                     .result();
        if (!f.empty()) {
          std::string chosen = f;
          if (chosen.length() < 4 ||
              chosen.substr(chosen.length() - 4) != ".mp4") {
            chosen += ".mp4";
          }
          snprintf(outputPathBuf, sizeof(outputPathBuf), "%s", chosen.c_str());
        }
      }
    } else {
      ImGui::InputText("Output Folder", outputPathBuf,
                       IM_ARRAYSIZE(outputPathBuf));
      ImGui::SameLine();
      if (ImGui::Button("Browse...##OutputFolder")) {
        auto f = pfd::select_folder("Choose Output Folder", ".").result();
        if (!f.empty()) {
          snprintf(outputPathBuf, sizeof(outputPathBuf), "%s", f.c_str());
        }
      }
    }

    if (!isProcessing) {
      if (ImGui::Button("Start Processing", ImVec2(150, 40))) {
        bool isSeq = (outputMode == 1);
        if (processor.setOutput(outputPathBuf, isSeq)) {
          // Re-populate filters for multi-pass
          processor.clearFilters();
          for (int p = 0; p < passes; ++p) {
            std::shared_ptr<VideoFilter> filter;
            if (flickerMode == 0) {
              auto lfilter = std::make_shared<FlickerRemovalFilter>();
              lfilter->setWindowSize(windowSize);
              lfilter->setStrength(strength);
              filter = lfilter;
            } else if (flickerMode == 1) {
              auto sfilter = std::make_shared<StructuralBlendFilter>();
              sfilter->setWindowSize(windowSize);
              sfilter->setStrength(strength);
              filter = sfilter;
            } else {
              auto nfilter = std::make_shared<NormalMapBlendFilter>();
              nfilter->setWindowSize(windowSize);
              nfilter->setStrength(strength);
              filter = nfilter;
            }
            processor.addFilter(filter);
          }
          isProcessing = true;
          statusMessage = "Processing started...";
          statusColor = ImVec4(1.0f, 1.0f, 0.2f, 1.0f);
          if (processingThread.joinable())
            processingThread.join();

          // Run process in separate thread to not block GUI
          processingThread = std::thread(
              [&processor, &isProcessing, &statusMessage, &statusColor]() {
                bool success = processor.process();
                isProcessing = false;
                if (success) {
                  statusMessage = "Processing completed successfully!";
                  statusColor = ImVec4(0.2f, 1.0f, 0.2f, 1.0f);
                } else {
                  statusMessage = "Processing stopped or failed.";
                  statusColor = ImVec4(1.0f, 0.2f, 0.2f, 1.0f);
                }
              });
        }
      }
    } else {
      if (ImGui::Button("Stop Early", ImVec2(150, 40))) {
        processor.stop();
      }
      ImGui::ProgressBar(processor.getProgress(), ImVec2(-1.0f, 0.0f));
      ImGui::Text("Processing frame %d / %d (%.1f %%)",
                  processor.getCurrentFrame(), processor.getTotalFrames(),
                  processor.getProgress() * 100.0f);
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
