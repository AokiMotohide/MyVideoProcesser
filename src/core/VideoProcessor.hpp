#pragma once
#include "../filters/VideoFilter.hpp"
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class VideoProcessor {
private:
  std::string inputPath;
  std::string outputPath;

  cv::VideoCapture capture;
  cv::VideoWriter writer;

  std::vector<std::shared_ptr<VideoFilter>> filters;

  VideoContext context;
  bool isProcessing;
  bool stopRequested;
  bool outputIsImageSequence;

public:
  VideoProcessor();
  ~VideoProcessor();

  // Open video file or image sequence (e.g. "image_%04d.png")
  bool loadInput(const std::string &path);

  // Set output file or folder
  bool setOutput(const std::string &path, bool isSequence = false);

  // Add a filter to the processing pipeline
  void addFilter(std::shared_ptr<VideoFilter> filter);

  // Clear all filters
  void clearFilters();

  // Start processing (synchronous for core logic, GUI will handle threading
  // manually if needed)
  bool process();

  // Process a single frame without saving (used for GUI preview)
  cv::Mat processPreviewFrame(int frameIndex);

  // Request processing to stop early
  void stop();

  // Getters for progress reporting
  float getProgress() const;
  int getTotalFrames() const;
  int getCurrentFrame() const;
  bool getIsProcessing() const;

  // Video info
  double getFPS() const;
  cv::Size getFrameSize() const;
};
