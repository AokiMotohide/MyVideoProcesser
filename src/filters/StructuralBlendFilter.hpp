#pragma once
#include "VideoFilter.hpp"
#include <deque>
#include <opencv2/opencv.hpp>

// Implementation of Temporal/Structural Blending for AI Generated Video Noise
class StructuralBlendFilter : public VideoFilter {
private:
  int windowSize;
  float blendWeight; // Controls how much of the original frame vs smoothed
                     // frame to keep

  // Buffer to hold recent frames for blending
  std::deque<cv::Mat> frameBuffer;

public:
  StructuralBlendFilter();

  cv::Mat apply(const cv::Mat &frame, int frameIndex,
                VideoContext &ctx) override;

  void init(const VideoContext &ctx) override;

  // strength scales between 0.0 (no effect) to 1.0 (max smoothing)
  void setStrength(float value) override;

  // Set how many past frames to blend over
  void setWindowSize(int size);
  int getWindowSize() const;

  std::string getName() const override;
};
