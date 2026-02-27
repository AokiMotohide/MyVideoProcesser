#include "StructuralBlendFilter.hpp"
#include <iostream>

StructuralBlendFilter::StructuralBlendFilter()
    : windowSize(3), blendWeight(0.5f) {}

void StructuralBlendFilter::init(const VideoContext &ctx) {
  frameBuffer.clear();
}

void StructuralBlendFilter::setStrength(float value) {
  // Map strength [0.0, 1.0] to blendWeight [0.0, 0.95]
  // 0 = no blending (keep original), 0.95 = heavily blend with past frames
  blendWeight = std::max(0.0f, std::min(0.95f, value * 0.95f));
}

void StructuralBlendFilter::setWindowSize(int size) {
  windowSize = std::max(1, size);
}

int StructuralBlendFilter::getWindowSize() const { return windowSize; }

std::string StructuralBlendFilter::getName() const {
  return "AI Structural Deflicker (Temporal Blend)";
}

cv::Mat StructuralBlendFilter::apply(const cv::Mat &frame, int frameIndex,
                                     VideoContext &ctx) {
  if (frame.empty())
    return frame;

  // Convert current frame to a float matrix for accurate blending
  cv::Mat currentFloat;
  frame.convertTo(currentFloat, CV_32F);

  if (frameBuffer.empty() || blendWeight == 0.0f) {
    frameBuffer.push_back(currentFloat);
    return frame;
  }

  // Temporal Blending:
  // Smoothed = (1 - weight) * Current + weight * Average(Past N frames)

  cv::Mat pastSum = cv::Mat::zeros(currentFloat.size(), currentFloat.type());
  for (const auto &pastFrame : frameBuffer) {
    pastSum += pastFrame;
  }
  cv::Mat pastAverage = pastSum / static_cast<float>(frameBuffer.size());

  cv::Mat smoothedFloat;
  cv::addWeighted(currentFloat, 1.0f - blendWeight, pastAverage, blendWeight,
                  0.0, smoothedFloat);

  // Store the smoothed result in the buffer to make the filter Recursive
  // (Exponential Moving Average style) or store original to make it purely
  // windowed. For AI noise, recursive tends to yield smoother results without
  // aggressive ghosting if weight isn't huge.
  frameBuffer.push_back(smoothedFloat);

  // Keep buffer strictly within window size
  if (frameBuffer.size() > static_cast<size_t>(windowSize)) {
    frameBuffer.pop_front();
  }

  cv::Mat result;
  smoothedFloat.convertTo(result, frame.type());
  return result;
}
