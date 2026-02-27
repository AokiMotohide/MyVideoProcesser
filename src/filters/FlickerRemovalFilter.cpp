#include "FlickerRemovalFilter.hpp"
#include <iostream>
#include <numeric>

FlickerRemovalFilter::FlickerRemovalFilter() : windowSize(5), strength(0.8f) {}

void FlickerRemovalFilter::init(const VideoContext &ctx) {
  luminanceBuffer.clear();
}

void FlickerRemovalFilter::setStrength(float value) {
  strength = std::max(0.0f, std::min(1.0f, value));
}

void FlickerRemovalFilter::setWindowSize(int size) {
  windowSize = std::max(1, size);
}

int FlickerRemovalFilter::getWindowSize() const { return windowSize; }

std::string FlickerRemovalFilter::getName() const {
  return "Luminance Smoothing";
}

double FlickerRemovalFilter::calculateFrameLuminance(const cv::Mat &frame) {
  if (frame.empty())
    return 0.0;

  cv::Mat gray;
  if (frame.channels() == 3) {
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
  } else if (frame.channels() == 4) {
    cv::cvtColor(frame, gray, cv::COLOR_BGRA2GRAY);
  } else {
    gray = frame;
  }

  // Calculate mean of all pixels
  cv::Scalar meanVal = cv::mean(gray);
  return meanVal[0];
}

cv::Mat FlickerRemovalFilter::apply(const cv::Mat &frame, int frameIndex,
                                    VideoContext &ctx) {
  if (frame.empty())
    return frame;

  double currentLuminance = calculateFrameLuminance(frame);

  // Add current frame's luminance to buffer
  luminanceBuffer.push_back(currentLuminance);

  // Keep buffer size within the window limit
  if (luminanceBuffer.size() > static_cast<size_t>(windowSize)) {
    luminanceBuffer.pop_front();
  }

  // Calculate moving average
  double sum =
      std::accumulate(luminanceBuffer.begin(), luminanceBuffer.end(), 0.0);
  double avgLuminance = sum / luminanceBuffer.size();

  // If strength is 0, we don't adjust anything.
  // If strength is 1, we try to match the average exactly.
  double targetLuminance =
      currentLuminance + (avgLuminance - currentLuminance) * strength;

  // Avoid division by zero
  if (currentLuminance < 1.0)
    currentLuminance = 1.0;

  double ratio = targetLuminance / currentLuminance;

  cv::Mat result;
  // Scale pixel values by the ratio depending on their type
  // Convert to floating point for precise multiplication, then convert back
  frame.convertTo(result, -1, ratio, 0);

  return result;
}
