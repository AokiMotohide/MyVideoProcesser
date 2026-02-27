#include "NormalMapBlendFilter.hpp"

NormalMapBlendFilter::NormalMapBlendFilter()
    : windowSize(3), blendWeight(0.5f) {}

void NormalMapBlendFilter::init(const VideoContext &ctx) {
  frameBuffer.clear();
}

void NormalMapBlendFilter::setStrength(float value) {
  // 0 = no blending, 0.95 = heavy blending
  blendWeight = std::max(0.0f, std::min(0.95f, value * 0.95f));
}

void NormalMapBlendFilter::setWindowSize(int size) {
  windowSize = std::max(1, size);
}

int NormalMapBlendFilter::getWindowSize() const { return windowSize; }

std::string NormalMapBlendFilter::getName() const {
  return "Normal Map Smoothing (Vector Renormalize)";
}

cv::Mat NormalMapBlendFilter::apply(const cv::Mat &frame, int frameIndex,
                                    VideoContext &ctx) {
  if (frame.empty())
    return frame;

  cv::Mat frame3c;
  if (frame.channels() == 4) {
    cv::cvtColor(frame, frame3c, cv::COLOR_BGRA2BGR);
  } else if (frame.channels() == 1) {
    cv::cvtColor(frame, frame3c, cv::COLOR_GRAY2BGR);
  } else {
    frame3c = frame;
  }

  // Normal map frames are typically BGR [0, 255].
  // Convert to Vec3f space [-1.0, 1.0] for accurate vector math.
  cv::Mat currentFloat;
  frame3c.convertTo(currentFloat, CV_32FC3); // 3-channel 32-bit float
  currentFloat = (currentFloat / 255.0f) * 2.0f - cv::Scalar::all(1.0f);

  if (frameBuffer.empty() || blendWeight == 0.0f) {
    frameBuffer.push_back(currentFloat);
    return frame;
  }

  // 1. Calculate the Average (Summing the vectors)
  cv::Mat pastSum = cv::Mat::zeros(currentFloat.size(), currentFloat.type());
  for (const auto &pastFrame : frameBuffer) {
    pastSum += pastFrame;
  }
  cv::Mat pastAverage = pastSum / static_cast<float>(frameBuffer.size());

  // 2. Blend the Current Vector with the Past Average Vector
  cv::Mat smoothedVector;
  cv::addWeighted(currentFloat, 1.0f - blendWeight, pastAverage, blendWeight,
                  0.0, smoothedVector);

  // 3. Re-Normalize the Vectors (Ensure length is 1.0)
  // We must do this pixel by pixel
  cv::Mat resultFloat =
      cv::Mat::zeros(smoothedVector.size(), smoothedVector.type());
  for (int y = 0; y < smoothedVector.rows; ++y) {
    for (int x = 0; x < smoothedVector.cols; ++x) {
      cv::Vec3f vec = smoothedVector.at<cv::Vec3f>(y, x);
      float length =
          std::sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
      if (length > 0.0001f) { // Prevent division by zero
        vec[0] /= length;
        vec[1] /= length;
        vec[2] /= length;
      } else {
        // If it collapses to 0, default to pointing straight out (Z up in
        // normal space) In DirectX typical normal maps: B=Z, G=Y, R=X. Z
        // is 1.0. In OpenGL: B=Z. Let's assume standard RGB->XYZ mapping, so
        // B(Z) = 1.0
        vec[0] = 0.0f; // B (X)
        vec[1] = 0.0f; // G (Y)
        vec[2] = 1.0f; // R (Z)
      }
      resultFloat.at<cv::Vec3f>(y, x) = vec;
    }
  }

  // Add smoothed & normalized vector to the history buffer
  frameBuffer.push_back(resultFloat);
  if (frameBuffer.size() > static_cast<size_t>(windowSize)) {
    frameBuffer.pop_front();
  }

  // 4. Encode back to BGR [0, 255]
  resultFloat = (resultFloat + cv::Scalar::all(1.0f)) / 2.0f * 255.0f;
  cv::Mat finalResult;
  resultFloat.convertTo(finalResult,
                        CV_8UC3); // Convert back to standard 8-bit unsigned

  // Since original input might have 4 channels (BGRA), ensure we convert safely
  // back
  if (frame.channels() == 4) {
    cv::cvtColor(finalResult, finalResult, cv::COLOR_BGR2BGRA);
  }

  return finalResult;
}
