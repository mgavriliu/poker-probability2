#pragma once

#include <iomanip>
#include <iostream>
#include <sstream>

inline void printProgress(float progress) {
  int barWidth = 70;
  std::cout << "[";
  int pos = static_cast<int>(barWidth * progress);
  for (int i = 0; i < barWidth; ++i) {
    if (i < pos)
      std::cout << "=";
    else if (i == pos)
      std::cout << ">";
    else
      std::cout << " ";
  }
  std::cout << "] " << int(progress * 100.0) << " %\r";
  std::cout.flush();
}

// Helper function to format large integers
inline std::string formatNumber(uint64_t num) {
  std::stringstream ss;
  ss.imbue(std::locale(""));
  ss << std::fixed << std::setprecision(0) << num;
  return ss.str();
}

// Helper function to format binary
inline std::string formatBinary16(uint16_t value) {
  std::string binary;
  const std::string yellow_start = "\033[33m"; // ANSI escape code for yellow
  const std::string color_end = "\033[0m";
  for (int i = 15; i >= 0; --i) {
    if (value & (1ULL << i)) {
      binary += yellow_start + '1' + color_end;
    } else {
      binary += '0';
    }
    if (i % 3 == 0 && i != 0) {
      binary += ',';
    }
  }
  return binary;
}

inline std::string formatBinary64(uint64_t value) {
  std::string binary;
  const std::string red_start = "\033[31m";    // ANSI escape code for red
  const std::string yellow_start = "\033[33m"; // ANSI escape code for yellow
  const std::string color_end = "\033[0m";
  for (int i = 63; i >= 0; --i) {
    if (value & (1ULL << i)) {
      binary += yellow_start + '1' + color_end;
    } else {
      binary += '0';
    }
    if (i % 3 == 0 && i != 0) {
      binary += ',';
    }
  }
  return binary;
}