#include "util_ops.h"

// #include <memory>
// #include <ostream>

#include "traits.h"

namespace abyss::core {

/**
 * CopyVisitor implementation
 */
CopyVisitor::CopyVisitor(std::vector<int> shape) : shape_{shape} {}

void CopyVisitor::visit(ArrayImpl<int32_t>* from) {
  eval(from);
}
void CopyVisitor::visit(ArrayImpl<double>* from) {
  eval(from);
}

/**
 * ArrayPrintVisitor Implementations
 */
void ArrayPrintVisitor::visit(ArrayImpl<bool>* a) {
  eval(a);
  // std::ostringstream oss;
  // std::vector<int> coords;

  // // int width = dec_str_width(a);
  // // fixed width for boolean statements
  // int width = 6;

  // int i = 0;
  // while (i < a->size()) {
  //   coords = unravel_index(i, shape_);

  //   // starting brackets
  //   // lambda to find the first none zero dimension
  //   auto first_none_zero = [](int idx) { return idx != 0; };
  //   int start_bracket_pad =
  //       std::find_if(coords.rbegin(), coords.rend(), first_none_zero) -
  //       coords.rbegin();

  //   if (start_bracket_pad) {
  //     if (i != 0)
  //       oss << std::setw(coords.size() - start_bracket_pad) << std::setfill(' ')
  //           << ' ';
  //     oss << std::setw(start_bracket_pad) << std::setfill('[') << "[";
  //   }

  //   // values
  //   oss << std::setw(width + 1) << std::setfill(' ') << std::right
  //       << std::boolalpha << a->at(i);

  //   // ending brackets
  //   int j = shape_.size();
  //   // lambda to check if the current dimension is at the last id.
  //   auto last_in_dim = [&j, this](int idx) {
  //     j--;
  //     return idx != shape_[j] - 1;
  //   };
  //   int end_bracket_pad =
  //       std::find_if(coords.rbegin(), coords.rend(), last_in_dim) -
  //       coords.rbegin();

  //   if (end_bracket_pad) {
  //     oss << std::setw(end_bracket_pad) << std::setfill(']') << std::left
  //         << "]";
  //     if (i != a->size()) oss << '\n';
  //   } else {
  //     oss << ",";
  //   }

  //   i++;
  // }

  // result_str_ = oss.str();
}
void ArrayPrintVisitor::visit(ArrayImpl<uint8_t>* a) {
  std::ostringstream oss;
  std::vector<int> coords;

  // oss << '[';
  int width = dec_str_width(a);

  int i = 0;
  while (i < a->size()) {
    coords = unravel_index(i, shape_);

    // starting brackets
    // lambda to find the first none zero dimension
    auto first_none_zero = [](int idx) { return idx != 0; };
    int start_bracket_pad =
        std::find_if(coords.rbegin(), coords.rend(), first_none_zero) -
        coords.rbegin();

    if (start_bracket_pad) {
      if (i != 0)
        oss << std::setw(coords.size() - start_bracket_pad) << std::setfill(' ')
            << ' ';
      oss << std::setw(start_bracket_pad) << std::setfill('[') << "[";
    }

    // values
    oss << std::setw(width + 1) << std::setfill(' ') << std::right
        << a->at(i);

    // ending brackets
    int j = shape_.size();
    // lambda to check if the current dimension is at the last id.
    auto last_in_dim = [&j, this](int idx) {
      j--;
      return idx != shape_[j] - 1;
    };
    int end_bracket_pad =
        std::find_if(coords.rbegin(), coords.rend(), last_in_dim) -
        coords.rbegin();

    if (end_bracket_pad) {
      oss << std::setw(end_bracket_pad) << std::setfill(']') << std::left
          << "]";
      if (i != a->size()) oss << '\n';
    } else {
      oss << ",";
    }

    i++;
  }

  result_str_ = oss.str();
}

void ArrayPrintVisitor::visit(ArrayImpl<int32_t>* a) {
  // std::ostringstream oss;
  // std::vector<int> coords;

  // // oss << '[';
  // int width = dec_str_width(a);

  // int i = 0;
  // while (i < a->size()) {
  //   coords = unravel_index(i, shape_);

  //   // starting brackets
  //   // lambda to find the first none zero dimension
  //   auto first_none_zero = [](int idx) { return idx != 0; };
  //   int start_bracket_pad =
  //       std::find_if(coords.rbegin(), coords.rend(), first_none_zero) -
  //       coords.rbegin();

  //   // std::cout << "pad: " << start_bracket_pad << std::endl;
  //   if (start_bracket_pad) {
  //     if (i != 0)
  //       oss << std::setw(coords.size() - start_bracket_pad) << std::setfill(' ')
  //           << ' ';
  //     oss << std::setw(start_bracket_pad) << std::setfill('[') << "[";
  //   }

  //   // values
  //   oss << std::setw(width + 1) << std::setfill(' ') << std::right
  //       << a->at(i);

  //   // ending brackets
  //   int j = shape_.size();
  //   // lambda to check if the current dimension is at the last id.
  //   auto last_in_dim = [&j, this](int idx) {
  //     j--;
  //     return idx != shape_[j] - 1;
  //   };
  //   int end_bracket_pad =
  //       std::find_if(coords.rbegin(), coords.rend(), last_in_dim) -
  //       coords.rbegin();

  //   if (end_bracket_pad) {
  //     oss << std::setw(end_bracket_pad) << std::setfill(']') << std::left
  //         << "]";
  //     if (i != a->size()) oss << '\n';
  //   } else {
  //     oss << ",";
  //   }

  //   // std::cout<< coords[0]<<", "<< coords[1] <<", "<< coords[2] << std::endl;

  //   // std::cout << std::setw(2) << std::setfill('0') << std::hex <<
  //   // +has_bracket_flag[0] << ", "
  //   //           << std::setw(2) << std::setfill('0') << std::hex <<
  //   //           +has_bracket_flag[1] << ", "
  //   //           << std::setw(2) << std::setfill('0') << std::hex <<
  //   //           +has_bracket_flag[2] << std::endl;
  //   i++;
  // }

  // // oss << ']';

  // result_str_ = oss.str();
  eval(a);
}
void ArrayPrintVisitor::visit(ArrayImpl<double>* a) {
  eval(a);
  // std::ostringstream oss;
  // std::vector<int> coords;

  // int i = 0;
  // while (i < a->size()) {
  //   coords = unravel_index(i, shape_);

  //   // starting brackets
  //   // lambda to find the first none zero dimension
  //   auto first_none_zero = [](int idx) { return idx != 0; };
  //   int start_bracket_pad =
  //       std::find_if(coords.rbegin(), coords.rend(), first_none_zero) -
  //       coords.rbegin();

  //   if (start_bracket_pad) {
  //     if (i != 0)
  //       oss << std::setw(coords.size() - start_bracket_pad) << std::setfill(' ')
  //           << ' ';
  //     oss << std::setw(start_bracket_pad) << std::setfill('[') << "[";
  //   }

  //   // values
  //   oss << std::setw(2) << std::setfill(' ') << std::right
  //       << a->at(i);

  //   // ending brackets
  //   int j = shape_.size();
  //   // lambda to check if the current dimension is at the last id.
  //   auto last_in_dim = [&j, this](int idx) {
  //     j--;
  //     return idx != shape_[j] - 1;
  //   };
  //   int end_bracket_pad =
  //       std::find_if(coords.rbegin(), coords.rend(), last_in_dim) -
  //       coords.rbegin();

  //   if (end_bracket_pad) {
  //     oss << ' ' << std::setw(end_bracket_pad) << std::setfill(']') << std::left
  //         << "]";
  //     if (i != a->size()) oss << '\n';
  //   } else {
  //     oss << ",";
  //   }

  //   i++;
  // }

  // result_str_ = oss.str();
}

}  // namespace abyss::core