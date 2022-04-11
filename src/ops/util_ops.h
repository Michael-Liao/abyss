#ifndef ABYSS_OPS_UTIL_OPS_H
#define ABYSS_OPS_UTIL_OPS_H

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <memory>
#include <sstream>
#include <type_traits>
#include <vector>

#include "core/array.h"
#include "core/dtype.h"
#include "core/visitor.h"
#include "traits.h"
#include "tensor.h"

namespace abyss::core {

/**
 * @brief Copy one Array to the other, it does not change the type.
 */
class CopyVisitor final : public VisitorBase,
                          public Tensor,
                          public UnaryVisitor<ArrayImpl<int32_t>>,
                          public UnaryVisitor<ArrayImpl<double>> {
 public:
  CopyVisitor(std::vector<int> shape);

  void visit(ArrayImpl<int32_t>*) override;
  void visit(ArrayImpl<double>*) override;

 private:
  std::vector<int> shape_;

  template <typename T>
  void eval(ArrayImpl<T>* from) {
    dtype_ = stypeof<T>();
    Tensor::shape_ = shape_;
    strides_ = shape2strides(shape_);
    data_ = std::make_shared<ArrayImpl<T>>(*from);
  }
};

class ArrayPrintVisitor final : public VisitorBase,
                                public Tensor,
                                public UnaryVisitor<ArrayImpl<bool>>,
                                public UnaryVisitor<ArrayImpl<uint8_t>>,
                                public UnaryVisitor<ArrayImpl<int32_t>>,
                                public UnaryVisitor<ArrayImpl<double>> {
 public:
  ArrayPrintVisitor(const std::vector<int>& shape) : shape_{shape} {
    // std::cout << shape_[0] << std::endl;
  }
  void visit(ArrayImpl<bool>*) override;
  void visit(ArrayImpl<uint8_t>*) override;
  void visit(ArrayImpl<int32_t>*) override;
  void visit(ArrayImpl<double>*) override;

  std::string str() { return result_str_; }

 private:
  const std::vector<int>& shape_;
  std::string result_str_;

  template <typename T>
  size_t max_str_len(const ArrayImpl<T>* const a) {
    std::ostringstream oss;

    size_t max_len = 1;
    for (auto& el : *a) {
      oss << el;
      if (max_len < oss.str().size()) {
        max_len = oss.str().length();
      }
      oss.str("");
    }

    return max_len;
  }

  /**
   * @brief find the string width before the decimal point
   */
  template <typename T>
  int dec_str_width(const ArrayImpl<T>* const a) {
    int digit_count = 0;

    T max_value = *std::max_element(a->begin(), a->end());
    if (max_value < 0) digit_count++;  // negative sign

    do {
      digit_count++;
      max_value /= 10;
    } while (std::trunc(max_value));

    return digit_count;
  }

  template <typename T, std::enable_if_t<std::is_integral<T>::value, int> = 0>
  void print_value(std::ostringstream& oss, T value, int width) {
    // int width = integral_width + 1;
    oss << std::setw(width) << std::setfill(' ') << std::right << value;
  }

  template <typename T,
            std::enable_if_t<std::is_floating_point<T>::value, int> = 1>
  void print_value(std::ostringstream& oss, T value, int width) {
    // int width = 1 + integral_width + oss.precision() + 1;
    // int precision = oss.precision();

    // if (value == std::trunc(value)) {
    //   precision = 1;
    //   oss << std::setprecision(precision);
    //   width = 1 + integral_width + 2;
    // }

    // oss.setf(std::ios::showpoint, std::ios::floatfield);
    // // oss << std::setw(width) << std::setfill(' ') << std::right
    // //     << value;
    oss << std::setw(width) << std::setfill(' ') << std::right << value;
  }

  template <typename T, std::enable_if_t<std::is_same<T, bool>::value, int> = 2>
  void print_value(std::ostringstream& oss, T value, int integral_width) {
    oss << std::setw(6) << std::setfill(' ') << std::right
        << std::boolalpha << value;
  }

  /**
   * @brief string formatting for integral types
   */
  template <typename T>
  void eval(ArrayImpl<T>* arr) {
    std::ostringstream oss;
    std::vector<int> coords;

    oss << '\n';

    // 1 extra whitespace, max decimal width of array elements, precision of
    // floats
    // int width = dec_str_width(arr);
    int width = (std::is_same<T, bool>::value) ? 6 : max_str_len(arr) + 1;

    int i = 0;
    while (i < arr->size()) {
      coords = unravel_index(i, shape_);

      // starting brackets
      // lambda to find the first none zero dimension
      auto first_none_zero = [](int idx) { return idx != 0; };
      int start_bracket_pad =
          std::find_if(coords.rbegin(), coords.rend(), first_none_zero) -
          coords.rbegin();

      if (start_bracket_pad) {
        if (i != 0)
          oss << std::setw(coords.size() - start_bracket_pad)
              << std::setfill(' ') << ' ';
        oss << std::setw(start_bracket_pad) << std::setfill('[') << "[";
      }

      // values printing strategies
      // print_value(oss, arr->at(i), width);

    oss << std::setw(width) << std::setfill(' ') <<std::boolalpha<< std::right << arr->at(i);
      // oss << std::setw(width + 1) << std::setfill(' ') << std::right
      //     << a->at(i);

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
        if (i != arr->size()) oss << '\n';
      } else {
        oss << ",";
      }

      i++;
    }

    result_str_ = oss.str();
  }
};

class DTypePrintVisitor final : public VisitorBase,
                                public Tensor,
                                public UnaryVisitor<DTypeImpl<bool>>,
                                public UnaryVisitor<DTypeImpl<uint8_t>>,
                                public UnaryVisitor<DTypeImpl<int32_t>>,
                                public UnaryVisitor<DTypeImpl<double>> {
 public:
  DTypePrintVisitor() = default;
  void visit(DTypeImpl<bool>*) override;
  void visit(DTypeImpl<uint8_t>*) override;
  void visit(DTypeImpl<int32_t>*) override;
  void visit(DTypeImpl<double>*) override;
};

// template <typename CallableTp>
// class ComparisonVisitor
//     : public VisitorBase,
//       public BinaryVisitor<ArrayImpl<int32_t>, ArrayImpl<int32_t>>/*,
//       public BinaryVisitor<ArrayImpl<int32_t>, ArrayImpl<double>>,
//       public BinaryVisitor<ArrayImpl<double>, ArrayImpl<int32_t>>,
//       public BinaryVisitor<ArrayImpl<double>, ArrayImpl<double>>*/ {
//  public:
//   ComparisonVisitor(std::vector<int> shape)
//       : shape_{shape}, comp_fn_() {}

//   void visit(ArrayImpl<int32_t>* a, ArrayImpl<int32_t>* b) override {
//     auto out = std::make_shared<ArrayImpl<bool>>(a->size());
//     for (size_t i = 0; i < a->size(); i++) {
//       out->at(i) = comp_fn_(a->at(i), b->at(i));
//     }

//     result_dtype_ = stypeof<bool>();
//     output_shape_ = shape_;
//     output_strides_ = shape2strides(shape_);
//     shared_data_ = out;
//   }
//   // void visit(ArrayImpl<int32_t>* a, ArrayImpl<double>* b) override;
//   // void visit(ArrayImpl<double>* a, ArrayImpl<int32_t>* b) override;
//   // void visit(ArrayImpl<double>* a, ArrayImpl<double>* b) override;

//  private:
//   std::vector<int> shape_;
//   CallableTp comp_fn_;
// };

}  // namespace abyss::core

#endif