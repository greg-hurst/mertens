#pragma once

// Exact coefficient modes and cutoff metadata for the outer-Q6, inner-Q6 S2
// transform. Runtime dispatch verifies the integer cutoff order exactly.

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace S2Q6Modes {

enum class Mode : std::uint8_t { M00, M01, M02, M03, M04, M05, M06, M07, M08, M09, M10, M11, M12, M13, M14, M15, M16, M17, M18, M19, M20, M21, M22, M23, M24, M25 };

inline constexpr std::size_t kModeCount = 26;
inline constexpr std::size_t kBasisCount = 9;
inline constexpr std::size_t kOuterClassCount = 4;
inline constexpr std::size_t kRawCellCount = 40;
inline constexpr std::array<std::int64_t, kBasisCount> kDenominators = { 1, 2, 3, 4, 6, 9, 12, 18, 36 };
inline constexpr std::array<std::array<std::int32_t, kBasisCount>, kModeCount> kCoefficients = {
    std::array<std::int32_t, kBasisCount>{ 1, -2, -2, 1, 0, 0, 0, 0, 0 },
    std::array<std::int32_t, kBasisCount>{ 1, -2, -2, 1, 1, 0, 0, 0, 0 },
    std::array<std::int32_t, kBasisCount>{ 1, -2, -2, 1, 2, 0, 0, 0, 0 },
    std::array<std::int32_t, kBasisCount>{ 1, -2, -2, 1, 2, 1, 0, 0, 0 },
    std::array<std::int32_t, kBasisCount>{ 1, -2, -2, 1, 3, 0, -1, 0, 0 },
    std::array<std::int32_t, kBasisCount>{ 1, -2, -2, 1, 3, 0, 0, 0, 0 },
    std::array<std::int32_t, kBasisCount>{ 1, -2, -2, 1, 3, 1, -1, -1, 0 },
    std::array<std::int32_t, kBasisCount>{ 1, -2, -2, 1, 3, 1, -1, 0, 0 },
    std::array<std::int32_t, kBasisCount>{ 1, -2, -2, 1, 3, 1, 0, 0, 0 },
    std::array<std::int32_t, kBasisCount>{ 1, -2, -2, 1, 4, 1, -2, -2, 0 },
    std::array<std::int32_t, kBasisCount>{ 1, -2, -2, 1, 4, 1, -2, -2, 1 },
    std::array<std::int32_t, kBasisCount>{ 1, -2, -2, 1, 4, 1, -2, -1, 0 },
    std::array<std::int32_t, kBasisCount>{ 1, -2, -2, 1, 4, 1, -1, -1, 0 },
    std::array<std::int32_t, kBasisCount>{ 1, -2, -2, 1, 4, 1, -1, 0, 0 },
    std::array<std::int32_t, kBasisCount>{ 1, -2, -1, 0, 0, 0, 0, 0, 0 },
    std::array<std::int32_t, kBasisCount>{ 1, -2, -1, 0, 1, 0, 0, 0, 0 },
    std::array<std::int32_t, kBasisCount>{ 1, -2, -1, 1, 0, 0, 0, 0, 0 },
    std::array<std::int32_t, kBasisCount>{ 1, -2, -1, 1, 1, 0, 0, 0, 0 },
    std::array<std::int32_t, kBasisCount>{ 1, -2, -1, 1, 2, 0, -1, 0, 0 },
    std::array<std::int32_t, kBasisCount>{ 1, -2, -1, 1, 2, 0, 0, 0, 0 },
    std::array<std::int32_t, kBasisCount>{ 1, -2, 0, 0, 0, 0, 0, 0, 0 },
    std::array<std::int32_t, kBasisCount>{ 1, -2, 0, 1, 0, 0, 0, 0, 0 },
    std::array<std::int32_t, kBasisCount>{ 1, -1, -1, 0, 0, 0, 0, 0, 0 },
    std::array<std::int32_t, kBasisCount>{ 1, -1, -1, 0, 1, 0, 0, 0, 0 },
    std::array<std::int32_t, kBasisCount>{ 1, -1, 0, 0, 0, 0, 0, 0, 0 },
    std::array<std::int32_t, kBasisCount>{ 1, 0, 0, 0, 0, 0, 0, 0, 0 },
};
inline constexpr std::array<Mode, kRawCellCount> kRawCellModes = { Mode::M10, Mode::M09, Mode::M11, Mode::M12, Mode::M13, Mode::M07, Mode::M04, Mode::M05, Mode::M02, Mode::M01, Mode::M17, Mode::M15, Mode::M14, Mode::M22, Mode::M24, Mode::M25, Mode::M06, Mode::M07, Mode::M08, Mode::M03, Mode::M02, Mode::M01, Mode::M00, Mode::M16, Mode::M14, Mode::M22, Mode::M24, Mode::M25, Mode::M18, Mode::M19, Mode::M17, Mode::M16, Mode::M21, Mode::M20, Mode::M24, Mode::M25, Mode::M23, Mode::M22, Mode::M24, Mode::M25 };
inline constexpr std::array<std::uint8_t, kRawCellCount> kCutoffOuterDivisors = { 6, 3, 2, 6, 1, 3, 6, 2, 3, 1, 2, 6, 1, 3, 2, 1, 3, 2, 1, 3, 2, 3, 1, 2, 1, 3, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1 };
inline constexpr std::array<std::uint8_t, kRawCellCount> kCutoffInnerDivisors = { 6, 6, 6, 3, 6, 3, 2, 3, 2, 3, 2, 1, 2, 1, 1, 1, 6, 6, 6, 3, 3, 2, 3, 2, 2, 1, 1, 1, 6, 6, 3, 3, 2, 2, 1, 1, 6, 3, 2, 1 };
inline constexpr std::array<std::int8_t, kRawCellCount> kRemovedSigns = { 1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1 };
inline constexpr std::array<std::uint8_t, kRawCellCount> kRemovedBasisIndices = { 8, 7, 6, 7, 4, 5, 6, 4, 4, 2, 3, 4, 1, 2, 1, 0, 7, 6, 4, 5, 4, 4, 2, 3, 1, 2, 1, 0, 6, 4, 4, 2, 3, 1, 1, 0, 4, 2, 1, 0 };
inline constexpr std::array<std::size_t, kOuterClassCount + 1> kOuterClassCellOffsets = { 0, 16, 28, 36, 40 };

template<Mode ModeValue, typename T>
struct Term;

template<typename T>
struct Term<Mode::M00, T> {
    static inline T eval(T q) {
        return T(0) + q - T(2) * (q / T(2)) - T(2) * (q / T(3)) + (q / T(4));
    }
};

template<typename T>
struct Term<Mode::M01, T> {
    static inline T eval(T q) {
        return T(0) + q - T(2) * (q / T(2)) - T(2) * (q / T(3)) + (q / T(4)) + (q / T(6));
    }
};

template<typename T>
struct Term<Mode::M02, T> {
    static inline T eval(T q) {
        return T(0) + q - T(2) * (q / T(2)) - T(2) * (q / T(3)) + (q / T(4)) + T(2) * (q / T(6));
    }
};

template<typename T>
struct Term<Mode::M03, T> {
    static inline T eval(T q) {
        return T(0) + q - T(2) * (q / T(2)) - T(2) * (q / T(3)) + (q / T(4)) + T(2) * (q / T(6)) + (q / T(9));
    }
};

template<typename T>
struct Term<Mode::M04, T> {
    static inline T eval(T q) {
        return T(0) + q - T(2) * (q / T(2)) - T(2) * (q / T(3)) + (q / T(4)) + T(3) * (q / T(6)) - (q / T(12));
    }
};

template<typename T>
struct Term<Mode::M05, T> {
    static inline T eval(T q) {
        return T(0) + q - T(2) * (q / T(2)) - T(2) * (q / T(3)) + (q / T(4)) + T(3) * (q / T(6));
    }
};

template<typename T>
struct Term<Mode::M06, T> {
    static inline T eval(T q) {
        return T(0) + q - T(2) * (q / T(2)) - T(2) * (q / T(3)) + (q / T(4)) + T(3) * (q / T(6)) + (q / T(9)) - (q / T(12)) - (q / T(18));
    }
};

template<typename T>
struct Term<Mode::M07, T> {
    static inline T eval(T q) {
        return T(0) + q - T(2) * (q / T(2)) - T(2) * (q / T(3)) + (q / T(4)) + T(3) * (q / T(6)) + (q / T(9)) - (q / T(12));
    }
};

template<typename T>
struct Term<Mode::M08, T> {
    static inline T eval(T q) {
        return T(0) + q - T(2) * (q / T(2)) - T(2) * (q / T(3)) + (q / T(4)) + T(3) * (q / T(6)) + (q / T(9));
    }
};

template<typename T>
struct Term<Mode::M09, T> {
    static inline T eval(T q) {
        return T(0) + q - T(2) * (q / T(2)) - T(2) * (q / T(3)) + (q / T(4)) + T(4) * (q / T(6)) + (q / T(9)) - T(2) * (q / T(12)) - T(2) * (q / T(18));
    }
};

template<typename T>
struct Term<Mode::M10, T> {
    static inline T eval(T q) {
        return T(0) + q - T(2) * (q / T(2)) - T(2) * (q / T(3)) + (q / T(4)) + T(4) * (q / T(6)) + (q / T(9)) - T(2) * (q / T(12)) - T(2) * (q / T(18)) + (q / T(36));
    }
};

template<typename T>
struct Term<Mode::M11, T> {
    static inline T eval(T q) {
        return T(0) + q - T(2) * (q / T(2)) - T(2) * (q / T(3)) + (q / T(4)) + T(4) * (q / T(6)) + (q / T(9)) - T(2) * (q / T(12)) - (q / T(18));
    }
};

template<typename T>
struct Term<Mode::M12, T> {
    static inline T eval(T q) {
        return T(0) + q - T(2) * (q / T(2)) - T(2) * (q / T(3)) + (q / T(4)) + T(4) * (q / T(6)) + (q / T(9)) - (q / T(12)) - (q / T(18));
    }
};

template<typename T>
struct Term<Mode::M13, T> {
    static inline T eval(T q) {
        return T(0) + q - T(2) * (q / T(2)) - T(2) * (q / T(3)) + (q / T(4)) + T(4) * (q / T(6)) + (q / T(9)) - (q / T(12));
    }
};

template<typename T>
struct Term<Mode::M14, T> {
    static inline T eval(T q) {
        return T(0) + q - T(2) * (q / T(2)) - (q / T(3));
    }
};

template<typename T>
struct Term<Mode::M15, T> {
    static inline T eval(T q) {
        return T(0) + q - T(2) * (q / T(2)) - (q / T(3)) + (q / T(6));
    }
};

template<typename T>
struct Term<Mode::M16, T> {
    static inline T eval(T q) {
        return T(0) + q - T(2) * (q / T(2)) - (q / T(3)) + (q / T(4));
    }
};

template<typename T>
struct Term<Mode::M17, T> {
    static inline T eval(T q) {
        return T(0) + q - T(2) * (q / T(2)) - (q / T(3)) + (q / T(4)) + (q / T(6));
    }
};

template<typename T>
struct Term<Mode::M18, T> {
    static inline T eval(T q) {
        return T(0) + q - T(2) * (q / T(2)) - (q / T(3)) + (q / T(4)) + T(2) * (q / T(6)) - (q / T(12));
    }
};

template<typename T>
struct Term<Mode::M19, T> {
    static inline T eval(T q) {
        return T(0) + q - T(2) * (q / T(2)) - (q / T(3)) + (q / T(4)) + T(2) * (q / T(6));
    }
};

template<typename T>
struct Term<Mode::M20, T> {
    static inline T eval(T q) {
        return T(0) + q - T(2) * (q / T(2));
    }
};

template<typename T>
struct Term<Mode::M21, T> {
    static inline T eval(T q) {
        return T(0) + q - T(2) * (q / T(2)) + (q / T(4));
    }
};

template<typename T>
struct Term<Mode::M22, T> {
    static inline T eval(T q) {
        return T(0) + q - (q / T(2)) - (q / T(3));
    }
};

template<typename T>
struct Term<Mode::M23, T> {
    static inline T eval(T q) {
        return T(0) + q - (q / T(2)) - (q / T(3)) + (q / T(6));
    }
};

template<typename T>
struct Term<Mode::M24, T> {
    static inline T eval(T q) {
        return T(0) + q - (q / T(2));
    }
};

template<typename T>
struct Term<Mode::M25, T> {
    static inline T eval(T q) {
        return T(0) + q;
    }
};

template<typename Callback>
static inline void dispatchMode(Mode mode, Callback&& callback) {
    switch (mode) {
        case Mode::M00: callback(std::integral_constant<Mode, Mode::M00>{}); return;
        case Mode::M01: callback(std::integral_constant<Mode, Mode::M01>{}); return;
        case Mode::M02: callback(std::integral_constant<Mode, Mode::M02>{}); return;
        case Mode::M03: callback(std::integral_constant<Mode, Mode::M03>{}); return;
        case Mode::M04: callback(std::integral_constant<Mode, Mode::M04>{}); return;
        case Mode::M05: callback(std::integral_constant<Mode, Mode::M05>{}); return;
        case Mode::M06: callback(std::integral_constant<Mode, Mode::M06>{}); return;
        case Mode::M07: callback(std::integral_constant<Mode, Mode::M07>{}); return;
        case Mode::M08: callback(std::integral_constant<Mode, Mode::M08>{}); return;
        case Mode::M09: callback(std::integral_constant<Mode, Mode::M09>{}); return;
        case Mode::M10: callback(std::integral_constant<Mode, Mode::M10>{}); return;
        case Mode::M11: callback(std::integral_constant<Mode, Mode::M11>{}); return;
        case Mode::M12: callback(std::integral_constant<Mode, Mode::M12>{}); return;
        case Mode::M13: callback(std::integral_constant<Mode, Mode::M13>{}); return;
        case Mode::M14: callback(std::integral_constant<Mode, Mode::M14>{}); return;
        case Mode::M15: callback(std::integral_constant<Mode, Mode::M15>{}); return;
        case Mode::M16: callback(std::integral_constant<Mode, Mode::M16>{}); return;
        case Mode::M17: callback(std::integral_constant<Mode, Mode::M17>{}); return;
        case Mode::M18: callback(std::integral_constant<Mode, Mode::M18>{}); return;
        case Mode::M19: callback(std::integral_constant<Mode, Mode::M19>{}); return;
        case Mode::M20: callback(std::integral_constant<Mode, Mode::M20>{}); return;
        case Mode::M21: callback(std::integral_constant<Mode, Mode::M21>{}); return;
        case Mode::M22: callback(std::integral_constant<Mode, Mode::M22>{}); return;
        case Mode::M23: callback(std::integral_constant<Mode, Mode::M23>{}); return;
        case Mode::M24: callback(std::integral_constant<Mode, Mode::M24>{}); return;
        case Mode::M25: callback(std::integral_constant<Mode, Mode::M25>{}); return;
    }
}

} // namespace S2Q6Modes
