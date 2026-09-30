/*
* String Pattern Match Engine
* Copyright 2026 Alexander Danileiko
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at:
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* This software is provided on an "AS IS" basis, WITHOUT WARRANTIES OR CONDITIONS
* OF ANY KIND, either express or implied.
*/

#pragma once

#include <vector>
#include <deque>
#include <algorithm>
#include <unordered_map>
#include <string>
#include <bit>
#include <string_view>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <cwctype>
#include <cwchar>
#include <concepts>
#include <stdexcept>

// Platform-specific SIMD headers
#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#ifdef _WIN32
#define NOMINMAX
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#elif defined(_M_ARM64) || defined(__aarch64__) || defined(_M_ARM) || defined(__arm__)
#include <arm_neon.h>
#endif

// Target attribute macros ensuring GCC and Clang allow compiling SIMD code paths
// without requiring global compiler flags like -mavx2 or -mavx512f.
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(_M_X64))
#define SPM_TARGET_SSE41 __attribute__((target("sse4.1")))
#define SPM_TARGET_AVX2 __attribute__((target("avx2")))
#define SPM_TARGET_AVX512 __attribute__((target("avx512f,avx512bw")))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpsabi"
#else
#define SPM_TARGET_SSE41
#define SPM_TARGET_AVX2
#define SPM_TARGET_AVX512
#endif

#define SPM_TARGET_NEON

//-------------------------------------------------------------------------------------------
// Algorithm Description
//-------------------------------------------------------------------------------------------
// This engine implements an optimized string pattern matching algorithm engineered
// for high-throughput prefix/suffix filtering and wildcard evaluation.
//
// Phase 1: Registration (AddPattern)
// Each inserted pattern is parsed to find the first wildcard character ('*' or '?').
// The string is split into a literal prefix, a middle segment, and an anchored literal suffix.
// The engine calculates a polynomial hash of the literal prefix and stores it
// in a hash map, grouped by the length of the prefix.
// Additionally, it extracts 64-bit SWAR (Single Word Multiple Characters) prefix signatures,
// calculates minimum required match lengths (non-'*' elements), and anchors trailing literals
// for instant O(1) rejection before entering the backtracking state machine.
//
// Phase 2: Execution (Search)
// During search, the engine iterates through known registered prefix lengths using length 
// vectors and radix/SWAR filters. It computes a rolling polynomial hash of the target text at
// these exact lengths using SIMD-accelerated instructions (AVX-512, AVX2, SSE4.1, or NEON) to
// process up to 16 characters concurrently.
// If a hash collision occurs, minimum required match lengths and tail anchors reject 
// non-matching targets in O(1) time. The engine performs a SWAR/exact string comparison of 
// the prefix. If the prefix matches, the remaining suffix is evaluated against an iterative,
// non-recursive wildcard backtracking engine augmented with remaining-length bounds checking.
//-------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------
// Enumerations & Concepts
//-------------------------------------------------------------------------------------------

// Restrict allowed character types to standard narrowly-defined boundaries
template <typename T>
concept SupportedChar = std::same_as<T, char> || std::same_as<T, wchar_t>;

// Defines compile-time validation, casing behavior, and normalization requirements
template <typename TPolicy, typename CharT>
concept CasePolicy = requires(TPolicy                       policy,
                              CharT                         character,
                              std::basic_string_view<CharT> text) 
{
    { TPolicy::isCaseInsensitive }          -> std::convertible_to<bool>;
    { policy(character)          } noexcept -> std::same_as<CharT>;
    { TPolicy::IsValid(text)     } noexcept -> std::same_as<bool>;
};

//-------------------------------------------------------------------------------------------
// WildcardScope
//
// Semantics:
//   Defines the boundary behavior of wildcard characters ('*' and '?') during evaluation.
//   Passed as a parameter to AddPattern() to govern whether wildcards are allowed to match
//   the OS-specific directory delimiter ('/' on POSIX, '\' on Windows).
//
// How to Use:
//   - Pass WildcardScope::Default when matching generic strings, raw text, MIME types, or
//     URLs where '*' should consume any character up to the end of the text or next token.
//   - Pass WildcardScope::PathSegment when evaluating filesystem paths, preventing patterns
//     like "*.txt" or "src/*" from matching deep nested subdirectories such as 
//     "src/core/main.cpp".
//-------------------------------------------------------------------------------------------
enum class WildcardScope : uint8_t
{
    Default     = 0, // Unrestricted wildcard matching; '*' and '?' match all characters, 
                     // including the native directory separator.
    PathSegment = 1  // Segment-restricted matching; '*' and '?' halt expansion and fail 
                     // if the native OS path separator is encountered.
};

//-------------------------------------------------------------------------------------------
// OptimizationMode
//
// Semantics:
//   Controls SIMD hardware vectorization selection for prefix rolling polynomial hashes
//   and vector equality checks. Passed to the StringPatternMatch constructor to configure
//   the internal function pointer dispatch table.
//
// How to Use:
//   - Pass OptimizationMode::Auto (recommended for production). It executes runtime CPUID
//     and OSXSAVE detection on x86/x64 to automatically dispatch to AVX-512, AVX2, or SSE4.1,
//     or utilizes ARM NEON on ARM platforms, falling back to portable scalar if unsupported.
//   - Pass OptimizationMode::Scalar for testing, debugging, profiling baseline performance,
//     or operating on constrained environments where SIMD register saving is undesirable.
//   - Pass OptimizationMode::Vectorized to enforce hardware acceleration path resolution.
//-------------------------------------------------------------------------------------------
enum class OptimizationMode : uint8_t
{
    Auto       = 0, // Default mode: dynamically detects CPU capabilities 
                    // (AVX-512/AVX2/SSE4.1/NEON) with scalar fallback
    Scalar     = 1, // Diagnostic mode: disables SIMD execution lanes and forces 
                    // portable scalar algorithms
    Vectorized = 2  // Explicit vector mode: queries hardware capabilities and dispatches to
                    // the highest supported SIMD backend
};

//-------------------------------------------------------------------------------------------
// Predefined Case Policies
//-------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------
// CaseSensitivePolicy
//
// Semantics:
//   Enforces exact binary code-unit equality with zero character transformation.
//   Characters are evaluated directly against target code units without modification.
//
// Validation (IsValid):
//   Always returns true. Accepts all input strings unconditionally, including raw
//   binary streams and arbitrary multi-byte UTF-8 sequences.
//
// Performance & SIMD:
//   Optimal execution path. Disables vector range checking (HasNonAscii) and SIMD
//   subtraction/masking lanes entirely. Hashing proceeds via direct vector multiply-
//   accumulate instructions. Prefix verification evaluates via O(1) 64-bit SWAR
//   comparisons and delegates to highly optimized memory routines (e.g., std::memcmp),
//   bypassing character-by-character loop overhead.
//
// Recommended Use Cases:
//   - Default policy for both char and wchar_t.
//   - Case-sensitive identifiers, binary streams, and exact-match UTF-8 string data.
//   - High-throughput environments where case variations are not permitted.
//-------------------------------------------------------------------------------------------
template <SupportedChar CharT>
struct CaseSensitivePolicy
{
    static constexpr bool isCaseInsensitive = false; // Indicates exact binary equality without folding

    static constexpr bool IsValid(std::basic_string_view<CharT> text) noexcept
    {
        (void)text;
        return true;
    }

    inline CharT operator()(CharT character) const noexcept
    {
        return character;
    }
};

//-------------------------------------------------------------------------------------------
// AsciiCaseFoldPolicy
//
// Semantics:
//   Performs 7-bit ASCII case-folding, mapping lowercase ASCII characters ('a'-'z')
//   to their uppercase equivalents ('A'-'Z'). Leaves code units >= 128 unchanged.
//
// Validation (IsValid):
//   Inspects pattern text and rejects any string containing non-ASCII code units
//   (byte or wide value > 127) at pattern registration time. Note: While this protects
//   the pattern, consumers must independently guarantee that the target text searched
//   is also ASCII-compliant. Otherwise, multi-byte UTF-8 sequence bytes in the target 
//   text may be sliced by byte-level single-character wildcard ('?') evaluations.
//
// Performance & SIMD:
//   Accelerated via branchless SIMD vector arithmetic (CaseFold). Evaluates ASCII
//   characters concurrently across 128-bit, 256-bit, or 512-bit vector registers.
//   Avoids CRT locale synchronization locks and memory allocations.
//
// Recommended Use Cases:
//   - Protocol headers (HTTP, SIP, MIME), URI schemes, and network commands.
//   - DOS/NTFS 8.3 names, file extensions (.txt, .dll, .exe), and ASCII path parsing.
//   - Systems-level string matching requiring safe, high-throughput case-insensitivity.
//-------------------------------------------------------------------------------------------
template <SupportedChar CharT>
struct AsciiCaseFoldPolicy
{
    static constexpr bool isCaseInsensitive = true; // Enables ASCII case-folding transformations

    static inline bool IsValid(std::basic_string_view<CharT> text) noexcept
    {
        for (CharT character : text)
        {
            if constexpr (std::same_as<CharT, char>)
            {
                if (static_cast<unsigned char>(character) > 127)
                {
                    return false;
                }
            }
            else
            {
                if (static_cast<std::make_unsigned_t<wchar_t>>(character) > 127)
                {
                    return false;
                }
            }
        }

        return true;
    }

    inline CharT operator()(CharT character) const noexcept
    {
        if constexpr (std::same_as<CharT, char>)
        {
            const auto unsignedChar = static_cast<unsigned char>(character);
            if (unsignedChar >= 'a' && unsignedChar <= 'z')
            {
                return static_cast<char>(unsignedChar - 32);
            }

            return character;
        }
        else
        {
            if (character >= L'a' && character <= L'z')
            {
                return static_cast<wchar_t>(character - 32);
            }

            return character;
        }
    }
};

//-------------------------------------------------------------------------------------------
// UnicodeCaseFoldPolicy
//
// Semantics:
//   Provides scalar wide-character case-folding, mapping localized lower-case
//   letters across Latin, Cyrillic, Greek, and other standard alphabets to uppercase
//   using towupper-compatible transformations.
//
// Validation (IsValid):
//   Always returns true. Constrained at compile-time via C++20 concepts to wchar_t;
//   attempting to instantiate this policy with char triggers a compilation failure,
//   preventing invalid single-byte case transformations on multi-byte UTF-8 text.
//
// Performance & SIMD:
//   Features a dual-mode vector pipeline. Fast-paths 7-bit ASCII blocks through pure
//   SIMD vector registers. If non-ASCII code units are detected in a chunk via
//   Traits::HasNonAscii, it falls back to scalar transformation for those lanes
//   before accumulating into the rolling hash.
//
// Recommended Use Cases:
//   - Standard Windows kernel/user-mode Unicode (UTF-16 LE) string matching.
//   - Localized file paths, user account names, and registry paths containing
//     international or accented characters.
//-------------------------------------------------------------------------------------------
template <SupportedChar CharT>
    requires std::same_as<CharT, wchar_t>
struct UnicodeCaseFoldPolicy
{
    static constexpr bool isCaseInsensitive = true; // Enables wide-character Unicode folding

    static constexpr bool IsValid(std::basic_string_view<CharT> text) noexcept
    {
        (void)text;
        return true;
    }

    inline wchar_t operator()(wchar_t character) const noexcept
    {
        if (character <= 127)
        {
            return character - ((character >= L'a' && character <= L'z') << 5);
        }

        return static_cast<wchar_t>(std::towupper(static_cast<wint_t>(character)));
    }
};

//-------------------------------------------------------------------------------------------
// CustomCasePolicy<CharT, TFunctor>
//
// Semantics:
//   Adapter policy delegating character normalization to a caller-supplied functor
//   or callable object. Allows injection of specialized mapping algorithms.
//
// Validation (IsValid):
//   Delegates validation to the client. Defaults to true unless specialized by the
//   consumer to enforce domain-specific bounds (e.g., rejecting surrogate pairs).
//
// Performance & SIMD:
//   Inlines the client functor directly into scalar backtracking loops and SIMD
//   lane unwinding. Eliminates virtual function call indirection, function pointer
//   lookups, and Branch Target Buffer (BTB) pipeline stalls.
//
// Recommended Use Cases:
//   - Table-driven upcasing routines (e.g., 64K / 128 KB flat lookup tables mirroring
//     NTFS / FsRtlUpcaseUnicodeChar in Windows kernel architectures).
//   - External ICU library integration or custom collation tables.
//   - Two-stage trie/page-table upcase policies designed for constrained memory footprints.
//-------------------------------------------------------------------------------------------
template <SupportedChar CharT, typename TFunctor>
struct CustomCasePolicy
{
    static constexpr bool isCaseInsensitive = true; // Indicates custom transformation policy
    TFunctor              functor;                  // User-provided transformation functor or mapping callable

    explicit CustomCasePolicy(TFunctor userFunctor = TFunctor()) noexcept : functor(std::move(userFunctor))
    {
    }

    static constexpr bool IsValid(std::basic_string_view<CharT> text) noexcept
    {
        (void)text;
        return true;
    }

    inline CharT operator()(CharT character) const noexcept
    {
        return functor(character);
    }
};

//-------------------------------------------------------------------------------------------
// OS-Specific Directory Separator
//-------------------------------------------------------------------------------------------

namespace StringPatternMatchDetail
{
    // -------------------------------------------------------------------------------------------
    // Determines if a character is the strict native path separator for the target OS.
    // Used exclusively when WildcardScope::PathSegment is active to halt wildcard expansion.
    //
    // Parameters:
    //   character - The character being evaluated during the string iteration.
    //
    // Return:
    //   True if the character matches the natively defined path separator ('\' on Windows, 
    //   '/' on POSIX), false otherwise.
    // -------------------------------------------------------------------------------------------
    template <SupportedChar CharT>
    inline bool IsPathSeparator(CharT character) noexcept
    {
#if defined(_WIN32)
        return character == static_cast<CharT>('\\');
#else
        return character == static_cast<CharT>('/');
#endif
    }

    //-------------------------------------------------------------------------------------------
    // Transparent Identity Hash for std::unordered_map
    // Bypasses secondary hashing on already-mixed 64-bit polynomial hash values
    //-------------------------------------------------------------------------------------------
    struct FastHashKey
    {
        [[nodiscard]] inline size_t operator()(uint64_t value) const noexcept
        {
            return static_cast<size_t>(value);
        }
    };

    //-------------------------------------------------------------------------------------------
    // Hardware Vectorization Detection
    //-------------------------------------------------------------------------------------------

    // Identifies available CPU vectorization instruction sets at runtime.
    enum class SimdInstructionSet
    {
        Scalar = 0,
        Sse41  = 1,
        Avx2   = 2,
        Avx512 = 3,
        Neon   = 4
    };

#if !defined(_WIN32) && (defined(_M_X64) || defined(__x86_64__))
    // Fallback for GCC/Clang where _xgetbv might require specific target pragmas
    static inline uint64_t XgetbvLinuxMac(uint32_t index)
    {
        uint32_t eax, edx;
        __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(index));
        return (static_cast<uint64_t>(edx) << 32) | eax;
    }
#endif

    // -------------------------------------------------------------------------------------------
    // Performs runtime detection of CPU SIMD capabilities using CPUID on x86/x64,
    // or falls back to compile-time architecture definitions on ARM.
    //
    // Return:
    //   The optimal detected SIMD instruction set available on the host processor.
    // -------------------------------------------------------------------------------------------
    inline SimdInstructionSet DetectCpuSimd() noexcept
    {
        static SimdInstructionSet cachedSimd = []() -> SimdInstructionSet
        {
#if defined(_M_X64) || defined(__x86_64__)
            bool osxsave     = false;
            bool hasSse41    = false;
            bool hasAvx2     = false;
            bool hasAvx512F  = false;
            bool hasAvx512Bw = false;

#ifdef _WIN32
            int cpuInfo[4] = {0};
            __cpuid(cpuInfo, 1);
            hasSse41 = (cpuInfo[2] & (1 << 19)) != 0;
            osxsave  = (cpuInfo[2] & (1 << 27)) != 0;

            __cpuidex(cpuInfo, 7, 0);
            hasAvx2     = (cpuInfo[1] & (1 << 5)) != 0;
            hasAvx512F  = (cpuInfo[1] & (1 << 16)) != 0;
            hasAvx512Bw = (cpuInfo[1] & (1 << 30)) != 0;
#else
            unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
            __get_cpuid(1, &eax, &ebx, &ecx, &edx);
            hasSse41 = (ecx & (1 << 19)) != 0;
            osxsave  = (ecx & (1 << 27)) != 0;

            __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx);
            hasAvx2     = (ebx & (1 << 5)) != 0;
            hasAvx512F  = (ebx & (1 << 16)) != 0;
            hasAvx512Bw = (ebx & (1 << 30)) != 0;
#endif
            if (osxsave)
            {
#ifdef _WIN32
                uint64_t xcr0 = _xgetbv(0);
#else
                uint64_t xcr0 = XgetbvLinuxMac(0);
#endif
                bool osAvxSupport = (xcr0 & 6) == 6;
                bool osAvx512Support = (xcr0 & 0xE6) == 0xE6;

                // Prioritize wider 512-bit vectors when supported by host CPU and OS
                if (hasAvx512F && hasAvx512Bw && osAvx512Support)
                {
                    return SimdInstructionSet::Avx512;
                }

                if (hasAvx2 && osAvxSupport)
                {
                    return SimdInstructionSet::Avx2;
                }
            }

            if (hasSse41)
            {
                return SimdInstructionSet::Sse41;
            }
            return SimdInstructionSet::Scalar;
#elif defined(_M_ARM64) || defined(__aarch64__) || defined(_M_ARM) || defined(__arm__)
            // ARM architectures natively mandate 128-bit NEON execution units
            return SimdInstructionSet::Neon;
#else
            return SimdInstructionSet::Scalar;
#endif
        }();

        return cachedSimd;
    }

    //-------------------------------------------------------------------------------------------
    // 64-bit SWAR Utilities
    //-------------------------------------------------------------------------------------------
    
    // -------------------------------------------------------------------------------------------
    // Checks if an entire 64-bit SWAR block is entirely ASCII.
    // Enables bypassing slow Unicode transformations for pure English segments.
    //
    // Parameters:
    //   w - The 64-bit block of characters.
    //
    // Return:
    //   True if every character in the block falls within the standard ASCII range.
    // -------------------------------------------------------------------------------------------
    template <typename TChar>
    inline bool IsAsciiSWAR(uint64_t w) noexcept
    {
        if constexpr (sizeof(TChar) == 1)
        {
            return (w & 0x8080808080808080ULL) == 0;
        }
        else
        {
            return (w & 0xFF80FF80FF80FF80ULL) == 0;
        }
    }

    // -------------------------------------------------------------------------------------------
    // Case-folds a fully ASCII 64-bit SWAR block inline via bitwise arithmetic.
    // WARNING: Callers must guarantee IsAsciiSWAR(w) is true prior to invocation to prevent 
    // destructive boundary arithmetic faults on higher-order code points.
    //
    // Parameters:
    //   w - The 64-bit block of purely ASCII characters.
    //
    // Return:
    //   The 64-bit block with all a-z characters converted to uppercase.
    // -------------------------------------------------------------------------------------------
    template <typename TChar>
    inline uint64_t ToUpperSWAR(uint64_t w) noexcept
    {
        if constexpr (sizeof(TChar) == 1)
        {
            uint64_t NotUnderA = w + 0x1F1F1F1F1F1F1F1FULL;
            uint64_t NotOverZ = 0xFAFAFAFAFAFAFAFAULL - w;
            uint64_t IsLower = NotUnderA & NotOverZ & 0x8080808080808080ULL;

            return w - ((IsLower >> 2) & 0x2020202020202020ULL);
        }
        else
        {
            uint64_t NotUnderA = w + 0x7F9F7F9F7F9F7F9FULL;
            uint64_t NotOverZ = 0x807A807A807A807AULL - w;
            uint64_t IsLower = NotUnderA & NotOverZ & 0x8000800080008000ULL;

            return w - ((IsLower >> 10) & 0x0020002000200020ULL);
        }
    }

    // Function pointer type for hardware-accelerated polynomial rolling hash backends.
    template <SupportedChar CharT, typename TPolicy>
    using HashFuncType = uint64_t (*)(const CharT* text,
                                      uint32_t     textLength,
                                      uint64_t     previousHash,
                                      TPolicy      policy) noexcept;

    // Function pointer type for hardware-accelerated case-insensitive comparison backends.
    template <SupportedChar CharT, typename TPolicy>
    using CompareFuncType = bool (*)(const CharT* upperText1,
                                     const CharT* text2,
                                     uint32_t     compareLength,
                                     TPolicy      policy) noexcept;

    //-------------------------------------------------------------------------------------------
    // Scalar Hash Implementations (Baseline & Tail Fallbacks)
    //-------------------------------------------------------------------------------------------

    // -------------------------------------------------------------------------------------------
    // Standard scalar polynomial rolling hash using a prime multiplier of 131.
    //
    // Parameters:
    //   text         - Pointer to the text buffer to be hashed.
    //   textLength   - The length of the text in characters.
    //   previousHash - The base hash seed or accumulation value.
    //   policy       - The casing/normalization policy.
    //
    // Return:
    //   The computed 64-bit hash representing the provided text buffer.
    // -------------------------------------------------------------------------------------------
    template <SupportedChar CharT, bool IsCaseInsensitive, typename TPolicy>
    inline uint64_t CalculateHashScalar(const CharT* text,
                                        uint32_t     textLength,
                                        uint64_t     previousHash,
                                        TPolicy      policy) noexcept
    {
        using UCharT = std::make_unsigned_t<CharT>;
        for (uint32_t i = 0; i < textLength; i++)
        {
            if constexpr (IsCaseInsensitive)
            {
                previousHash = (previousHash * 131ULL) + static_cast<uint64_t>(static_cast<UCharT>(policy(text[i])));
            }
            else
            {
                previousHash = (previousHash * 131ULL) + static_cast<uint64_t>(static_cast<UCharT>(text[i]));
            }
        }

        return previousHash;
    }

    // -------------------------------------------------------------------------------------------
    // Standard scalar prefix verification comparing normalized text.
    //
    // Parameters:
    //   upperText1    - Pointer to the first pre-normalized pattern text.
    //   text2         - Pointer to the raw target text requiring normalization.
    //   compareLength - The length of the texts in characters.
    //   policy        - The casing/normalization policy.
    //
    // Return:
    //   True if the texts match under the current case policy, false otherwise.
    // -------------------------------------------------------------------------------------------
    template <SupportedChar CharT, typename TPolicy>
    inline bool ComparePrefixCI_Scalar(const CharT* upperText1,
                                       const CharT* text2,
                                       uint32_t     compareLength,
                                       TPolicy      policy) noexcept
    {
        for (uint32_t i = 0; i < compareLength; i++)
        {
            if (upperText1[i] != policy(text2[i]))
            {
                return false;
            }
        }
        return true;
    }

//-------------------------------------------------------------------------------------------
// SIMD Traits Definitions
//-------------------------------------------------------------------------------------------
#define SPM_GENERATE_PROCESS_CASING(TARGET_ATTR) \
    template <SupportedChar CharT, typename TPolicy> \
    TARGET_ATTR static inline VecT ProcessCasing(VecT c, const CharT* text, uint32_t offset, const TPolicy& policy) \
    { \
        if constexpr (std::same_as<TPolicy, AsciiCaseFoldPolicy<CharT>>) \
        { \
            return CaseFold(c); \
        } \
        else if constexpr (std::same_as<CharT, wchar_t> && std::same_as<TPolicy, UnicodeCaseFoldPolicy<wchar_t>>) \
        { \
            if (HasNonAscii(c)) \
            { \
                CharT temp[Step]; \
                for (uint32_t j = 0; j < Step; j++) \
                { \
                    temp[j] = policy(text[offset + j]); \
                } \
                return Load<CharT>(temp); \
            } \
            return CaseFold(c); \
        } \
        else \
        { \
            CharT temp[Step]; \
            for (uint32_t j = 0; j < Step; j++) \
            { \
                temp[j] = policy(text[offset + j]); \
            } \
            return Load<CharT>(temp); \
        } \
    }

#define SPM_GENERATE_CALCULATE_HASH(TTraits) \
    using UCharT = std::make_unsigned_t<CharT>; \
    using VecT = typename TTraits::VecT; \
    uint64_t hash = previousHash; \
    uint32_t i = 0; \
    constexpr uint64_t M4   = 294499921ULL; \
    constexpr uint64_t M8   = M4 * M4; \
    constexpr uint64_t M12  = M8 * M4; \
    constexpr uint64_t M16  = M8 * M8; \
    constexpr uint64_t M24  = M16 * M8; \
    constexpr uint64_t M32  = M16 * M16; \
    constexpr uint32_t Step = TTraits::Step; \
    if constexpr (Step == 16) \
    { \
        for (; i + Step * 2 - 1 < textLength; i += Step * 2) \
        { \
            VecT c0 = TTraits::template Load<CharT>(text + i); \
            VecT c1 = TTraits::template Load<CharT>(text + i + Step); \
            if constexpr (IsCaseInsensitive) \
            { \
                c0 = TTraits::template ProcessCasing<CharT, TPolicy>(c0, text, i, policy); \
                c1 = TTraits::template ProcessCasing<CharT, TPolicy>(c1, text, i + Step, policy); \
            } \
            uint64_t b0 = TTraits::HashBlock(c0); \
            uint64_t b1 = TTraits::HashBlock(c1); \
            hash = (hash * M32) + (b0 * M16) + b1; \
        } \
    } \
    else \
    { \
        constexpr uint64_t MulA = (Step == 8) ? M32 : M16; \
        constexpr uint64_t MulB = (Step == 8) ? M24 : M12; \
        constexpr uint64_t MulC = (Step == 8) ? M16 : M8; \
        constexpr uint64_t MulD = (Step == 8) ? M8  : M4; \
        for (; i + Step * 4 - 1 < textLength; i += Step * 4) \
        { \
            VecT c0 = TTraits::template Load<CharT>(text + i); \
            VecT c1 = TTraits::template Load<CharT>(text + i + Step); \
            VecT c2 = TTraits::template Load<CharT>(text + i + Step * 2); \
            VecT c3 = TTraits::template Load<CharT>(text + i + Step * 3); \
            if constexpr (IsCaseInsensitive) \
            { \
                c0 = TTraits::template ProcessCasing<CharT, TPolicy>(c0, text, i, policy); \
                c1 = TTraits::template ProcessCasing<CharT, TPolicy>(c1, text, i + Step, policy); \
                c2 = TTraits::template ProcessCasing<CharT, TPolicy>(c2, text, i + Step * 2, policy); \
                c3 = TTraits::template ProcessCasing<CharT, TPolicy>(c3, text, i + Step * 3, policy); \
            } \
            uint64_t b0 = TTraits::HashBlock(c0); \
            uint64_t b1 = TTraits::HashBlock(c1); \
            uint64_t b2 = TTraits::HashBlock(c2); \
            uint64_t b3 = TTraits::HashBlock(c3); \
            hash = (hash * MulA) + (b0 * MulB) + (b1 * MulC) + (b2 * MulD) + b3; \
        } \
    } \
    constexpr uint64_t MulSingle = (Step == 16) ? M16 : ((Step == 8) ? M8 : M4); \
    for (; i + Step - 1 < textLength; i += Step) \
    { \
        VecT c0 = TTraits::template Load<CharT>(text + i); \
        if constexpr (IsCaseInsensitive) \
        { \
            c0 = TTraits::template ProcessCasing<CharT, TPolicy>(c0, text, i, policy); \
        } \
        hash = (hash * MulSingle) + TTraits::HashBlock(c0); \
    } \
    for (; i < textLength; i++) \
    { \
        if constexpr (IsCaseInsensitive) \
        { \
            hash = (hash * 131ULL) + static_cast<uint64_t>(static_cast<UCharT>(policy(text[i]))); \
        } \
        else \
        { \
            hash = (hash * 131ULL) + static_cast<uint64_t>(static_cast<UCharT>(text[i])); \
        } \
    } \
    return hash;

#define SPM_GENERATE_COMPARE_PREFIX(TTraits) \
    uint32_t i = 0; \
    constexpr uint32_t Step = TTraits::Step; \
    using VecT = typename TTraits::VecT; \
    for (; i + Step - 1 < compareLength; i += Step) \
    { \
        VecT c1 = TTraits::template Load<CharT>(upperText1 + i); \
        VecT rawText = TTraits::template Load<CharT>(text2 + i); \
        VecT c2; \
        if constexpr (std::same_as<TPolicy, AsciiCaseFoldPolicy<CharT>>) \
        { \
            c2 = TTraits::CaseFold(rawText); \
        } \
        else if constexpr (std::same_as<CharT, wchar_t> && std::same_as<TPolicy, UnicodeCaseFoldPolicy<wchar_t>>) \
        { \
            if (TTraits::HasNonAscii(rawText)) \
            { \
                CharT temp[Step]; \
                for (uint32_t j = 0; j < Step; j++) \
                { \
                    temp[j] = policy(text2[i + j]); \
                } \
                c2 = TTraits::template Load<CharT>(temp); \
            } \
            else \
            { \
                c2 = TTraits::CaseFold(rawText); \
            } \
        } \
        else \
        { \
            CharT temp[Step]; \
            for (uint32_t j = 0; j < Step; j++) \
            { \
                temp[j] = policy(text2[i + j]); \
            } \
            c2 = TTraits::template Load<CharT>(temp); \
        } \
        if (!TTraits::CmpEq(c1, c2)) \
        { \
            return false; \
        } \
    } \
    for (; i < compareLength; i++) \
    { \
        if (upperText1[i] != policy(text2[i])) \
        { \
            return false; \
        } \
    } \
    return true;

#if defined(_M_X64) || defined(__x86_64__)

    struct TraitsSse41
    {
        using VecT = __m128i;
        static constexpr uint32_t Step = 4; // Number of code units evaluated per 128-bit vector iteration during polynomial hashing.
                                            // Prefix comparisons utilize wider byte-level strides.

        template <SupportedChar CharT>
        SPM_TARGET_SSE41 static inline VecT Load(const CharT* ptr)
        {
            if constexpr (std::is_same_v<CharT, char>)
            {
                return _mm_cvtepu8_epi32(_mm_cvtsi32_si128(*reinterpret_cast<const int32_t*>(ptr)));
            }
            else
            {
#if WCHAR_MAX <= 0xFFFF
                return _mm_cvtepu16_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr)));
#else
                return _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
#endif
            }
        }

        SPM_TARGET_SSE41 static inline bool HasNonAscii(VecT v)
        {
            return !_mm_testz_si128(v, _mm_set1_epi32(~127));
        }

        SPM_TARGET_SSE41 static inline VecT CaseFold(VecT v)
        {
            __m128i isLower = _mm_and_si128(_mm_cmpgt_epi32(v, _mm_set1_epi32('a' - 1)), _mm_cmplt_epi32(v, _mm_set1_epi32('z' + 1)));
            return _mm_sub_epi32(v, _mm_and_si128(isLower, _mm_set1_epi32(32)));
        }

        SPM_TARGET_SSE41 static inline bool CmpEq(VecT a,
                                                  VecT b)
        {
            return _mm_movemask_epi8(_mm_cmpeq_epi32(a, b)) == 0xFFFF;
        }

        SPM_TARGET_SSE41 static inline uint64_t HashBlock(VecT chars32)
        {
            const __m128i weights = _mm_setr_epi32(131 * 131 * 131, 131 * 131, 131, 1);

            __m128i mul1  = _mm_mul_epu32(chars32, weights);
            __m128i mul2  = _mm_mul_epu32(_mm_srli_si128(chars32, 4), _mm_srli_si128(weights, 4));
            __m128i sum64 = _mm_add_epi64(mul1, mul2);

            return static_cast<uint64_t>(_mm_cvtsi128_si64(_mm_add_epi64(sum64, _mm_unpackhi_epi64(sum64, sum64))));
        }

        SPM_GENERATE_PROCESS_CASING(SPM_TARGET_SSE41)

        template <SupportedChar CharT, bool IsCaseInsensitive, typename TPolicy>
        SPM_TARGET_SSE41 static uint64_t CalculateHash(const CharT* text,
                                                       uint32_t     textLength,
                                                       uint64_t     previousHash,
                                                       TPolicy      policy) noexcept
        {
            SPM_GENERATE_CALCULATE_HASH(TraitsSse41)
        }

        template <SupportedChar CharT, typename TPolicy>
        SPM_TARGET_SSE41 static bool ComparePrefix(const CharT* upperText1,
                                                   const CharT* text2,
                                                   uint32_t     compareLength,
                                                   TPolicy      policy) noexcept
        {
            SPM_GENERATE_COMPARE_PREFIX(TraitsSse41)
        }
    };

    struct TraitsAvx2
    {
        using VecT = __m256i;
        static constexpr uint32_t Step = 8; // Number of code units evaluated per 256-bit vector iteration during polynomial hashing.
                                            // Prefix comparisons utilize wider byte-level strides.

        template <SupportedChar CharT>
        SPM_TARGET_AVX2 static inline VecT Load(const CharT* ptr)
        {
            if constexpr (std::is_same_v<CharT, char>)
            {
                return _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr)));
            }
            else
            {
#if WCHAR_MAX <= 0xFFFF
                return _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr)));
#else
                return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
#endif
            }
        }

        SPM_TARGET_AVX2 static inline bool HasNonAscii(VecT v)
        {
            return !_mm256_testz_si256(v, _mm256_set1_epi32(~127));
        }

        SPM_TARGET_AVX2 static inline VecT CaseFold(VecT v)
        {
            __m256i isLower = _mm256_and_si256(_mm256_cmpgt_epi32(v, _mm256_set1_epi32('a' - 1)), _mm256_cmpgt_epi32(_mm256_set1_epi32('z' + 1), v));
            return _mm256_sub_epi32(v, _mm256_and_si256(isLower, _mm256_set1_epi32(32)));
        }

        SPM_TARGET_AVX2 static inline bool CmpEq(VecT a,
                                                 VecT b)
        {
            return static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi32(a, b))) == 0xFFFFFFFF;
        }

        SPM_TARGET_AVX2 static inline uint64_t HashBlock(VecT chars32)
        {
            const __m128i w128 = _mm_setr_epi32(131 * 131 * 131, 131 * 131, 131, 1);
            const __m256i w256 = _mm256_set_m128i(w128, w128);

            __m256i mul1  = _mm256_mul_epu32(chars32, w256);
            __m256i mul2  = _mm256_mul_epu32(_mm256_srli_si256(chars32, 4), _mm256_srli_si256(w256, 4));
            __m256i sum64 = _mm256_add_epi64(mul1, mul2);
            __m256i total = _mm256_add_epi64(sum64, _mm256_unpackhi_epi64(sum64, sum64));

            uint64_t low = static_cast<uint64_t>(_mm_cvtsi128_si64(_mm256_castsi256_si128(total)));
            uint64_t high = static_cast<uint64_t>(_mm_cvtsi128_si64(_mm256_extracti128_si256(total, 1)));
            constexpr uint64_t M4 = 294499921ULL;

            return (low * M4) + high;
        }

        SPM_GENERATE_PROCESS_CASING(SPM_TARGET_AVX2)

        template <SupportedChar CharT, bool IsCaseInsensitive, typename TPolicy>
        SPM_TARGET_AVX2 static uint64_t CalculateHash(const CharT* text,
                                                      uint32_t     textLength,
                                                      uint64_t     previousHash,
                                                      TPolicy      policy) noexcept
        {
            SPM_GENERATE_CALCULATE_HASH(TraitsAvx2)
        }

        template <SupportedChar CharT, typename TPolicy>
        SPM_TARGET_AVX2 static bool ComparePrefix(const CharT* upperText1,
                                                  const CharT* text2,
                                                  uint32_t     compareLength,
                                                  TPolicy      policy) noexcept
        {
            uint32_t i = 0;
            if constexpr (std::is_same_v<CharT, char>)
            {
                constexpr uint32_t VecStep = 32;

                const __m256i aMin    = _mm256_set1_epi8('a' - 1);
                const __m256i zMax    = _mm256_set1_epi8('z' + 1);
                const __m256i foldSub = _mm256_set1_epi8(32);

                for (; i + VecStep - 1 < compareLength; i += VecStep)
                {
                    __m256i c1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(upperText1 + i));
                    __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(text2 + i));

                    if constexpr (TPolicy::isCaseInsensitive)
                    {
                        __m256i isLower = _mm256_and_si256(_mm256_cmpgt_epi8(raw, aMin), _mm256_cmpgt_epi8(zMax, raw));
                        raw = _mm256_sub_epi8(raw, _mm256_and_si256(isLower, foldSub));
                    }

                    if (static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(c1, raw))) != 0xFFFFFFFF)
                    {
                        return false;
                    }
                }
            }
            else
            {
                constexpr uint32_t VecStep = 16;

                const __m256i aMin         = _mm256_set1_epi16('a' - 1);
                const __m256i zMax         = _mm256_set1_epi16('z' + 1);
                const __m256i foldSub      = _mm256_set1_epi16(32);
                const __m256i nonAsciiMask = _mm256_set1_epi16(static_cast<short>(~127));

                for (; i + VecStep - 1 < compareLength; i += VecStep)
                {
                    __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(text2 + i));

                    if constexpr (!std::same_as<TPolicy, AsciiCaseFoldPolicy<wchar_t>> && TPolicy::isCaseInsensitive)
                    {
                        if (!_mm256_testz_si256(raw, nonAsciiMask))
                        {
                            for (uint32_t k = 0; k < VecStep; ++k)
                            {
                                if (upperText1[i + k] != policy(text2[i + k]))
                                {
                                    return false;
                                }
                            }

                            continue;
                        }
                    }
                    __m256i c1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(upperText1 + i));

                    if constexpr (TPolicy::isCaseInsensitive)
                    {
                        __m256i isLower = _mm256_and_si256(_mm256_cmpgt_epi16(raw, aMin), _mm256_cmpgt_epi16(zMax, raw));
                        raw = _mm256_sub_epi16(raw, _mm256_and_si256(isLower, foldSub));
                    }

                    if (static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi16(c1, raw))) != 0xFFFFFFFF)
                    {
                        return false;
                    }
                }
            }

            for (; i < compareLength; i++)
            {
                if (upperText1[i] != policy(text2[i]))
                {
                    return false;
                }
            }

            return true;
        }
    };

    struct TraitsAvx512
    {
        using VecT = __m512i;
        static constexpr uint32_t Step = 16; // Number of code units evaluated per 512-bit vector iteration during polynomial hashing.
                                             // Prefix comparisons utilize wider byte-level strides.

        template <SupportedChar CharT>
        SPM_TARGET_AVX512 static inline VecT Load(const CharT* ptr)
        {
            if constexpr (std::is_same_v<CharT, char>)
            {
                return _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr)));
            }
            else
            {
#if WCHAR_MAX <= 0xFFFF
                return _mm512_cvtepu16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr)));
#else
                return _mm512_loadu_si512(reinterpret_cast<const void*>(ptr));
#endif
            }
        }

        SPM_TARGET_AVX512 static inline bool HasNonAscii(VecT v)
        {
            return _mm512_test_epi32_mask(v, _mm512_set1_epi32(~127)) != 0;
        }

        SPM_TARGET_AVX512 static inline VecT CaseFold(VecT v)
        {
            __mmask16 isLower = _mm512_cmpgt_epi32_mask(v, _mm512_set1_epi32('a' - 1)) & _mm512_cmplt_epi32_mask(v, _mm512_set1_epi32('z' + 1));
            return _mm512_mask_sub_epi32(v, isLower, v, _mm512_set1_epi32(32));
        }

        SPM_TARGET_AVX512 static inline bool CmpEq(VecT a,
                                                   VecT b)
        {
            return _mm512_cmpneq_epi32_mask(a, b) == 0;
        }

        SPM_TARGET_AVX512 static inline uint64_t HashBlock(VecT chars32)
        {
            const __m512i w512 = _mm512_broadcast_i32x4(_mm_setr_epi32(131 * 131 * 131, 131 * 131, 131, 1));

            __m512i mul1    = _mm512_mul_epu32(chars32, w512);
            __m512i mul2    = _mm512_mul_epu32(_mm512_srli_epi64(chars32, 32), _mm512_srli_epi64(w512, 32));
            __m512i sum64   = _mm512_add_epi64(mul1, mul2);
            __m512i total   = _mm512_add_epi64(sum64, _mm512_unpackhi_epi64(sum64, sum64));
            __m256i low256  = _mm512_castsi512_si256(total);
            __m256i high256 = _mm512_extracti64x4_epi64(total, 1);

            constexpr uint64_t M4 = 294499921ULL;
            constexpr uint64_t M8 = M4 * M4;

            uint64_t b0 = static_cast<uint64_t>(_mm_cvtsi128_si64(_mm256_castsi256_si128(low256)));
            uint64_t b1 = static_cast<uint64_t>(_mm_cvtsi128_si64(_mm256_extracti128_si256(low256, 1)));
            uint64_t b2 = static_cast<uint64_t>(_mm_cvtsi128_si64(_mm256_castsi256_si128(high256)));
            uint64_t b3 = static_cast<uint64_t>(_mm_cvtsi128_si64(_mm256_extracti128_si256(high256, 1)));
            return (b0 * M8 * M4) + (b1 * M8) + (b2 * M4) + b3;
        }

        SPM_GENERATE_PROCESS_CASING(SPM_TARGET_AVX512)

        template <SupportedChar CharT, bool IsCaseInsensitive, typename TPolicy>
        SPM_TARGET_AVX512 static uint64_t CalculateHash(const CharT* text,
                                                        uint32_t     textLength,
                                                        uint64_t     previousHash,
                                                        TPolicy      policy) noexcept
        {
            SPM_GENERATE_CALCULATE_HASH(TraitsAvx512)
        }

        template <SupportedChar CharT, typename TPolicy>
        SPM_TARGET_AVX512 static bool ComparePrefix(const CharT* upperText1,
                                                    const CharT* text2,
                                                    uint32_t     compareLength,
                                                    TPolicy      policy) noexcept
        {
            uint32_t i = 0;
            if constexpr (std::is_same_v<CharT, char>)
            {
                constexpr uint32_t VecStep = 64;

                const __m512i aMin    = _mm512_set1_epi8('a' - 1);
                const __m512i zMax    = _mm512_set1_epi8('z' + 1);
                const __m512i foldSub = _mm512_set1_epi8(32);

                for (; i + VecStep - 1 < compareLength; i += VecStep)
                {
                    __m512i c1 = _mm512_loadu_si512(reinterpret_cast<const void*>(upperText1 + i));
                    __m512i raw = _mm512_loadu_si512(reinterpret_cast<const void*>(text2 + i));

                    if constexpr (TPolicy::isCaseInsensitive)
                    {
                        __mmask64 isLower = _mm512_cmpgt_epi8_mask(raw, aMin) & _mm512_cmplt_epi8_mask(raw, zMax);
                        raw = _mm512_mask_sub_epi8(raw, isLower, raw, foldSub);
                    }
                    if (_mm512_cmpneq_epi8_mask(c1, raw) != 0)
                    {
                        return false;
                    }
                }
            }
            else
            {
            #if WCHAR_MAX <= 0xFFFF
                constexpr uint32_t VecStep = 32;

                const __m512i aMin         = _mm512_set1_epi16('a' - 1);
                const __m512i zMax         = _mm512_set1_epi16('z' + 1);
                const __m512i foldSub      = _mm512_set1_epi16(32);
                const __m512i nonAsciiMask = _mm512_set1_epi16(static_cast<short>(~127));

                for (; i + VecStep - 1 < compareLength; i += VecStep)
                {
                    __m512i raw = _mm512_loadu_si512(reinterpret_cast<const void*>(text2 + i));

                    if constexpr (!std::same_as<TPolicy, AsciiCaseFoldPolicy<wchar_t>> && TPolicy::isCaseInsensitive)
                    {
                        if (_mm512_test_epi16_mask(raw, nonAsciiMask) != 0)
                        {
                            for (uint32_t k = 0; k < VecStep; ++k)
                            {
                                if (upperText1[i + k] != policy(text2[i + k])) return false;
                            }
                            continue;
                        }
                    }
                    __m512i c1 = _mm512_loadu_si512(reinterpret_cast<const void*>(upperText1 + i));

                    if constexpr (TPolicy::isCaseInsensitive)
                    {
                        __mmask32 isLower = _mm512_cmpgt_epi16_mask(raw, aMin) & _mm512_cmplt_epi16_mask(raw, zMax);
                        raw = _mm512_mask_sub_epi16(raw, isLower, raw, foldSub);
                    }
                    if (_mm512_cmpneq_epi16_mask(c1, raw) != 0)
                    {
                        return false;
                    }
                }
            #else
                constexpr uint32_t VecStep = 16;

                const __m512i aMin         = _mm512_set1_epi32('a' - 1);
                const __m512i zMax         = _mm512_set1_epi32('z' + 1);
                const __m512i foldSub      = _mm512_set1_epi32(32);
                const __m512i nonAsciiMask = _mm512_set1_epi32(~127);

                for (; i + VecStep - 1 < compareLength; i += VecStep)
                {
                    __m512i raw = _mm512_loadu_si512(reinterpret_cast<const void*>(text2 + i));

                    if constexpr (!std::same_as<TPolicy, AsciiCaseFoldPolicy<wchar_t>> && TPolicy::isCaseInsensitive)
                    {
                        if (_mm512_test_epi32_mask(raw, nonAsciiMask) != 0)
                        {
                            for (uint32_t k = 0; k < VecStep; ++k)
                            {
                                if (upperText1[i + k] != policy(text2[i + k])) return false;
                            }
                            continue;
                        }
                    }
                    __m512i c1 = _mm512_loadu_si512(reinterpret_cast<const void*>(upperText1 + i));

                    if constexpr (TPolicy::isCaseInsensitive)
                    {
                        __mmask16 isLower = _mm512_cmpgt_epi32_mask(raw, aMin) & _mm512_cmplt_epi32_mask(raw, zMax);
                        raw = _mm512_mask_sub_epi32(raw, isLower, raw, foldSub);
                    }

                    if (_mm512_cmpneq_epi32_mask(c1, raw) != 0)
                    {
                        return false;
                    }
                }
            #endif
            }

            // Process tail scalar remnants
            for (; i < compareLength; i++)
            {
                if (upperText1[i] != policy(text2[i]))
                {
                    return false;
                }
            }

            return true;
        }
    };

#elif defined(_M_ARM64) || defined(__aarch64__) || defined(_M_ARM) || defined(__arm__)

    struct TraitsNeon
    {
        using VecT = uint32x4_t;
        static constexpr uint32_t Step = 4; // Number of code units evaluated per 128-bit vector iteration during polynomial hashing.
                                            // Prefix comparisons utilize wider byte-level strides.

        template <SupportedChar CharT>
        static inline VecT Load(const CharT* ptr)
        {
            if constexpr (std::is_same_v<CharT, char>)
            {
                uint32_t val;
                std::memcpy(&val, ptr, sizeof(val));
                return vmovl_u16(vget_low_u16(vmovl_u8(vreinterpret_u8_u32(vset_lane_u32(val, vdup_n_u32(0), 0)))));
            }
            else
            {
#if WCHAR_MAX <= 0xFFFF
                uint64_t val;
                std::memcpy(&val, ptr, sizeof(val));
                return vmovl_u16(vcreate_u16(val));
#else
                uint32x4_t val;
                std::memcpy(&val, ptr, sizeof(val));
                return val;
#endif
            }
        }

        static inline bool HasNonAscii(VecT v)
        {
#if defined(__aarch64__)
            return vmaxvq_u32(v) > 127;
#else
            uint32x2_t maxHalves = vmax_u32(vget_low_u32(v), vget_high_u32(v));
            uint32x2_t maxAll    = vpmax_u32(maxHalves, maxHalves);
            return vget_lane_u32(maxAll, 0) > 127;
#endif
        }

        static inline VecT CaseFold(VecT v)
        {
            uint32x4_t isLower = vandq_u32(vcgtq_u32(v, vdupq_n_u32('a' - 1)), vcgtq_u32(vdupq_n_u32('z' + 1), v));
            return vsubq_u32(v, vandq_u32(isLower, vdupq_n_u32(32)));
        }

        static inline bool CmpEq(VecT a,
                                 VecT b)
        {
            return vminvq_u32(vceqq_u32(a, b)) == 0xFFFFFFFF;
        }

        static inline uint64_t HashBlock(VecT chars32)
        {
            const uint32_t weightsArr[4] = {131 * 131 * 131, 131 * 131, 131, 1};

            uint32x4_t weights = vld1q_u32(weightsArr);
            uint64x2_t mulLow  = vmull_u32(vget_low_u32(chars32), vget_low_u32(weights));
            uint64x2_t mulHigh = vmull_u32(vget_high_u32(chars32), vget_high_u32(weights));
            uint64x2_t sum64   = vaddq_u64(mulLow, mulHigh);
#if defined(__aarch64__)
            return vaddvq_u64(sum64);
#else
            return vgetq_lane_u64(sum64, 0) + vgetq_lane_u64(sum64, 1);
#endif
        }

        SPM_GENERATE_PROCESS_CASING(SPM_TARGET_NEON)

        template <SupportedChar CharT, bool IsCaseInsensitive, typename TPolicy>
        static uint64_t CalculateHash(const CharT* text,
                                      uint32_t     textLength,
                                      uint64_t     previousHash,
                                      TPolicy      policy) noexcept
        {
            SPM_GENERATE_CALCULATE_HASH(TraitsNeon)
        }

        template <SupportedChar CharT, typename TPolicy>
        static bool ComparePrefix(const CharT* upperText1,
                                  const CharT* text2,
                                  uint32_t     compareLength,
                                  TPolicy      policy) noexcept
        {
            SPM_GENERATE_COMPARE_PREFIX(TraitsNeon)
        }
    };

#endif

    //-------------------------------------------------------------------------------------------
    // Centralized Function Pointer Dispatch Resolvers (O(1) Array Lookup)
    //-------------------------------------------------------------------------------------------

    //-------------------------------------------------------------------------------------------
    // Resolves the optimal hash function based on compile-time types, runtime CPU capabilities,
    // and requested optimization constraints.
    //
    // Dispatch Table Mechanics & Architecture Fallbacks:
    // The backend arrays (ciBackends and csBackends) map directly to the SimdInstructionSet enum:
    //   [0] = Scalar, [1] = Sse41, [2] = Avx2, [3] = Avx512, [4] = Neon.
    // To achieve zero-overhead O(1) branchless indexing via static_cast<size_t>(simd), every
    // instruction set index must contain a valid callable function pointer across all platforms:
    //   - On x86/x64: NEON (index 4) cannot compile or execute; it is populated with the portable
    //     scalar hash backend as a safe fallback.
    //   - On ARM/ARM64: SSE4.1, AVX2, and AVX-512 (indices 1, 2, 3) cannot compile or execute;
    //     they are populated with the scalar hash backend as safe fallbacks.
    //   - On other architectures (#else): All non-scalar slots (1 through 4) fall back directly
    //     to CalculateHashScalar to prevent out-of-bounds table indexing or undefined compilation.
    //
    // Parameters:
    //   mode - The selected SIMD optimization constraint for the resolution process.
    //
    // Return:
    //   A function pointer strictly tailored for rolling polynomial block hashing.
    //-------------------------------------------------------------------------------------------
    template <SupportedChar CharT, typename TPolicy>
    inline HashFuncType<CharT, TPolicy> ResolveHashBackend(OptimizationMode mode) noexcept
    {
        if (mode == OptimizationMode::Scalar)
        {
            if constexpr (TPolicy::isCaseInsensitive)
            {
                return &CalculateHashScalar<CharT, true, TPolicy>;
            }
            else
            {
                return &CalculateHashScalar<CharT, false, TPolicy>;
            }
        }

        const SimdInstructionSet simd = DetectCpuSimd();

        if constexpr (TPolicy::isCaseInsensitive)
        {
            static const HashFuncType<CharT, TPolicy> ciBackends[] =
                {
                    &CalculateHashScalar<CharT, true, TPolicy>,         // Index 0: SimdInstructionSet::Scalar (Baseline portable loop)
#if defined(_M_X64) || defined(__x86_64__)
                    &TraitsSse41::CalculateHash<CharT, true, TPolicy>,  // Index 1: SimdInstructionSet::Sse41 (128-bit SSE4.1 pipeline)
                    &TraitsAvx2::CalculateHash<CharT, true, TPolicy>,   // Index 2: SimdInstructionSet::Avx2 (256-bit AVX2 pipeline)
                    &TraitsAvx512::CalculateHash<CharT, true, TPolicy>, // Index 3: SimdInstructionSet::Avx512 (512-bit AVX-512 pipeline)
                    &CalculateHashScalar<CharT, true, TPolicy>          // Index 4: SimdInstructionSet::Neon (Fallback: NEON unavailable on x86)
#elif defined(_M_ARM64) || defined(__aarch64__) || defined(_M_ARM) || defined(__arm__)
                    &CalculateHashScalar<CharT, true, TPolicy>,         // Index 1: SimdInstructionSet::Sse41 (Fallback: SSE4.1 unavailable on ARM)
                    &CalculateHashScalar<CharT, true, TPolicy>,         // Index 2: SimdInstructionSet::Avx2 (Fallback: AVX2 unavailable on ARM)
                    &CalculateHashScalar<CharT, true, TPolicy>,         // Index 3: SimdInstructionSet::Avx512 (Fallback: AVX-512 unavailable on ARM)
                    &TraitsNeon::CalculateHash<CharT, true, TPolicy>    // Index 4: SimdInstructionSet::Neon (128-bit ARM NEON pipeline)
#else
                    &CalculateHashScalar<CharT, true, TPolicy>,         // Index 1: SimdInstructionSet::Sse41 (Fallback: Non-x86/ARM platform)
                    &CalculateHashScalar<CharT, true, TPolicy>,         // Index 2: SimdInstructionSet::Avx2 (Fallback: Non-x86/ARM platform)
                    &CalculateHashScalar<CharT, true, TPolicy>,         // Index 3: SimdInstructionSet::Avx512 (Fallback: Non-x86/ARM platform)
                    &CalculateHashScalar<CharT, true, TPolicy>          // Index 4: SimdInstructionSet::Neon (Fallback: Non-x86/ARM platform)
#endif
                };
            return ciBackends[static_cast<size_t>(simd)];
        }
        else
        {
            static const HashFuncType<CharT, TPolicy> csBackends[] =
                {
                    &CalculateHashScalar<CharT, false, TPolicy>,         // Index 0: SimdInstructionSet::Scalar (Baseline portable loop)
#if defined(_M_X64) || defined(__x86_64__)
                    &TraitsSse41::CalculateHash<CharT, false, TPolicy>,  // Index 1: SimdInstructionSet::Sse41 (128-bit SSE4.1 pipeline)
                    &TraitsAvx2::CalculateHash<CharT, false, TPolicy>,   // Index 2: SimdInstructionSet::Avx2 (256-bit AVX2 pipeline)
                    &TraitsAvx512::CalculateHash<CharT, false, TPolicy>, // Index 3: SimdInstructionSet::Avx512 (512-bit AVX-512 pipeline)
                    &CalculateHashScalar<CharT, false, TPolicy>          // Index 4: SimdInstructionSet::Neon (Fallback: NEON unavailable on x86)
#elif defined(_M_ARM64) || defined(__aarch64__) || defined(_M_ARM) || defined(__arm__)
                    &CalculateHashScalar<CharT, false, TPolicy>,         // Index 1: SimdInstructionSet::Sse41 (Fallback: SSE4.1 unavailable on ARM)
                    &CalculateHashScalar<CharT, false, TPolicy>,         // Index 2: SimdInstructionSet::Avx2 (Fallback: AVX2 unavailable on ARM)
                    &CalculateHashScalar<CharT, false, TPolicy>,         // Index 3: SimdInstructionSet::Avx512 (Fallback: AVX-512 unavailable on ARM)
                    &TraitsNeon::CalculateHash<CharT, false, TPolicy>    // Index 4: SimdInstructionSet::Neon (128-bit ARM NEON pipeline)
#else
                    &CalculateHashScalar<CharT, false, TPolicy>,         // Index 1: SimdInstructionSet::Sse41 (Fallback: Non-x86/ARM platform)
                    &CalculateHashScalar<CharT, false, TPolicy>,         // Index 2: SimdInstructionSet::Avx2 (Fallback: Non-x86/ARM platform)
                    &CalculateHashScalar<CharT, false, TPolicy>,         // Index 3: SimdInstructionSet::Avx512 (Fallback: Non-x86/ARM platform)
                    &CalculateHashScalar<CharT, false, TPolicy>          // Index 4: SimdInstructionSet::Neon (Fallback: Non-x86/ARM platform)
#endif
                };
            return csBackends[static_cast<size_t>(simd)];
        }
    }

    //-------------------------------------------------------------------------------------------
    // Resolves the optimal collision verification backend, bypassing atomic guards during 
    // execution.
    //
    // Parameters:
    //   mode - The selected SIMD optimization constraint for the resolution process.
    //
    // Return:
    //   A function pointer evaluating case-insensitive normalized matching lengths.
    //-------------------------------------------------------------------------------------------
    template <SupportedChar CharT, typename TPolicy>
    inline CompareFuncType<CharT, TPolicy> ResolveCompareBackend(OptimizationMode mode) noexcept
    {
        if constexpr (!TPolicy::isCaseInsensitive)
        {
            return nullptr;
        }
        else
        {
            if (mode == OptimizationMode::Scalar)
            {
                return &ComparePrefixCI_Scalar<CharT, TPolicy>;
            }

            const SimdInstructionSet simd = DetectCpuSimd();

            static const CompareFuncType<CharT, TPolicy> compareBackends[] =
                {
                    &ComparePrefixCI_Scalar<CharT, TPolicy>,         // Index 0: SimdInstructionSet::Scalar (Baseline portable loop)
#if defined(_M_X64) || defined(__x86_64__)
                    &TraitsSse41::ComparePrefix<CharT, TPolicy>,     // Index 1: SimdInstructionSet::Sse41 (128-bit SSE4.1 pipeline)
                    &TraitsAvx2::ComparePrefix<CharT, TPolicy>,      // Index 2: SimdInstructionSet::Avx2 (256-bit AVX2 pipeline)
                    &TraitsAvx512::ComparePrefix<CharT, TPolicy>,    // Index 3: SimdInstructionSet::Avx512 (512-bit AVX-512 pipeline)
                    &ComparePrefixCI_Scalar<CharT, TPolicy>          // Index 4: SimdInstructionSet::Neon (Fallback: NEON unavailable on x86)
#elif defined(_M_ARM64) || defined(__aarch64__) || defined(_M_ARM) || defined(__arm__)
                    &ComparePrefixCI_Scalar<CharT, TPolicy>,         // Index 1: SimdInstructionSet::Sse41 (Fallback: SSE4.1 unavailable on ARM)
                    &ComparePrefixCI_Scalar<CharT, TPolicy>,         // Index 2: SimdInstructionSet::Avx2 (Fallback: AVX2 unavailable on ARM)
                    &ComparePrefixCI_Scalar<CharT, TPolicy>,         // Index 3: SimdInstructionSet::Avx512 (Fallback: AVX-512 unavailable on ARM)
                    &TraitsNeon::ComparePrefix<CharT, TPolicy>       // Index 4: SimdInstructionSet::Neon (128-bit ARM NEON pipeline)
#else
                    &ComparePrefixCI_Scalar<CharT, TPolicy>,         // Index 1: SimdInstructionSet::Sse41 (Fallback: Non-x86/ARM platform)
                    &ComparePrefixCI_Scalar<CharT, TPolicy>,         // Index 2: SimdInstructionSet::Avx2 (Fallback: Non-x86/ARM platform)
                    &ComparePrefixCI_Scalar<CharT, TPolicy>,         // Index 3: SimdInstructionSet::Avx512 (Fallback: Non-x86/ARM platform)
                    &ComparePrefixCI_Scalar<CharT, TPolicy>          // Index 4: SimdInstructionSet::Neon (Fallback: Non-x86/ARM platform)
#endif
                };
            return compareBackends[static_cast<size_t>(simd)];
        }
    }

    //-----------------------------------------------------------------------------------------------------------------------------------------------------------
    //                                                     Base Engine Implementation
    //-----------------------------------------------------------------------------------------------------------------------------------------------------------
    // Base internal implementation of the lock-free string matching engine.
    // Provides the core hash-existence filter, SWAR chunking, and rolling hash progression
    // abstracted away from the public wrapper's standard string_view types.
    //
    // Template Parameters:
    //   CharT    - The character type (e.g., wchar_t or char).
    //   TContext - The user-defined context payload type returned upon successful matches.
    //   TPolicy  - The policy encapsulating character casing, normalization, and validation.
    //-----------------------------------------------------------------------------------------------------------------------------------------------------------
    template <SupportedChar CharT, typename TContext, CasePolicy<CharT> TPolicy>
    class StringPatternMatchImpl
    {
    public:
        using StringType     = std::basic_string<CharT>;
        using StringViewType = std::basic_string_view<CharT>;

    protected:
        // Cache-aligned, bounds-eliminated struct linking separated pattern components to their context
        struct PatternEntry
        {
            uint64_t      prefixSignature;        // 64-bit SWAR/packed signature of prefix start for O(1) early rejection
            const CharT*  prefixData;             // Pointer to start of literal prefix string in m_stringArena
            const CharT*  suffixData;             // Pointer to start of wildcard suffix pattern in m_stringArena
            const CharT*  tailData;               // Pointer to start of trailing literal segment in m_stringArena
            uint32_t      prefixLength;           // Length of literal prefix in code units
            uint32_t      suffixLength;           // Length of wildcard suffix pattern in code units
            uint32_t      tailLength;             // Length of anchored literal tail in code units
            uint32_t      contextIndex;           // Index of user context in m_contextArena
            uint32_t      nextEntryIndex;         // Index of next pattern in collision chain (UINT32_MAX if last)
            uint32_t      id;                     // Monotonically increasing pattern registration ID
            uint16_t      minWildcardMatchLength; // Minimum required characters to satisfy non-'*' wildcard suffix
            WildcardScope scope;                  // Boundary scoping rule (Default vs PathSegment)
            uint8_t       padding[5];             // Explicit padding to achieve exact 64-byte L1 cache-line alignment
        };

        struct HashBucket
        {
            uint32_t headIndex; // Index of the first pattern in collision chain (UINT32_MAX if empty)
            uint32_t tailIndex; // Index of the last pattern in collision chain for FIFO append
        };

        // -------------------------------------------------------------------------------------------
        // Initializes the pattern matcher and resolves optimal backend function pointers.
        //
        // Parameters:
        //   mode   - Hardware acceleration mode setting.
        //   policy - The casing and normalization policy payload.
        // -------------------------------------------------------------------------------------------
        StringPatternMatchImpl(OptimizationMode mode,
                               TPolicy          policy) noexcept;

        // -------------------------------------------------------------------------------------------
        // Destroys the engine, freeing all resources intrinsically managed by standard containers.
        // -------------------------------------------------------------------------------------------
        ~StringPatternMatchImpl() noexcept = default;

    public:
        // -------------------------------------------------------------------------------------------
        // Parses a pattern, extracts its literal prefix, calculates its hash, and
        // maps it to the provided context.
        //
        // Parameters:
        //   pattern        - Pointer to the character array defining the wildcard pattern.
        //   patternLength  - The length of the pattern in characters.
        //   wildScope      - The boundary scope to apply for wildcard operators.
        //   patternContext - The user-defined context payload mapped to this specific pattern.
        //
        // Return:
        //   True if the pattern was successfully registered. Throws std::invalid_argument on policy violation.
        // -------------------------------------------------------------------------------------------
        bool AddPattern(const CharT*  pattern,
                        uint32_t      patternLength,
                        WildcardScope wildScope,
                        TContext&&    patternContext);

        // -------------------------------------------------------------------------------------------
        // Executes a single-match search, returning true and populating PatternContext
        // upon the first successful match.
        //
        // Parameters:
        //   text           - Pointer to the target text buffer to be evaluated.
        //   textLength     - The length of the target text in characters.
        //   patternContext - Out-pointer receiving the matched payload context on success.
        //
        // Return:
        //   True if a match was found and patternContext populated, false otherwise.
        // -------------------------------------------------------------------------------------------
        bool Search(const CharT* text,
                    uint32_t     textLength,
                    TContext*&   patternContext) const noexcept;

        // -------------------------------------------------------------------------------------------
        // Executes a multi-match search, appending all matching contexts into the vector.
        //
        // Parameters:
        //   text            - Pointer to the target text buffer to be evaluated.
        //   textLength      - The length of the target text in characters.
        //   patternContexts - Vector receiving all context payloads that successfully matched.
        //
        // Return:
        //   True if at least one match was found and appended, false otherwise.
        // -------------------------------------------------------------------------------------------
        bool Search(const CharT*            text,
                    uint32_t                textLength,
                    std::vector<TContext*>& patternContexts) const;

        // -------------------------------------------------------------------------------------------
        // Clears all hashed patterns and resets internal memory arenas.
        // -------------------------------------------------------------------------------------------
        void Clear() noexcept;

    private:
        // -------------------------------------------------------------------------------------------
        // Generates a 64-bit SWAR signature from the prefix of a given string.
        // Accelerates O(1) early rejection before invoking exact string comparisons.
        //
        // Parameters:
        //   str    - Pointer to the target text buffer.
        //   len    - The length of the text buffer in characters.
        //   policy - The casing and normalization policy to apply during signature generation.
        //
        // Return:
        //   The 64-bit packed signature of the string prefix.
        // -------------------------------------------------------------------------------------------
        inline uint64_t GetTextSignature(const CharT* str,
                                         uint32_t     len,
                                         TPolicy      policy) const noexcept;

        // -------------------------------------------------------------------------------------------
        // Masks a 64-bit SWAR signature to a specific character length.
        // Ensures signature comparisons do not read out-of-bounds trailing garbage.
        //
        // Parameters:
        //   sig    - The original 64-bit signature.
        //   length - The length in characters to mask against.
        //
        // Return:
        //   The length-masked 64-bit signature.
        // -------------------------------------------------------------------------------------------
        inline uint64_t MaskSignature(uint64_t sig,
                                      uint32_t length) const noexcept;

        // -------------------------------------------------------------------------------------------
        // Evaluates whether a given pattern's literal prefix matches the target text segment.
        // Utilizes 64-bit SWAR block early rejection when applicable.
        //
        // Parameters:
        //   pattern     - The pattern entry containing the literal prefix to evaluate.
        //   text        - Pointer to the target text buffer segment.
        //   matchLength - The length of the prefix in characters to compare.
        //
        // Return:
        //   True if the prefix completely matches the text segment, false otherwise.
        // -------------------------------------------------------------------------------------------
        inline bool MatchesPrefix(const PatternEntry& pattern,
                                  const CharT*        text,
                                  uint32_t            matchLength) const noexcept;

        // -------------------------------------------------------------------------------------------
        // Evaluates the wildcard suffix of a pattern against the remaining target text.
        // Routes the comparison to the appropriate wildcard state machine specialization.
        //
        // Parameters:
        //   pattern      - The pattern entry containing the wildcard suffix.
        //   suffixText   - Pointer to the remaining unmatched text segment.
        //   suffixLength - Character length of the remaining text.
        //
        // Return:
        //   True if the wildcard suffix matches the remaining text, false otherwise.
        // -------------------------------------------------------------------------------------------
        inline bool MatchesSuffix(const PatternEntry& pattern,
                                  const CharT*        suffixText,
                                  uint32_t            suffixLength) const noexcept;

        // -------------------------------------------------------------------------------------------
        // Evaluates whether a given pattern's anchored tail matches the end of the target text.
        //
        // Parameters:
        //   pattern    - The pattern entry containing the tail anchor.
        //   text       - Pointer to the target text buffer.
        //   textLength - Total character length of the target text.
        //
        // Return:
        //   True if the tail anchor matches or is empty, false otherwise.
        // -------------------------------------------------------------------------------------------
        inline bool MatchesTail(const PatternEntry& pattern,
                                const CharT*        text,
                                uint32_t            textLength) const noexcept;

        // -------------------------------------------------------------------------------------------
        // Internal execution engine that drives the rolling hash length progression
        // and fires the templated collector callback when matches are verified.
        //
        // Parameters:
        //   text       - Pointer to the target text.
        //   textLength - Character length of the target text.
        //   collector  - Functor executing context assignment when a match is found.
        //
        // Return:
        //   True if the early exit condition in the collector is met, false otherwise.
        // -------------------------------------------------------------------------------------------
        template <typename F>
        bool SearchInternal(const CharT* text,
                            uint32_t     textLength,
                            F&&          collector) const noexcept;

        // -------------------------------------------------------------------------------------------
        // Non-recursive state machine that evaluates the wildcard suffix against the string.
        //
        // Parameters:
        //   pattern            - Pointer to the wildcard suffix of the pattern.
        //   patternLength      - Character length of the wildcard suffix.
        //   textString         - Pointer to the remaining unmatched text segment.
        //   textStringLength   - Character length of the remaining text.
        //   remainingMinLength - Minimum matching characters required by the remaining pattern states.
        //
        // Return:
        //   True if the wildcard suffix wholly matches the remaining text, false otherwise.
        // -------------------------------------------------------------------------------------------
        template <bool IsCaseInsensitive, bool LimitWildScope>
        bool WildCardMatch(const CharT* pattern,
                           uint32_t     patternLength,
                           const CharT* textString,
                           uint32_t     textStringLength,
                           uint32_t     remainingMinLength) const noexcept;

    private:
        static constexpr uint32_t kHashFilterWords = 64; // 4096-bit filter array

        uint32_t m_patternCount         = 0;           // Total count of registered patterns in the engine
        uint32_t m_globalMinMatchLength = UINT32_MAX;  // Global minimum string length required across all registered patterns
        uint32_t m_globalMaxMatchLength = 0;           // Maximum literal prefix length observed across all patterns
        bool     m_wildCardPresent      = false;       // True if any registered pattern contains wildcard characters

        uint64_t m_hashFilter[kHashFilterWords] = {0}; // Hash-existence filter to instantly reject map misses
        uint64_t m_lengthFilter[4]              = {0}; // O(1) bitmask filter for rejecting unregistered prefix lengths < 256

        TPolicy m_policy; // Policy instance encapsulating character casing, normalization, and validation

        // Hardware-selected implementation pointers
        HashFuncType<CharT, TPolicy>    m_hashFunction    = nullptr; // Dispatched polynomial rolling hash function pointer
        CompareFuncType<CharT, TPolicy> m_compareFunction = nullptr; // Dispatched case-insensitive string comparison pointer

        std::vector<uint32_t>                                 m_lengthVector; // Sorted list of all known unique prefix lengths
        std::deque<StringType>                                m_stringArena;  // Stable deque storage backing pattern string_views
        std::deque<TContext>                                  m_contextArena; // Stable deque storage backing pattern user contexts
        std::vector<PatternEntry>                             m_patternArena; // Contiguous storage for pattern entries and collision chains
        std::unordered_map<uint64_t, HashBucket, FastHashKey> m_hashMap;      // Maps prefix hashes to intrusive bucket head/tail indices

        static constexpr CharT kStar     = static_cast<CharT>('*'); // Multi-character wildcard token ('*')
        static constexpr CharT kQuestion = static_cast<CharT>('?'); // Single-character wildcard token ('?')
        static constexpr CharT kEscape   = static_cast<CharT>('|'); // Literal escape token ('|')
    };

    //-------------------------------------------------------------------------------------------
    // Implementation
    //-------------------------------------------------------------------------------------------

    // -------------------------------------------------------------------------------------------
    // Initializes the pattern matcher and resolves optimal backend function pointers.
    //
    // Parameters:
    //   mode   - Hardware acceleration mode setting.
    //   policy - The casing and normalization policy payload.
    // -------------------------------------------------------------------------------------------
    template <SupportedChar CharT, typename TContext, CasePolicy<CharT> TPolicy>
    StringPatternMatchImpl<CharT, TContext, TPolicy>::StringPatternMatchImpl(OptimizationMode mode,
                                                                             TPolicy          policy) noexcept : m_policy(std::move(policy))
    {
        std::memset(m_hashFilter, 0, sizeof(m_hashFilter));
        std::memset(m_lengthFilter, 0, sizeof(m_lengthFilter));

        m_hashFunction = ResolveHashBackend<CharT, TPolicy>(mode);

        if constexpr (TPolicy::isCaseInsensitive)
        {
            m_compareFunction = ResolveCompareBackend<CharT, TPolicy>(mode);
        }
    }

    // -------------------------------------------------------------------------------------------
    // Parses a pattern, extracts its literal prefix, calculates its hash, and
    // maps it to the provided context.
    //
    // Parameters:
    //   pattern        - Pointer to the character array defining the wildcard pattern.
    //   patternLength  - The length of the pattern in characters.
    //   wildScope      - The boundary scope to apply for wildcard operators.
    //   patternContext - The user-defined context payload mapped to this specific pattern.
    //
    // Return:
    //   True if the pattern was successfully registered. Throws std::invalid_argument on policy 
    //   violation.
    // -------------------------------------------------------------------------------------------
    template <SupportedChar CharT, typename TContext, CasePolicy<CharT> TPolicy>
    bool StringPatternMatchImpl<CharT, TContext, TPolicy>::AddPattern(const CharT*  pattern,
                                                                      uint32_t      patternLength,
                                                                      WildcardScope wildScope,
                                                                      TContext&&    patternContext)
    {
        std::basic_string_view<CharT> patternView(pattern, patternLength);

        // Enforce policy validation; throw if characters are rejected by active policy
        if (!TPolicy::IsValid(patternView))
        {
            throw std::invalid_argument("Pattern contains characters rejected by the active CasePolicy.");
        }

        // Emplace the string into the deque to guarantee pointer stability for the view
        m_stringArena.emplace_back(pattern, patternLength);
        StringType& stableString = m_stringArena.back();

        // If case-insensitive mode is enabled, pre-normalize the registered pattern memory once
        if constexpr (TPolicy::isCaseInsensitive)
        {
            for (auto& character : stableString)
            {
                character = m_policy(character);
            }
        }

        StringViewType fullView(stableString);

        m_contextArena.emplace_back(std::move(patternContext));
        uint32_t ctxIndex = static_cast<uint32_t>(m_contextArena.size() - 1);

        auto isWildCard = [](const CharT character)
        {
            return character == static_cast<CharT>('*') || 
                   character == static_cast<CharT>('?') || 
                   character == static_cast<CharT>('|');
        };

        // Locate the first wildcard to partition the literal prefix from the suffix
        auto findIterator = std::find_if(fullView.begin(), fullView.end(), isWildCard);

        uint32_t prefixLength = static_cast<uint32_t>(std::distance(fullView.begin(), findIterator));
        uint32_t wildPartLength = patternLength - prefixLength;

        // Calculate minimum required match length (non-'*' elements) for the suffix
        uint32_t suffixMinLength = 0;
        for (size_t i = 0; i < wildPartLength; ++i)
        {
            CharT c = fullView[prefixLength + i];
            if (c == kEscape && i + 1 < wildPartLength &&
                (fullView[prefixLength + i + 1] == kStar || 
                 fullView[prefixLength + i + 1] == kQuestion || 
                 fullView[prefixLength + i + 1] == kEscape))
            {
                suffixMinLength++;
                i++;
            }
            else if (c != kStar)
            {
                suffixMinLength++;
            }
        }

        // Tail Anchoring: extract trailing literal segment if suffix ends with literal characters
        uint32_t tailStart = prefixLength + wildPartLength;
        uint32_t tailLen = 0;

        if (wildPartLength > 0)
        {
            while (tailStart > prefixLength)
            {
                CharT c = fullView[tailStart - 1];

                // Immediately break upon encountering an active control or escape symbol, avoiding
                // corrupted extraction bounds that cause false-negative memory comparisons later.
                if (c == kStar || c == kQuestion || c == kEscape)
                {
                    break;
                }

                if (tailStart >= 2 && fullView[tailStart - 2] == kEscape)
                {
                    break; // Do not extract escaped literals as raw tail anchor
                }

                tailStart--;
            }
            tailLen = (prefixLength + wildPartLength) - tailStart;
        }

        PatternEntry newEntry;
        newEntry.prefixData             = fullView.data();
        newEntry.prefixLength           = prefixLength;
        newEntry.suffixData             = (wildPartLength != 0) ? (fullView.data() + prefixLength) : nullptr;
        newEntry.suffixLength           = wildPartLength;
        newEntry.tailData               = (tailLen != 0) ? (fullView.data() + tailStart) : nullptr;
        newEntry.tailLength             = tailLen;
        newEntry.id                     = ++m_patternCount;
        newEntry.contextIndex           = ctxIndex;
        newEntry.minWildcardMatchLength = static_cast<uint16_t>(suffixMinLength);
        newEntry.scope                  = wildScope;
        newEntry.nextEntryIndex         = UINT32_MAX;

        // Cache prefix signature for O(1) evaluation bypass during search loops
        newEntry.prefixSignature = GetTextSignature(newEntry.prefixData, prefixLength, m_policy);

        uint32_t requiredLength = prefixLength + suffixMinLength;
        m_globalMinMatchLength  = std::min<uint32_t>(m_globalMinMatchLength, requiredLength);
        m_globalMaxMatchLength  = std::max<uint32_t>(m_globalMaxMatchLength, prefixLength);

        if (prefixLength < 256)
        {
            m_lengthFilter[prefixLength >> 6] |= (1ULL << (prefixLength & 63));
        }

        if (wildPartLength != 0)
        {
            m_wildCardPresent = true;
        }

        // Calculate hash of the extracted literal prefix (handles CI normalization via backend func)
        uint64_t hash = m_hashFunction(pattern, prefixLength, 0, m_policy);

        // Register hash-existence filter signature to accelerate search map bypass
        uint32_t filterIdx = (hash >> 6) & (kHashFilterWords - 1);
        m_hashFilter[filterIdx] |= (1ULL << (hash & 63));

        // Keep the length vector sorted for efficient rolling hash progression
        auto lengthIterator = std::lower_bound(m_lengthVector.begin(), m_lengthVector.end(), prefixLength);

        if (lengthIterator == m_lengthVector.end() || *lengthIterator != prefixLength)
        {
            m_lengthVector.insert(lengthIterator, prefixLength);
        }

        // Register pattern in the lookup bucket using intrusive chaining (FIFO Append)
        uint32_t newIndex = static_cast<uint32_t>(m_patternArena.size());
        auto mapIterator = m_hashMap.find(hash);

        if (mapIterator != m_hashMap.end())
        {
            // Append to the existing tail to preserve FIFO evaluation order.
            // This is done before push_back to ensure arena references are valid.
            uint32_t tailIndex = mapIterator->second.tailIndex;
            m_patternArena[tailIndex].nextEntryIndex = newIndex;

            // Update the bucket's tail tracker
            mapIterator->second.tailIndex = newIndex;
        }
        else
        {
            // First entry for this hash bucket
            m_hashMap[hash] = {newIndex, newIndex};
        }

        // Push to the contiguous arena. Because m_hashMap stores indices rather than pointers,
        // std::vector reallocations will not invalidate the map tracking.
        m_patternArena.push_back(std::move(newEntry));

        return true;
    }

    // -------------------------------------------------------------------------------------------
    // Clears all hashed patterns and resets internal memory arenas.
    // -------------------------------------------------------------------------------------------
    template <SupportedChar CharT, typename TContext, CasePolicy<CharT> TPolicy>
    void StringPatternMatchImpl<CharT, TContext, TPolicy>::Clear() noexcept
    {
        m_hashMap.clear();
        m_patternArena.clear();
        m_lengthVector.clear();
        m_stringArena.clear();
        m_contextArena.clear();

        m_patternCount         = 0;
        m_globalMinMatchLength = UINT32_MAX;
        m_globalMaxMatchLength = 0;
        m_wildCardPresent      = false;

        std::memset(m_hashFilter, 0, sizeof(m_hashFilter));
        std::memset(m_lengthFilter, 0, sizeof(m_lengthFilter));
    }

    // -------------------------------------------------------------------------------------------
    // Generates a 64-bit SWAR signature from the prefix of a given string.
    // Accelerates O(1) early rejection before invoking exact string comparisons.
    //
    // Parameters:
    //   str    - Pointer to the target text buffer.
    //   len    - The length of the text buffer in characters.
    //   policy - The casing and normalization policy to apply during signature generation.
    //
    // Return:
    //   The 64-bit packed signature of the string prefix.
    // -------------------------------------------------------------------------------------------
    template <SupportedChar CharT, typename TContext, CasePolicy<CharT> TPolicy>
    inline uint64_t StringPatternMatchImpl<CharT, TContext, TPolicy>::GetTextSignature(const CharT* str,
                                                                                       uint32_t     len,
                                                                                       TPolicy      policy) const noexcept
    {
        uint64_t sig = 0;
        constexpr uint32_t maxChars = sizeof(uint64_t) / sizeof(CharT);
        using UCharT = std::make_unsigned_t<CharT>;

        if (len >= maxChars)
        {
            std::memcpy(&sig, str, sizeof(uint64_t));

            if constexpr (TPolicy::isCaseInsensitive)
            {
                if (StringPatternMatchDetail::IsAsciiSWAR<CharT>(sig))
                {
                    return StringPatternMatchDetail::ToUpperSWAR<CharT>(sig);
                }

                uint64_t folded = 0;
                for (uint32_t i = 0; i < maxChars; ++i)
                {
                    uint32_t shift = i * sizeof(CharT) * 8;
                    if constexpr (std::endian::native == std::endian::big)
                    {
                        shift = (maxChars - 1 - i) * sizeof(CharT) * 8;
                    }

                    folded |= (static_cast<uint64_t>(static_cast<UCharT>(policy(str[i]))) << shift);
                }

                return folded;
            }

            return sig;
        }

        if (len > 0)
        {
            std::memcpy(&sig, str, len * sizeof(CharT));

            if constexpr (TPolicy::isCaseInsensitive)
            {
                if (StringPatternMatchDetail::IsAsciiSWAR<CharT>(sig))
                {
                    return StringPatternMatchDetail::ToUpperSWAR<CharT>(sig);
                }

                uint64_t folded = 0;
                for (uint32_t i = 0; i < len; ++i)
                {
                    uint32_t shift = i * sizeof(CharT) * 8;
                    if constexpr (std::endian::native == std::endian::big)
                    {
                        shift = (maxChars - 1 - i) * sizeof(CharT) * 8;
                    }

                    folded |= (static_cast<uint64_t>(static_cast<UCharT>(policy(str[i]))) << shift);
                }

                return folded;
            }
        }

        return sig;
    }

    // -------------------------------------------------------------------------------------------
    // Masks a 64-bit SWAR signature to a specific character length.
    // Ensures signature comparisons do not read out-of-bounds trailing garbage.
    //
    // Parameters:
    //   sig    - The original 64-bit signature.
    //   length - The length in characters to mask against.
    //
    // Return:
    //   The length-masked 64-bit signature.
    // -------------------------------------------------------------------------------------------
    template <SupportedChar CharT, typename TContext, CasePolicy<CharT> TPolicy>
    inline uint64_t StringPatternMatchImpl<CharT, TContext, TPolicy>::MaskSignature(uint64_t sig,
                                                                                    uint32_t length) const noexcept
    {
        constexpr uint32_t maxChars = sizeof(uint64_t) / sizeof(CharT);

        if (length >= maxChars)
        {
            return sig;
        }

        if (length == 0)
        {
            return 0;
        }

        uint64_t mask = (1ULL << (length * sizeof(CharT) * 8)) - 1;

        if constexpr (std::endian::native == std::endian::big)
        {
            // Align mask to the Most Significant Bytes (MSB) to match memcpy's memory-order layout
            mask <<= ((maxChars - length) * sizeof(CharT) * 8);
        }

        return sig & mask;
    }

    // -------------------------------------------------------------------------------------------
    // Evaluates whether a given pattern's literal prefix matches the target text segment.
    // Utilizes 64-bit SWAR block early rejection when applicable.
    //
    // Parameters:
    //   pattern     - The pattern entry containing the literal prefix to evaluate.
    //   text        - Pointer to the target text buffer segment.
    //   matchLength - The length of the prefix in characters to compare.
    //
    // Return:
    //   True if the prefix completely matches the text segment, false otherwise.
    // -------------------------------------------------------------------------------------------
    template <SupportedChar CharT, typename TContext, CasePolicy<CharT> TPolicy>
    inline bool StringPatternMatchImpl<CharT, TContext, TPolicy>::MatchesPrefix(const PatternEntry& pattern,
                                                                                const CharT*        text,
                                                                                uint32_t            matchLength) const noexcept
    {
        if (pattern.prefixLength != matchLength)
        {
            return false;
        }

        constexpr uint32_t maxSwarChars = sizeof(uint64_t) / sizeof(CharT);
        uint32_t offset = std::min<uint32_t>(matchLength, maxSwarChars);

        // Fast-path: String equality is fully proven by the 64-bit prefixSignature
        if (matchLength == offset)
        {
            return true;
        }

        uint32_t remaining = matchLength - offset;

        if constexpr (TPolicy::isCaseInsensitive)
        {
            return m_compareFunction(pattern.prefixData + offset, text + offset, remaining, m_policy);
        }
        else
        {
            return std::memcmp(pattern.prefixData + offset, text + offset, remaining * sizeof(CharT)) == 0;
        }
    }

    // -------------------------------------------------------------------------------------------
    // Evaluates whether a given pattern's anchored tail matches the end of the target text.
    //
    // Parameters:
    //   pattern    - The pattern entry containing the tail anchor.
    //   text       - Pointer to the target text buffer.
    //   textLength - Total character length of the target text.
    //
    // Return:
    //   True if the tail anchor matches or is empty, false otherwise.
    // -------------------------------------------------------------------------------------------
    template <SupportedChar CharT, typename TContext, CasePolicy<CharT> TPolicy>
    inline bool StringPatternMatchImpl<CharT, TContext, TPolicy>::MatchesTail(const PatternEntry& pattern,
                                                                              const CharT*        text,
                                                                              uint32_t            textLength) const noexcept
    {
        if (pattern.tailLength == 0)
        {
            return true;
        }

        if (textLength < pattern.tailLength)
        {
            return false;
        }

        const CharT* textTail = text + (textLength - pattern.tailLength);

        if constexpr (TPolicy::isCaseInsensitive)
        {
            return m_compareFunction(pattern.tailData, textTail, pattern.tailLength, m_policy);
        }
        else
        {
            return std::memcmp(pattern.tailData, textTail, pattern.tailLength * sizeof(CharT)) == 0;
        }
    }

    // -------------------------------------------------------------------------------------------
    // Evaluates the wildcard suffix of a pattern against the remaining target text.
    // Routes the comparison to the appropriate wildcard state machine specialization.
    //
    // Parameters:
    //   pattern      - The pattern entry containing the wildcard suffix.
    //   suffixText   - Pointer to the remaining unmatched text segment.
    //   suffixLength - Character length of the remaining text.
    //
    // Return:
    //   True if the wildcard suffix matches the remaining text, false otherwise.
    // -------------------------------------------------------------------------------------------
    template <SupportedChar CharT, typename TContext, CasePolicy<CharT> TPolicy>
    inline bool StringPatternMatchImpl<CharT, TContext, TPolicy>::MatchesSuffix(const PatternEntry& pattern,
                                                                                const CharT*        suffixText,
                                                                                uint32_t            suffixLength) const noexcept
    {
        const bool isSlashScoped = (pattern.scope == WildcardScope::PathSegment);

        // Compile-time evaluation of case branch removes jump instructions from the backtracking engine
        if constexpr (TPolicy::isCaseInsensitive)
        {
            if (isSlashScoped)
            {
                return WildCardMatch<true, true>(pattern.suffixData, 
                                                 pattern.suffixLength, 
                                                 suffixText, 
                                                 suffixLength, 
                                                 pattern.minWildcardMatchLength);
            }
            else
            {
                return WildCardMatch<true, false>(pattern.suffixData, 
                                                  pattern.suffixLength, 
                                                  suffixText, 
                                                  suffixLength, 
                                                  pattern.minWildcardMatchLength);
            }
        }
        else
        {
            if (isSlashScoped)
            {
                return WildCardMatch<false, true>(pattern.suffixData, 
                                                  pattern.suffixLength, 
                                                  suffixText, 
                                                  suffixLength, 
                                                  pattern.minWildcardMatchLength);
            }
            else
            {
                return WildCardMatch<false, false>(pattern.suffixData, 
                                                   pattern.suffixLength, 
                                                   suffixText, 
                                                   suffixLength, 
                                                   pattern.minWildcardMatchLength);
            }
        }
    }

    // -------------------------------------------------------------------------------------------
    // Executes a single-match search, returning true and populating PatternContext
    // upon the first successful match.
    //
    // Parameters:
    //   text           - Pointer to the target text buffer to be evaluated.
    //   textLength     - The length of the target text in characters.
    //   patternContext - Out-pointer receiving the matched payload context on success.
    //
    // Return:
    //   True if a match was found and patternContext populated, false otherwise.
    // -------------------------------------------------------------------------------------------
    template <SupportedChar CharT, typename TContext, CasePolicy<CharT> TPolicy>
    bool StringPatternMatchImpl<CharT, TContext, TPolicy>::Search(const CharT* text,
                                                                  uint32_t     textLength,
                                                                  TContext*&   patternContext) const noexcept
    {
        patternContext = nullptr;

        // Define a collector that stops on the first match
        auto collector = [&](const TContext* context) -> bool
        {
            patternContext = const_cast<TContext*>(context);
            return true;
        };

        return SearchInternal(text, textLength, std::move(collector));
    }

    // -------------------------------------------------------------------------------------------
    // Executes a multi-match search, appending all matching contexts into the vector.
    //
    // Parameters:
    //   text            - Pointer to the target text buffer to be evaluated.
    //   textLength      - The length of the target text in characters.
    //   patternContexts - Vector receiving all context payloads that successfully matched.
    //
    // Return:
    //   True if at least one match was found and appended, false otherwise.
    // -------------------------------------------------------------------------------------------
    template <SupportedChar CharT, typename TContext, CasePolicy<CharT> TPolicy>
    bool StringPatternMatchImpl<CharT, TContext, TPolicy>::Search(const CharT*            text,
                                                                  uint32_t                textLength,
                                                                  std::vector<TContext*>& patternContexts) const
    {
        patternContexts.clear();

        // Define a collector that continues processing after a match
        auto collector = [&](const TContext* context) -> bool
        {
            patternContexts.push_back(const_cast<TContext*>(context));
            return false;
        };

        SearchInternal(text, textLength, std::move(collector));

        return !patternContexts.empty();
    }

    // -------------------------------------------------------------------------------------------
    // Internal execution engine that drives the rolling hash length progression
    // and fires the templated collector callback when matches are verified.
    //
    // Parameters:
    //   text       - Pointer to the target text.
    //   textLength - Character length of the target text.
    //   collector  - Functor executing context assignment when a match is found.
    //
    // Return:
    //   True if the early exit condition in the collector is met, false otherwise.
    // -------------------------------------------------------------------------------------------
    template <SupportedChar CharT, typename TContext, CasePolicy<CharT> TPolicy>
    template <typename F>
    bool StringPatternMatchImpl<CharT, TContext, TPolicy>::SearchInternal(const CharT* text,
                                                                          uint32_t     textLength,
                                                                          F&&          collector) const noexcept
    {
        // Global minimum match length rejection guard
        if (m_lengthVector.empty() || textLength < m_globalMinMatchLength)
        {
            return false;
        }

        // Fast-path: If no wildcards exist in the registry, the engine operates as an exact string matcher.
        // Bypasses iterative length loops using O(1) length bitmasks and direct hash evaluation.
        if (!m_wildCardPresent)
        {
            if (textLength > m_globalMaxMatchLength)
            {
                return false;
            }

            if (textLength < 256)
            {
                if ((m_lengthFilter[textLength >> 6] & (1ULL << (textLength & 63))) == 0)
                {
                    return false;
                }
            }
            else if (!std::binary_search(m_lengthVector.begin(), m_lengthVector.end(), textLength))
            {
                return false;
            }

            uint64_t hashValue = m_hashFunction(text, textLength, 0, m_policy);

            uint32_t filterIdx = (hashValue >> 6) & (kHashFilterWords - 1);
            if ((m_hashFilter[filterIdx] & (1ULL << (hashValue & 63))) == 0)
            {
                return false;
            }

            auto mapIterator = m_hashMap.find(hashValue);
            if (mapIterator != m_hashMap.end())
            {
                uint32_t entryIndex = mapIterator->second.headIndex;
                uint64_t textSig    = GetTextSignature(text, textLength, m_policy);

                while (entryIndex != UINT32_MAX)
                {
                    const PatternEntry& currentPattern = m_patternArena[entryIndex];

                    if (currentPattern.prefixSignature == textSig)
                    {
                        if (MatchesPrefix(currentPattern, text, textLength))
                        {
                            if (collector(&m_contextArena[currentPattern.contextIndex]))
                            {
                                return true;
                            }
                        }
                    }

                    entryIndex = currentPattern.nextEntryIndex;
                }
            }

            return false;
        }

        uint64_t hashValue   = 0;
        uint32_t textIndex   = 0;
        uint64_t fullTextSig = 0;
        bool     sigComputed = false;

        // Iterate through all applicable registered prefix lengths starting from our optimized iterator
        for (auto it = m_lengthVector.begin(); it != m_lengthVector.end(); ++it)
        {
            uint32_t matchLength = *it;

            // Engine aborts immediately if the remaining registered lengths exceed the string itself
            if (matchLength > textLength)
            {
                break;
            }

            // Inlined micro-steps: bypass indirect function call overhead when advancing by small byte deltas
            uint32_t delta = matchLength - textIndex;
            if (delta <= 4)
            {
                for (uint32_t k = 0; k < delta; ++k)
                {
                    if constexpr (TPolicy::isCaseInsensitive)
                    {
                        hashValue = (hashValue * 131ULL) + static_cast<uint64_t>(static_cast<std::make_unsigned_t<CharT>>(m_policy(text[textIndex + k])));
                    }
                    else
                    {
                        hashValue = (hashValue * 131ULL) + static_cast<uint64_t>(static_cast<std::make_unsigned_t<CharT>>(text[textIndex + k]));
                    }
                }
            }
            else
            {
                hashValue = m_hashFunction(text + textIndex, delta, hashValue, m_policy);
            }
            textIndex = matchLength;

            // Fast-path rejection: Check hash-existence filter before touching hash map memory
            uint32_t filterIdx = (hashValue >> 6) & (kHashFilterWords - 1);
            if ((m_hashFilter[filterIdx] & (1ULL << (hashValue & 63))) == 0)
            {
                continue;
            }

            // Check if the computed hash exists in our registered pattern buckets
            auto mapIterator = m_hashMap.find(hashValue);

            if (mapIterator != m_hashMap.end())
            {
                // Evaluate all patterns that share this hash bucket to resolve collisions
                // Starting from headIndex guarantees FIFO evaluation order
                uint32_t entryIndex = mapIterator->second.headIndex;

                if (!sigComputed && matchLength > 0)
                {
                    fullTextSig = GetTextSignature(text, textLength, m_policy);
                    sigComputed = true;
                }

                uint64_t maskedSig = matchLength > 0 ? MaskSignature(fullTextSig, matchLength) : 0;

                while (entryIndex != UINT32_MAX)
                {
                    const PatternEntry& currentPattern = m_patternArena[entryIndex];

                    // Per-pattern minimum required match length guard
                    if (textLength < matchLength + currentPattern.minWildcardMatchLength)
                    {
                        entryIndex = currentPattern.nextEntryIndex;
                        continue;
                    }

                    // O(1) Cache-local signature comparison avoids branchy exact literal tests
                    if (currentPattern.prefixSignature == maskedSig)
                    {
                        // Verify the hash collision with an exact literal string comparison of the prefix
                        if (MatchesPrefix(currentPattern, text, matchLength))
                        {
                            bool isMatch = false;

                            // Branch 1: The matched prefix covers the entire input text
                            if (matchLength == textLength)
                            {
                                // Valid match only if the suffix is empty or contains ONLY trailing '*'
                                // Evaluates consecutive trailing wildcards (e.g., "**") to prevent truncation bugs.
                                if (currentPattern.suffixLength == 0)
                                {
                                    isMatch = true;
                                }
                                else
                                {
                                    isMatch = true;
                                    for (uint32_t i = 0; i < currentPattern.suffixLength; i++)
                                    {
                                        if (currentPattern.suffixData[i] != kStar)
                                        {
                                            isMatch = false;
                                            break;
                                        }
                                    }
                                }
                            }
                            // Branch 2: The prefix matched, but extra characters remain in the input text
                            else
                            {
                                // Tail anchor validation deferred until prefix succeeds
                                if (MatchesTail(currentPattern, text, textLength))
                                {
                                    // The remaining text characters must be evaluated against the wildcard suffix
                                    isMatch = MatchesSuffix(currentPattern, text + matchLength, textLength - matchLength);
                                }
                            }

                            if (isMatch && collector(&m_contextArena[currentPattern.contextIndex]))
                            {
                                return true;
                            }
                        }
                    }

                    // Advance to the next pattern in the collision chain
                    entryIndex = currentPattern.nextEntryIndex;
                }
            }
        }

        return false;
    }

// -------------------------------------------------------------------------------------------
    // Non-recursive state machine that evaluates the wildcard suffix against the string.
    //
    // Parameters:
    //   pattern            - Pointer to the wildcard suffix of the pattern.
    //   patternLength      - Character length of the wildcard suffix.
    //   textString         - Pointer to the remaining unmatched text segment.
    //   textStringLength   - Character length of the remaining text.
    //   remainingMinLength - Minimum matching characters required by the remaining pattern states.
    //
    // Return:
    //   True if the wildcard suffix wholly matches the remaining text, false otherwise.
    // -------------------------------------------------------------------------------------------
    template <SupportedChar CharT, typename TContext, CasePolicy<CharT> TPolicy>
    template <bool IsCaseInsensitive, bool LimitWildScope>
    bool StringPatternMatchImpl<CharT, TContext, TPolicy>::WildCardMatch(const CharT* pattern,
                                                                         uint32_t     patternLength,
                                                                         const CharT* textString,
                                                                         uint32_t     textStringLength,
                                                                         uint32_t     remainingMinLength) const noexcept
    {
        const CharT* s = textString;
        const CharT* p = pattern;
        const CharT* const textEnd    = textString + textStringLength;
        const CharT* const patternEnd = pattern + patternLength;

        // Pointers tracking the backtracking state upon encountering a mismatch
        const CharT* lastStarPattern        = nullptr;
        const CharT* lastStarString         = nullptr;
        uint32_t     starRemainingMinLength = 0;

        // Propels the evaluation cursor towards the required sequence without invoking
        // recursive state transitions inside the wildcard state machine loop.
        // Fuses directory scope checking and bounds finding into a single pass.
        auto fastForward = [&](CharT target) noexcept -> bool
        {
            if constexpr (!IsCaseInsensitive && !LimitWildScope)
            {
                if constexpr (std::is_same_v<CharT, char>)
                {
                    const void* found = std::memchr(s, target, textEnd - s);
                    if (found)
                    {
                        s = static_cast<const CharT*>(found);
                        return true;
                    }

                    s = textEnd;
                    return false;
                }
                else
                {
                    const wchar_t* found = std::wmemchr(s, target, textEnd - s);
                    if (found)
                    {
                        s = static_cast<const CharT*>(found);
                        return true;
                    }

                    s = textEnd;
                    return false;
                }
            }
            else
            {
                // Unrolled manual scan handles custom CI policies and path separator constraints
                while (s + 3 < textEnd)
                {
                    if constexpr (IsCaseInsensitive)
                    {
                        if (m_policy(s[0]) == target)
                        {
                            return true;
                        }

                        if constexpr (LimitWildScope) 
                        { 
                            if (IsPathSeparator(s[0])) [[unlikely]] 
                                return false; 
                        }

                        if (m_policy(s[1]) == target) 
                        { 
                            s += 1; 
                            return true; 
                        }

                        if constexpr (LimitWildScope)
                        {
                            if (IsPathSeparator(s[1])) [[unlikely]]
                            {
                                return false;
                            }
                        }

                        if (m_policy(s[2]) == target) 
                        { 
                            s += 2; 
                            return true; 
                        }

                        if constexpr (LimitWildScope) 
                        { 
                            if (IsPathSeparator(s[2])) [[unlikely]]
                            {
                                return false;
                            }
                        }

                        if (m_policy(s[3]) == target) 
                        { 
                            s += 3; 
                            return true; 
                        }

                        if constexpr (LimitWildScope) 
                        { 
                            if (IsPathSeparator(s[3])) [[unlikely]]
                            {
                                return false;
                            }
                        }
                    }
                    else
                    {
                        if (s[0] == target)
                        {
                            return true;
                        }

                        if constexpr (LimitWildScope) 
                        { 
                            if (IsPathSeparator(s[0])) [[unlikely]]
                            {
                                return false;
                            }
                        }

                        if (s[1] == target) 
                        { 
                            s += 1; 
                            return true; 
                        }

                        if constexpr (LimitWildScope) 
                        { 
                            if (IsPathSeparator(s[1])) [[unlikely]]
                            {
                                return false;
                            }
                        }

                        if (s[2] == target) 
                        { 
                            s += 2; 
                            return true; 
                        }

                        if constexpr (LimitWildScope) 
                        { 
                            if (IsPathSeparator(s[2])) [[unlikely]]
                            {
                                return false;
                            }
                        }

                        if (s[3] == target) 
                        { 
                            s += 3; 
                            return true; 
                        }

                        if constexpr (LimitWildScope) 
                        { 
                            if (IsPathSeparator(s[3])) [[unlikely]]
                            {
                                return false;
                            }
                        }
                    }

                    s += 4;
                }

                // Handle remaining trailing characters
                while (s < textEnd)
                {
                    if constexpr (IsCaseInsensitive)
                    {
                        if (m_policy(*s) == target) return true;
                    }
                    else
                    {
                        if (*s == target) return true;
                    }

                    if constexpr (LimitWildScope)
                    {
                        if (IsPathSeparator(*s)) [[unlikely]] return false;
                    }

                    s++;
                }

                return false;
            }
        };

        // Inspects post-wildcard symbol and triggers linear fast-forward scan
        auto advanceToNextTarget = [&]() noexcept -> bool
        {
            // Unpack an escaped character as the target if necessary
            if (p < patternEnd && *p != kQuestion)
            {
                CharT target = (*p == kEscape && p + 1 < patternEnd && 
                                (p[1] == kStar || 
                                 p[1] == kQuestion || 
                                 p[1] == kEscape)) ? p[1] : *p;
                return fastForward(target);
            }

            return true;
        };

        while (s < textEnd)
        {
            // Proceed to character evaluation only if bounds and pattern permit.
            // Preserves the branch prediction hint for the hot path.
            if (static_cast<size_t>(textEnd - s) >= remainingMinLength && p < patternEnd) [[likely]]
            {
                switch (*p)
                {
                case kStar:
                {
                    // Collapse consecutive '*' characters to optimize execution
                    do
                    {
                        p++;
                    } while (p < patternEnd && *p == kStar);

                    // Fast-path: A trailing '*' instantly matches the remainder of the string
                    if (p == patternEnd)
                    {
                        // Ensure we don't violate the directory path segment boundary if active
                        if constexpr (LimitWildScope)
                        {
                            while (s < textEnd)
                            {
                                if (IsPathSeparator(*s)) [[unlikely]]
                                {
                                    return false;
                                }
                                s++;
                            }
                        }

                        return true; // Match succeeds instantly
                    }

                    // Save the current state to resume from if a future literal mismatch occurs
                    lastStarPattern = p;
                    starRemainingMinLength = remainingMinLength;

                    // Engage the linear fast-forward cursor instead of locking into character increments
                    if (!advanceToNextTarget())
                    {
                        return false;
                    }

                    // Mark the start string position for future backtracking
                    lastStarString = s;
                    continue; // Fast-forward success, resume next iteration
                }

                case kQuestion:
                {
                    // '?' matches exactly one character, unless it hits a protected boundary
                    if constexpr (LimitWildScope)
                    {
                        if (IsPathSeparator(*s)) [[unlikely]]
                        {
                            break; // Triggers fallback handling below
                        }
                    }

                    p++;
                    s++;
                    remainingMinLength--;
                    continue; // Match success, resume next iteration
                }

                case kEscape:
                {
                    // Handle literal escaping logic. Look ahead for an escaped token
                    if (p + 1 < patternEnd) [[likely]]
                    {
                        if (p[1] == kStar || p[1] == kQuestion || p[1] == kEscape) [[likely]]
                        {
                            if constexpr (IsCaseInsensitive)
                            {
                                if (p[1] == m_policy(*s))
                                {
                                    p += 2;
                                    s++;
                                    remainingMinLength--;
                                    continue; // Match success, resume next iteration
                                }
                            }
                            else
                            {
                                if (p[1] == *s)
                                {
                                    p += 2;
                                    s++;
                                    remainingMinLength--;
                                    continue; // Match success, resume next iteration
                                }
                            }

                            break; // Escaped match failure triggers fallback handling below
                        }
                    }

                    // If unescaped or dangling pipe, treat as literal
                    [[fallthrough]];
                }

                default:
                {
                    // Standard literal character match evaluation
                    if constexpr (IsCaseInsensitive)
                    {
                        if (*p == m_policy(*s))
                        {
                            p++;
                            s++;
                            remainingMinLength--;
                            continue; // Match success, resume next iteration
                        }
                    }
                    else
                    {
                        if (*p == *s)
                        {
                            p++;
                            s++;
                            remainingMinLength--;
                            continue; // Match success, resume next iteration
                        }
                    }

                    break; // Literal match failure triggers fallback handling below
                }
                }
            }
            
            // ------------------------------------------------------------------------
            // Mismatch Handling & Backtracking 
            // ------------------------------------------------------------------------
            // Reached if:
            // 1. The bounds check failed.
            // 2. The pattern ended but the text string did not.
            // 3. A literal, escape, or question-mark check broke out of the switch.

            if (lastStarPattern != nullptr)
            {
                // Ensure backtracking does not cross a hard directory boundary
                if constexpr (LimitWildScope)
                {
                    if (IsPathSeparator(*lastStarString)) [[unlikely]]
                    {
                        return false;
                    }
                }

                p = lastStarPattern;
                s = lastStarString + 1; // Increment to resume backtracking past previous failure position
                remainingMinLength = starRemainingMinLength;

                // Re-engage the linear fast-forward cursor for subsequent tracking jumps
                if (!advanceToNextTarget())
                {
                    return false; // Fast-forward failed to find the required target
                }

                lastStarString = s;
                continue;
            }

            // Mismatch occurred and no prior '*' exists to branch from
            return false;
        }

        // After the string is exhausted, consume any trailing '*' in the pattern
        while (p < patternEnd && *p == kStar)
        {
            p++;
        }

        // Match succeeds only if both the string and the pattern are fully consumed
        return (p == patternEnd);
    }
} // namespace StringPatternMatchDetail

//-----------------------------------------------------------------------------------------------------------------------------------------------------------
//                                                      Main Class Interface
//-----------------------------------------------------------------------------------------------------------------------------------------------------------
template <SupportedChar CharT,
          typename TContext,
          CasePolicy<CharT> TPolicy = CaseSensitivePolicy<CharT>>
class StringPatternMatch : public StringPatternMatchDetail::StringPatternMatchImpl<CharT, TContext, TPolicy>
{
private:
    using Base = StringPatternMatchDetail::StringPatternMatchImpl<CharT, TContext, TPolicy>;

public:
    using StringViewType = typename Base::StringViewType;
    using StringType     = typename Base::StringType;

    // Inherit the zero-cost primitive pointer+length APIs directly from the base class
    using Base::AddPattern;
    using Base::Search;
    using Base::Clear;

    // -------------------------------------------------------------------------------------------
    // Constructor & Destructor
    // -------------------------------------------------------------------------------------------
    
    // -------------------------------------------------------------------------------------------
    // Initializes the pattern matcher with explicit hardware optimization bounds and custom policies.
    //
    // Parameters:
    //   mode   - The SIMD hardware acceleration preference (Auto by default).
    //   policy - The casing and normalization policy payload (Empty by default).
    // -------------------------------------------------------------------------------------------
    explicit StringPatternMatch(OptimizationMode mode   = OptimizationMode::Auto,
                                TPolicy          policy = TPolicy{}) noexcept : Base(mode, std::move(policy))
    {
    }

    ~StringPatternMatch() noexcept = default;

    // -------------------------------------------------------------------------------------------
    // Pattern Registration Wrapper
    // -------------------------------------------------------------------------------------------
    
    // -------------------------------------------------------------------------------------------
    // Registers a pattern via string_view wrapper, forwarding to the core engine.
    //
    // Parameters:
    //   pattern        - The string view representing the wildcard pattern.
    //   wildScope      - The boundary scope to apply for wildcard operators.
    //   patternContext - The user-defined context payload mapped to this specific pattern.
    //
    // Return:
    //   True if successfully registered, or throws an exception based on active policy constraints.
    // -------------------------------------------------------------------------------------------
    inline bool AddPattern(StringViewType pattern,
                           WildcardScope  wildScope,
                           TContext&&     patternContext)
    {
        return Base::AddPattern(pattern.data(),
                                static_cast<uint32_t>(pattern.size()),
                                wildScope,
                                std::move(patternContext));
    }

    // -------------------------------------------------------------------------------------------
    // Single-Match Query Wrapper (First Match)
    // -------------------------------------------------------------------------------------------
    
    // -------------------------------------------------------------------------------------------
    // Searches for the first matching pattern context against the provided target text.
    //
    // Parameters:
    //   text           - The string view representing the target text.
    //   patternContext - Out-pointer receiving the matched payload context on success.
    //
    // Return:
    //   True if a match was found and patternContext populated, false otherwise.
    // -------------------------------------------------------------------------------------------
    inline bool Search(StringViewType text,
                       TContext*&     patternContext) const noexcept
    {
        return Base::Search(text.data(),
                            static_cast<uint32_t>(text.size()),
                            patternContext);
    }

    // -------------------------------------------------------------------------------------------
    // Multi-Match Query Wrapper (All Matches)
    // -------------------------------------------------------------------------------------------
    
    // -------------------------------------------------------------------------------------------
    // Searches and collects all matching pattern contexts against the provided target text.
    //
    // Parameters:
    //   text            - The string view representing the target text.
    //   patternContexts - Vector receiving all context payloads that successfully matched.
    //
    // Return:
    //   True if at least one match was found and appended, false otherwise.
    // -------------------------------------------------------------------------------------------
    inline bool Search(StringViewType          text,
                       std::vector<TContext*>& patternContexts) const
    {
        return Base::Search(text.data(),
                            static_cast<uint32_t>(text.size()),
                            patternContexts);
    }
};

//-------------------------------------------------------------------------------------------
// Convenience Semantic Typedefs
//-------------------------------------------------------------------------------------------

// Narrow strings: Default is strictly case-sensitive to ensure 100% UTF-8 safety
template <typename TContext>
using StringPatternMatchA = StringPatternMatch<char, TContext, CaseSensitivePolicy<char>>;

// Narrow strings: Explicitly opt in to ASCII-only folding (leaves multi-byte UTF-8 as exact bytes)
template <typename TContext>
using AsciiPatternMatchA = StringPatternMatch<char, TContext, AsciiCaseFoldPolicy<char>>;

// Wide strings: Default is case-sensitive
template <typename TContext>
using StringPatternMatchW = StringPatternMatch<wchar_t, TContext, CaseSensitivePolicy<wchar_t>>;

// Wide strings: Safe Unicode wide-character case-folding
template <typename TContext>
using CaseInsensitivePatternMatchW = StringPatternMatch<wchar_t, TContext, UnicodeCaseFoldPolicy<wchar_t>>;
