/*++
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
Module Name:
    halfgemm_kernel_avx2.cpp
Abstract:
    This module implements the half precision (fp16) matrix/matrix multiply
    operation (QGEMM) with avx2.
--
struct MLAS_HALF_GEMM_KERNEL_AVX2 {
    const float *A;
    const float *B;
    float *C;
    size_t CountK;
    size_t CountM;
    size_t CountN;
    size_t lda;
    size_t ldc;
    float alpha;
    bool ZeroMode;
    static constexpr bool PackNeeded = false;
    static constexpr size_t KernelMaxM = 128;  // max # rows the vectorized kernel can process
    static constexpr size_t PackedK = 1;
    static constexpr MLAS_HALF_GEMM_STRIDES Strides{8, 16, 32};
};*/

#include <exception>

#include "halfgemm.h"
#include "mlas_float16.h"
#include "mlasi.h"

struct MLAS_HALF_GEMM_KERNEL_AVX2 {
    static constexpr bool PackNeeded = false;
    static constexpr size_t KernelMaxM = 128;  // max # rows the vectorized kernel can process
    static constexpr size_t PackedK = 1;
    static constexpr size_t BufOverRead = 0;

    static constexpr MLAS_HALF_GEMM_STRIDES Strides{8, 16, 32};
};

template <>
MLAS_FORCEINLINE void
MlasHalfGemmKernel<MLAS_HALF_GEMM_KERNEL_AVX2>(
    size_t CountM,
    size_t CountN,
    size_t CountK,
    _mlas_fp16_* C,
    size_t ldc,
    const _mlas_fp16_* Bias,
    const _mlas_fp16_* A,
    size_t lda,
    const _mlas_fp16_* B,
    size_t ldb,
    const bool ZeroMode  // overwrite the output buffer
)
{
    printf("Doing simple dummy FP16 GEMM AVX2 Style yooooo\n");
    for (size_t m = 0; m < CountM; m++) {
        for (size_t n = 0; n < CountN; n++) {
            const auto* a = A + (m * lda);
            const auto* b = B + n;
            auto* c = C + (m * ldc) + n;

            float sum = Bias == nullptr ? 0.0f : MLAS_Half2Float(Bias[n]);
            if (!ZeroMode) {
                sum += MLAS_Half2Float(*c);
            }

            for (size_t k = 0; k < CountK; k++) {
                auto down = MLAS_Float2Half(MLAS_Half2Float(*a) * MLAS_Half2Float(*b) + sum);
                sum = MLAS_Half2Float(down);
                b += ldb;
                a += 1;
            }

            *c = MLAS_Float2Half(sum);
        }
    }
}

template <>
MLAS_FORCEINLINE void
MlasHalfGemmConvertPackB<MLAS_HALF_GEMM_KERNEL_AVX2>(
    _mlas_fp16_* D,
    const float* B,
    size_t ldb,
    size_t CountN,
    size_t CountK
)
{
    for (size_t k = 0; k < CountK; k++) {
        for (size_t n = 0; n < CountN; n++) {
            *D++ = MLAS_Float2Half(*(B + k * ldb + n));
        }
    }
}

template <>
MLAS_FORCEINLINE void
MlasHalfGemmCopyPackB<MLAS_HALF_GEMM_KERNEL_AVX2>(
    _mlas_fp16_* D,
    const _mlas_fp16_* B,
    size_t ldb,
    size_t CountN,
    size_t CountK
)
{
    MLAS_UNREFERENCED_PARAMETER(D);
    MLAS_UNREFERENCED_PARAMETER(B);
    MLAS_UNREFERENCED_PARAMETER(ldb);
    MLAS_UNREFERENCED_PARAMETER(CountN);
    MLAS_UNREFERENCED_PARAMETER(CountK);
    // No packing needed by default
}

template <>
void
MlasHalfGemmConvertPackA<MLAS_HALF_GEMM_KERNEL_AVX2>(
    _mlas_fp16_* D,
    const float* A,
    size_t lda,
    size_t CountM,
    size_t CountK
)
{
    for (size_t m = 0; m < CountM; m++) {
        for (size_t k = 0; k < CountK; k++) {
            *D++ = MLAS_Float2Half(*(A + m * lda + k));
        }
    }
}

template <>
MLAS_FORCEINLINE const _mlas_fp16_*
MlasHalfGemmPackedBOffset<MLAS_HALF_GEMM_KERNEL_AVX2>(
    const _mlas_fp16_* PackedB,
    size_t DimN,
    size_t DimK,
    size_t StartN,
    size_t StartK
)
{
    // By default the packed buffer is just a row major
    // K row by N column buffer
    MLAS_UNREFERENCED_PARAMETER(DimK);
    return PackedB + StartK * DimN + StartN;
}

template <>
MLAS_FORCEINLINE
    size_t
    MlasHalfGemmPackedBLeadingDim<MLAS_HALF_GEMM_KERNEL_AVX2>(
        size_t DimN,
        size_t DimK
    )
{
    // By default the packed buffer is just a row major
    // K row by N column buffer
    MLAS_UNREFERENCED_PARAMETER(DimK);
    return DimN;
}

template <>
MLAS_FORCEINLINE void
MlasHalfGemmNoPackOperation<MLAS_HALF_GEMM_KERNEL_AVX2>(
    const size_t N,
    const size_t K,
    const MLAS_HALF_GEMM_DATA_PARAMS* Data,
    const size_t RangeStartM,
    const size_t RangeCountM,
    const size_t RangeStartN,
    const size_t RangeCountN
)
{
    //
    // Optimize for the special case where no packing is needed.
    // Simpler tiling as we are not restricted by packing panel size
    //

    const size_t lda = Data->lda;
    size_t ldb = Data->ldb;  // 0 if prepacked
    const size_t ldc = Data->ldc;

    const auto* pa = reinterpret_cast<const _mlas_fp16_*>(Data->A) + RangeStartM * lda;
    const _mlas_fp16_* pb;
    if (ldb == 0) {
        pb = MlasHalfGemmPackedBOffset<MLAS_HALF_GEMM_KERNEL_AVX2>(
            reinterpret_cast<const _mlas_fp16_*>(Data->B),
            N,
            K,
            RangeStartN,
            0
        );
        ldb = MlasHalfGemmPackedBLeadingDim<MLAS_HALF_GEMM_KERNEL_AVX2>(N, K);
    } else {
        pb = reinterpret_cast<const _mlas_fp16_*>(Data->B) + RangeStartN;
    }

    const _mlas_fp16_* Bias = (nullptr == Data->Bias)
                                  ? nullptr
                                  : reinterpret_cast<const _mlas_fp16_*>(Data->Bias) + RangeStartN;
    _mlas_fp16_* c = reinterpret_cast<_mlas_fp16_*>(Data->C) + RangeStartM * ldc + RangeStartN;

    size_t RowsRemaining = RangeCountM;
    while (RowsRemaining > 0) {
        MlasHalfGemmKernel<MLAS_HALF_GEMM_KERNEL_AVX2>(
            RowsRemaining,
            RangeCountN,
            K,
            c,
            ldc,
            Bias,
            pa,
            lda,
            pb,
            ldb,
            true
        );

        size_t RowsHandled = std::min(RowsRemaining, MLAS_HALF_GEMM_KERNEL_AVX2::KernelMaxM);

        if (Data->OutputProcessor != nullptr) {
            Data->OutputProcessor->Process(
                Data->C,
                RangeStartM + RangeCountM - RowsRemaining,
                RangeStartN,
                RowsHandled,
                RangeCountN,
                Data->ldc
            );
        }

        c += ldc * RowsHandled;
        pa += lda * RowsHandled;
        RowsRemaining -= RowsHandled;
    }
}

template <>
void
MlasHalfGemmOperation<MLAS_HALF_GEMM_KERNEL_AVX2>(
    const size_t N,
    const size_t K,
    const MLAS_HALF_GEMM_DATA_PARAMS* Data,
    const size_t RangeStartM,
    const size_t RangeCountM,
    const size_t RangeStartN,
    const size_t RangeCountN
)
{
    const size_t lda = Data->lda;
    const size_t ldb = Data->ldb;
    const size_t ldc = Data->ldc;

    if (!Data->AIsfp32 && (ldb == 0 || (!MLAS_HALF_GEMM_KERNEL_AVX2::PackNeeded && !Data->BIsfp32))) {
        // !Data->AIsfp32 => A is fp16, no packing on the left hand side
        // ldb == 0 => B is already packed, no packing on the right hand side
        // !KernelType::PackNeeded && !Data->BIsfp32  => B is fp16 and the kernel
        //      does not require packing
        //
        // So no packing needed on either A or B, use a simpler driver instead

        MlasHalfGemmNoPackOperation<MLAS_HALF_GEMM_KERNEL_AVX2>(
            N,
            K,
            Data,
            RangeStartM,
            RangeCountM,
            RangeStartN,
            RangeCountN
        );
        return;
    }

    const auto* Bias = reinterpret_cast<const _mlas_fp16_*>(Data->Bias);
    _mlas_fp16_* C = reinterpret_cast<_mlas_fp16_*>(Data->C) + RangeStartM * ldc + RangeStartN;

    //
    // Three dimensional tiling due to limited packing panel size
    //
    constexpr MLAS_HALF_GEMM_STRIDES Strides = MLAS_HALF_GEMM_KERNEL_AVX2::Strides;
    constexpr size_t packASize = UpAlignSize(Strides.M * Strides.K * FP16_SIZE);
    constexpr size_t packBSize = UpAlignSize(Strides.N * Strides.K * FP16_SIZE);
    MlasThreadedBufAlloc(packASize + packBSize);

    uint8_t* p = ThreadedBufHolder.get();
    auto* PanelA = reinterpret_cast<_mlas_fp16_*>(p);
    p += packASize;
    auto* PanelB = reinterpret_cast<_mlas_fp16_*>(p);

    //
    // Step through each slice of matrix B along the K dimension.
    //

    size_t CountK;
    for (size_t k = 0; k < K; k += CountK) {
        CountK = std::min(K - k, Strides.K);
        const size_t PackedCountK = (CountK + MLAS_HALF_GEMM_KERNEL_AVX2::PackedK - 1) / MLAS_HALF_GEMM_KERNEL_AVX2::PackedK;

        //
        // Step through each slice of matrix B along the N dimension.
        //

        size_t CountN;
        for (size_t n = 0; n < RangeCountN; n += CountN) {
            CountN = std::min(RangeCountN - n, Strides.N);

            //
            // Copy a panel of matrix B to a local packed buffer.
            //
            size_t ld_pb;
            const _mlas_fp16_* pb;
            if (ldb == 0) {
                // Already packed
                pb = MlasHalfGemmPackedBOffset<MLAS_HALF_GEMM_KERNEL_AVX2>(
                    reinterpret_cast<const _mlas_fp16_*>(Data->B),
                    N,
                    K,
                    RangeStartN + n,
                    k
                );
                ld_pb = MlasHalfGemmPackedBLeadingDim<MLAS_HALF_GEMM_KERNEL_AVX2>(N, K);
            } else if (Data->BIsfp32) {
                // fp32, need conversion and packing
                MlasHalfGemmConvertPackB<MLAS_HALF_GEMM_KERNEL_AVX2>(
                    PanelB,
                    reinterpret_cast<const float*>(Data->B) + ldb * k + RangeStartN + n,
                    ldb,
                    CountN,
                    CountK
                );
                pb = PanelB;
                ld_pb = MlasHalfGemmPackedBLeadingDim<MLAS_HALF_GEMM_KERNEL_AVX2>(CountN, CountK);
            } else if (MLAS_HALF_GEMM_KERNEL_AVX2::PackNeeded) {
                // fp16, need packing
                MlasHalfGemmCopyPackB<MLAS_HALF_GEMM_KERNEL_AVX2>(
                    PanelB,
                    reinterpret_cast<const _mlas_fp16_*>(Data->B) + ldb * k + RangeStartN + n,
                    ldb,
                    CountN,
                    CountK
                );
                pb = PanelB;
                ld_pb = MlasHalfGemmPackedBLeadingDim<MLAS_HALF_GEMM_KERNEL_AVX2>(CountN, CountK);
            } else {
                // fp16, and no packing needed
                pb = reinterpret_cast<const _mlas_fp16_*>(Data->B) + ldb * k + RangeStartN + n;
                ld_pb = ldb;
            }

            //
            // Step through each slice of matrix A along the M dimension.
            //

            auto* c = C + n;
            const auto* pbias = (nullptr == Bias) ? nullptr : Bias + RangeStartN + n;
            size_t CountM;
            for (size_t m = 0; m < RangeCountM; m += CountM) {
                CountM = std::min(RangeCountM - m, Strides.M);

                //
                // Copy a panel of matrix A to a local packed buffer.
                //
                const _mlas_fp16_* pa;
                size_t ld_pa;
                if (Data->AIsfp32) {
                    MlasHalfGemmConvertPackA<MLAS_HALF_GEMM_KERNEL_AVX2>(
                        PanelA,
                        reinterpret_cast<const float*>(Data->A) + (RangeStartM + m) * lda + k,
                        lda,
                        CountM,
                        CountK
                    );
                    pa = PanelA;
                    ld_pa = MLAS_HALF_GEMM_KERNEL_AVX2::PackedK * PackedCountK;
                } else {
                    pa = reinterpret_cast<const _mlas_fp16_*>(Data->A) + (RangeStartM + m) * lda + k;
                    ld_pa = lda;
                }

                size_t RowsRemaining = CountM;
                bool ZeroMode = (k == 0);
                bool PostProcess = (k + CountK == K);

                while (RowsRemaining > 0) {
                    MlasHalfGemmKernel<MLAS_HALF_GEMM_KERNEL_AVX2>(
                        RowsRemaining,
                        CountN,
                        CountK,
                        c,
                        ldc,
                        ZeroMode ? pbias : nullptr,
                        pa,
                        ld_pa,
                        pb,
                        ld_pb,
                        ZeroMode
                    );

                    size_t RowsHandled = std::min(RowsRemaining, MLAS_HALF_GEMM_KERNEL_AVX2::KernelMaxM);

                    if (PostProcess && Data->OutputProcessor != nullptr) {
                        Data->OutputProcessor->Process(
                            Data->C,
                            RangeStartM + m + CountM - RowsRemaining,
                            RangeStartN + n,
                            RowsHandled,
                            CountN,
                            Data->ldc
                        );
                    }

                    c += ldc * RowsHandled;
                    pa += ld_pa * RowsHandled;
                    RowsRemaining -= RowsHandled;
                }
            }
        }
    }
}

const MLAS_HALFGEMM_DISPATCH MlasHalfGemmDispatchAVX2 = {
    MlasHalfGemmOperation<MLAS_HALF_GEMM_KERNEL_AVX2>,    /**< HalfGemm driver */
    nullptr,                                              /**< Pack function for B */
    MlasHalfGemmConvertPackB<MLAS_HALF_GEMM_KERNEL_AVX2>, /**< Convert and pack function for B */
    MLAS_HALF_GEMM_KERNEL_AVX2::PackedK,
    MLAS_HALF_GEMM_KERNEL_AVX2::KernelMaxM,
    MLAS_HALF_GEMM_KERNEL_AVX2::BufOverRead
};