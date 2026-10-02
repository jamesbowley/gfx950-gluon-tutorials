// Standalone hipBLASLt BF16 TN GEMM benchmark (no torch).
//
//   hbl_bench <shapes.txt> <heuristic|best256> [reps=3] [iters=100]
//
// shapes.txt: one GEMM per line, "m n k dtypeD" in hipBLASLt column-major terms, dtypeD bf16|f32.
// A torch F.linear(x[M,K], w[N,K]) is m=N, n=M, k=K, opA=T, opB=N, lda=ldb=K, ldc=ldd=N.
// A/B BF16, compute fp32, C=D type, alpha=1, beta=0, no bias.
//
// heuristic: hipblasLtMatmulAlgoGetHeuristic, 1 requested solution, 128 MiB workspace
//            (hipblaslt-bench's defaults).
// best256:   every solution from hipblaslt_ext::getAllAlgos whose kernel name contains
//            MT256x256x64 and that passes matmulIsAlgoSupported; screened with a short run,
//            the fastest re-timed with the full protocol.
//
// Timing: 20 warm-up calls, then `reps` repetitions of `iters` back-to-back calls between two
// hipEvents; the median repetition is reported. CSV rows go to stdout, "#" lines are metadata.

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#define CHECK_HIP(x)                                                                       \
    do {                                                                                   \
        hipError_t e_ = (x);                                                               \
        if (e_ != hipSuccess) {                                                            \
            fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); \
            exit(1);                                                                       \
        }                                                                                  \
    } while (0)
#define CHECK_BLAS(x)                                                                      \
    do {                                                                                   \
        hipblasStatus_t s_ = (x);                                                          \
        if (s_ != HIPBLAS_STATUS_SUCCESS) {                                                \
            fprintf(stderr, "hipBLASLt error %d at %s:%d\n", (int)s_, __FILE__, __LINE__); \
            exit(1);                                                                       \
        }                                                                                  \
    } while (0)

// 128 MiB by default (hipblaslt-bench); HBL_WORKSPACE_KB overrides it.
static const size_t kWorkspace = getenv("HBL_WORKSPACE_KB")
                                     ? (size_t)strtoull(getenv("HBL_WORKSPACE_KB"), nullptr, 10) << 10
                                     : 128ull << 20;

__global__ void fill_bf16(__hip_bfloat16* p, size_t n, uint32_t seed)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    for (; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t h = (uint32_t)i * 2654435761u ^ seed;
        h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
        p[i] = __float2bfloat16((float)(h & 0xffff) / 32768.0f - 1.0f);
    }
}

struct Gemm {
    hipblasLtHandle_t       handle;
    hipblasLtMatmulDesc_t   desc;
    hipblasLtMatrixLayout_t a, b, c, d;
    void *dA, *dB, *dD, *ws;
    float alpha = 1.f, beta = 0.f;
    // Workspace size passed to hipblasLtMatmul: the allocated buffer (as frameworks do), or with
    // HBL_MATMUL_WS=heuristic the heuristic's reported workspaceSize (as hipblaslt-bench does).
    size_t wsize = kWorkspace;
    hipStream_t stream;
};

// Returns ms per call, or -(hipblasStatus_t) if hipblasLtMatmul fails.
static float time_algo(Gemm& g, hipblasLtMatmulAlgo_t& algo, int warm, int reps, int iters)
{
    hipblasStatus_t st = hipblasLtMatmul(g.handle, g.desc, &g.alpha, g.dA, g.a, g.dB, g.b, &g.beta, g.dD,
                                         g.c, g.dD, g.d, &algo, g.ws, g.wsize, g.stream);
    if (st != HIPBLAS_STATUS_SUCCESS) return -(float)st;
    for (int i = 1; i < warm; ++i)
        CHECK_BLAS(hipblasLtMatmul(g.handle, g.desc, &g.alpha, g.dA, g.a, g.dB, g.b, &g.beta, g.dD, g.c,
                                   g.dD, g.d, &algo, g.ws, g.wsize, g.stream));
    hipEvent_t s, e;
    CHECK_HIP(hipEventCreate(&s));
    CHECK_HIP(hipEventCreate(&e));
    std::vector<float> ms;
    for (int r = 0; r < reps; ++r) {
        CHECK_HIP(hipEventRecord(s, g.stream));
        for (int i = 0; i < iters; ++i)
            CHECK_BLAS(hipblasLtMatmul(g.handle, g.desc, &g.alpha, g.dA, g.a, g.dB, g.b, &g.beta, g.dD,
                                       g.c, g.dD, g.d, &algo, g.ws, g.wsize, g.stream));
        CHECK_HIP(hipEventRecord(e, g.stream));
        CHECK_HIP(hipEventSynchronize(e));
        float t;
        CHECK_HIP(hipEventElapsedTime(&t, s, e));
        ms.push_back(t / iters);
    }
    CHECK_HIP(hipEventDestroy(s));
    CHECK_HIP(hipEventDestroy(e));
    std::sort(ms.begin(), ms.end());
    return ms[ms.size() / 2];
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s shapes.txt heuristic|best256 [reps] [iters]\n", argv[0]);
        return 2;
    }
    std::string mode = argv[2];
    int reps = argc > 3 ? atoi(argv[3]) : 3;
    int iters = argc > 4 ? atoi(argv[4]) : 100;

    int ndev = 0;
    CHECK_HIP(hipGetDeviceCount(&ndev));
    if (ndev != 1) {
        fprintf(stderr, "expected exactly one visible device, got %d\n", ndev);
        return 1;
    }
    hipDeviceProp_t prop;
    CHECK_HIP(hipGetDeviceProperties(&prop, 0));

    Gemm g;
    CHECK_BLAS(hipblasLtCreate(&g.handle));
    int ver = 0;
    char rev[256] = {0};
    CHECK_BLAS(hipblasLtGetVersion(g.handle, &ver));
    CHECK_BLAS(hipblasLtGetGitRevision(g.handle, rev));
    const char* pick = getenv("ANALYTICAL_GEMM_PICK");
    printf("# hipblaslt=%d rev=%s gpu=%s pci_bus=%d pci_domain=%d mode=%s pick=%s reps=%d iters=%d "
           "workspace_kb=%zu\n",
           ver, rev, prop.gcnArchName, prop.pciBusID, prop.pciDomainID, mode.c_str(),
           pick ? pick : "", reps, iters, kWorkspace >> 10);
    printf("m,n,k,dtype_d,mode,kernel,us,tflops,candidates\n");
    fflush(stdout);

    CHECK_HIP(hipStreamCreate(&g.stream));
    CHECK_HIP(hipMalloc(&g.ws, kWorkspace));

    std::ifstream in(argv[1]);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        long m, n, k;
        std::string dt;
        ss >> m >> n >> k >> dt;
        hipDataType td = dt == "f32" ? HIP_R_32F : HIP_R_16BF;
        size_t esz = dt == "f32" ? 4 : 2;

        CHECK_HIP(hipMalloc(&g.dA, (size_t)k * m * 2));
        CHECK_HIP(hipMalloc(&g.dB, (size_t)k * n * 2));
        CHECK_HIP(hipMalloc(&g.dD, (size_t)m * n * esz));
        fill_bf16<<<4096, 256, 0, g.stream>>>((__hip_bfloat16*)g.dA, (size_t)k * m, 1u);
        fill_bf16<<<4096, 256, 0, g.stream>>>((__hip_bfloat16*)g.dB, (size_t)k * n, 2u);
        CHECK_HIP(hipStreamSynchronize(g.stream));

        CHECK_BLAS(hipblasLtMatrixLayoutCreate(&g.a, HIP_R_16BF, k, m, k));
        CHECK_BLAS(hipblasLtMatrixLayoutCreate(&g.b, HIP_R_16BF, k, n, k));
        CHECK_BLAS(hipblasLtMatrixLayoutCreate(&g.c, td, m, n, m));
        CHECK_BLAS(hipblasLtMatrixLayoutCreate(&g.d, td, m, n, m));
        CHECK_BLAS(hipblasLtMatmulDescCreate(&g.desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
        hipblasOperation_t opA = HIPBLAS_OP_T, opB = HIPBLAS_OP_N;
        CHECK_BLAS(hipblasLtMatmulDescSetAttribute(g.desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
        CHECK_BLAS(hipblasLtMatmulDescSetAttribute(g.desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));

        std::string kernel = "NONE";
        float ms = -1.f;
        int candidates = 0;
        if (mode == "heuristic") {
            hipblasLtMatmulPreference_t pref;
            CHECK_BLAS(hipblasLtMatmulPreferenceCreate(&pref));
            size_t wsz = kWorkspace;
            CHECK_BLAS(hipblasLtMatmulPreferenceSetAttribute(
                pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz, sizeof(wsz)));
            hipblasLtMatmulHeuristicResult_t res[1];
            int returned = 0;
            hipblasStatus_t st = hipblasLtMatmulAlgoGetHeuristic(g.handle, g.desc, g.a, g.b, g.c, g.d,
                                                                 pref, 1, res, &returned);
            if (st == HIPBLAS_STATUS_SUCCESS && returned > 0) {
                candidates = returned;
                kernel = hipblaslt_ext::getKernelNameFromAlgo(g.handle, res[0].algo);
                if (getenv("HBL_PRINT_WS"))
                    fprintf(stderr, "%ld %ld %ld heuristic workspaceSize=%zu limit=%zu\n", m, n, k,
                            res[0].workspaceSize, kWorkspace);
                const char* mws = getenv("HBL_MATMUL_WS");
                g.wsize = mws && std::string(mws) == "heuristic" ? res[0].workspaceSize : kWorkspace;
                ms = time_algo(g, res[0].algo, 20, reps, iters);
                g.wsize = kWorkspace;
            }
            hipblasLtMatmulPreferenceDestroy(pref);
        } else if (mode == "best256") {
            std::vector<hipblasLtMatmulHeuristicResult_t> all;
            hipblaslt_ext::getAllAlgos(g.handle, hipblaslt_ext::GemmType::HIPBLASLT_GEMM, opA, opB,
                                       HIP_R_16BF, HIP_R_16BF, td, td, HIPBLAS_COMPUTE_32F, all);
            float best = 1e30f;
            int bi = -1;
            for (size_t i = 0; i < all.size(); ++i) {
                std::string kn = hipblaslt_ext::getKernelNameFromAlgo(g.handle, all[i].algo);
                if (kn.find("MT256x256x64") == std::string::npos) continue;
                size_t need = 0;
                if (hipblaslt_ext::matmulIsAlgoSupported(g.handle, g.desc, &g.alpha, g.a, g.b, &g.beta,
                                                         g.c, g.d, all[i].algo, need)
                        != HIPBLAS_STATUS_SUCCESS
                    || need > kWorkspace)
                    continue;
                ++candidates;
                float t = time_algo(g, all[i].algo, 3, 1, 10);
                if (t > 0 && t < best) { best = t; bi = (int)i; }
            }
            if (bi >= 0) {
                kernel = hipblaslt_ext::getKernelNameFromAlgo(g.handle, all[bi].algo);
                ms = time_algo(g, all[bi].algo, 20, reps, iters);
            }
        } else {
            fprintf(stderr, "unknown mode %s\n", mode.c_str());
            return 2;
        }
        double tflops = ms > 0 ? 2.0 * m * n * k / (ms * 1e-3) * 1e-12 : 0.0;
        if (ms < 0 && kernel != "NONE") kernel += " [hipblasLtMatmul status " + std::to_string((int)-ms) + "]";
        printf("%ld,%ld,%ld,%s,%s,%s,%.3f,%.1f,%d\n", m, n, k, dt.c_str(), mode.c_str(), kernel.c_str(),
               ms > 0 ? ms * 1e3 : 0.0, tflops, candidates);
        fflush(stdout);

        hipblasLtMatmulDescDestroy(g.desc);
        hipblasLtMatrixLayoutDestroy(g.a);
        hipblasLtMatrixLayoutDestroy(g.b);
        hipblasLtMatrixLayoutDestroy(g.c);
        hipblasLtMatrixLayoutDestroy(g.d);
        CHECK_HIP(hipFree(g.dA));
        CHECK_HIP(hipFree(g.dB));
        CHECK_HIP(hipFree(g.dD));
    }
    CHECK_HIP(hipFree(g.ws));
    hipblasLtDestroy(g.handle);
    return 0;
}
