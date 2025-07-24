# ggml-neon-opt
The goal of this repository is to optimize the ggml library for ARM Neon processors and achieve as high of a performance as I can on my Raspberry Pi 5. I hope to learn more about ML, and enjoy doing more with my Raspberry Pi.
My Raspberry Pi 5 is a 4 core ARM Cortex-A76 with 16GB of RAM.

## Summary of Results
- initial benchmarks
- optimized benchmarks

## Getting Started
### Prerequisites
Make sure to have arm neon and `sudo apt install build-essential linux-tools-$(uname -r) linux-perf libopenblas-dev libcurl4-openssl-dev libomp-dev wget`

### Cloning the Repoistory
```bash
git clone --recursive git@github.com:george-tyler-315-incline/ggml-neon-opt.git
cd ggml-neon-opt
# If you forgot --recursive:
# git submodule update --init --recursive
```
This project uses llama.cpp as a submodule. For detailed build instructions, see the [llama.cpp build documentation](external/llama.cpp/docs/build.md).

### Building and Verifying Functionality
```bash
# Build llama.cpp release with debug info
# TODO explain why we want BLAS and OPENMP off
cmake -S external/llama.cpp -B external/llama.cpp/build \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DGGML_NATIVE=ON \
  -DGGML_BLAS=OFF \
  -DGGML_BLAS_VENDOR=OpenBLAS \
  -DGGML_OPENMP=OFF \
  -DLLAMA_CURL=OFF
cmake --build external/llama.cpp/build

# Download a test model (TinyLlama)
mkdir -p models
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -O models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# Run the model using llama.cpp to verify setup
./external/llama.cpp/build/bin/llama-cli   -m models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf   -p "What's the coolest thing about the C programming language?"   -n 100   -t 1
```

## My Journey
### Initial GGML Profiling on the Pi
I took an initial benchmarks T₀ of GGML with TinyLlama using 512 tokens for the prompt, 128 tokens for the generation, and one thread for ggml threads. I ran it on an isolated core.
```bash
# T₀: Core 0 with OS activity, gave 15.48 tokens/second on prompt processing and 10.09 tokens/second on generation
brandonneway@raspberrypi:~/dev/transformers/ggml-neon-opt $ taskset -c 0 ./external/llama.cpp/build/bin/llama-bench -m models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf  -p 512 -n 128 -t 1
| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| llama 1B Q4_K - Medium         | 636.18 MiB |     1.10 B | BLAS       |       1 |           pp512 |         15.48 ± 0.00 |
| llama 1B Q4_K - Medium         | 636.18 MiB |     1.10 B | BLAS       |       1 |           tg128 |         10.09 ± 0.07 |

build: 36c15324 (5944)
# T₁: Core 2 isolated
brandonneway@raspberrypi:~/dev/transformers/ggml-neon-opt $ taskset -c 2 ./external/llama.cpp/build/bin/llama-bench -m models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf  -p 512 -n 128 -t 1
| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| llama 1B Q4_K - Medium         | 636.18 MiB |     1.10 B | BLAS       |       1 |           pp512 |         15.49 ± 0.00 |
| llama 1B Q4_K - Medium         | 636.18 MiB |     1.10 B | BLAS       |       1 |           tg128 |         10.09 ± 0.06 |

build: 36c15324 (5944)
```

Next, I used perf record for some inital cpu time profiling.
```bash
brandonneway@raspberrypi:~/dev/transformers/ggml-neon-opt $ sudo perf record -F 1000 -g --call-graph=dwarf    taskset -c 2 ./external/llama.cpp/build/bin/llama-bench -m models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -p 0 -n 128 -t 1
| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| llama 1B Q4_K - Medium         | 636.18 MiB |     1.10 B | BLAS       |       1 |           tg128 |          8.86 ± 0.01 |

build: 36c15324 (5944)
[ perf record: Woken up 2362 times to write data ]
[ perf record: Captured and wrote 590.417 MB perf.data (72453 samples) ]

brandonneway@raspberrypi:~/dev/transformers/ggml-neon-opt $ sudo perf report --stdio --no-children | head -300
# To display the perf.data header info, please use --header/--header-only options.
#
#
# Total Lost Samples: 0
#
# Samples: 72K of event 'cycles:P'
# Event count (approx.): 173521193861
#
# Overhead  Command      Shared Object          Symbol
# ........  ...........  .....................  ................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................
#
    66.48%  llama-bench  libggml-cpu.so         [.] ggml_vec_dot_q4_K_q8_K
            |
            |--35.96%--ggml_vec_dot_q4_K_q8_K
            |          ggml_compute_forward_mul_mat_one_chunk (inlined)
            |          ggml_compute_forward_mul_mat
            |          ggml_compute_forward (inlined)
            |          ggml_graph_compute_thread (inlined)
            |          ggml_graph_compute
            |          ggml_backend_cpu_graph_compute(ggml_backend*, ggml_cgraph*)
            |          ggml_backend_sched_graph_compute_async
            |          ggml_backend_sched_graph_compute_async
            |          llama_context::graph_compute(ggml_cgraph*, bool)
            |          llama_context::process_ubatch(llama_ubatch const&, llm_graph_type, llama_memory_context_i*, ggml_status&)
            |          llama_context::decode(llama_batch const&)
            |
            |--11.15%--vdotq_s32 (inlined)
            |          ggml_vec_dot_q4_K_q8_K
            |          ggml_compute_forward_mul_mat_one_chunk (inlined)
            |          ggml_compute_forward_mul_mat
            |          ggml_compute_forward (inlined)
            |          ggml_graph_compute_thread (inlined)
            |          ggml_graph_compute
            |          ggml_backend_cpu_graph_compute(ggml_backend*, ggml_cgraph*)
            |          ggml_backend_sched_graph_compute_async
            |          ggml_backend_sched_graph_compute_async
            |          llama_context::graph_compute(ggml_cgraph*, bool)
            |          llama_context::process_ubatch(llama_ubatch const&, llm_graph_type, llama_memory_context_i*, ggml_status&)
            |          llama_context::decode(llama_batch const&)
            |
            |--8.29%--vandq_u8 (inlined)
            |          ggml_vec_dot_q4_K_q8_K
            |          ggml_compute_forward_mul_mat_one_chunk (inlined)
            |          ggml_compute_forward_mul_mat
            |          ggml_compute_forward (inlined)
            |          ggml_graph_compute_thread (inlined)
            |          ggml_graph_compute
            |          ggml_backend_cpu_graph_compute(ggml_backend*, ggml_cgraph*)
            |          ggml_backend_sched_graph_compute_async
            |          ggml_backend_sched_graph_compute_async
            |          llama_context::graph_compute(ggml_cgraph*, bool)
            |          llama_context::process_ubatch(llama_ubatch const&, llm_graph_type, llama_memory_context_i*, ggml_status&)
            |          llama_context::decode(llama_batch const&)
            |
            |--4.49%--vaddvq_s32 (inlined)
            |          ggml_vec_dot_q4_K_q8_K
            |          ggml_compute_forward_mul_mat_one_chunk (inlined)
            |          ggml_compute_forward_mul_mat
            |          ggml_compute_forward (inlined)
            |          ggml_graph_compute_thread (inlined)
            |          ggml_graph_compute
            |          ggml_backend_cpu_graph_compute(ggml_backend*, ggml_cgraph*)
            |          ggml_backend_sched_graph_compute_async
            |          ggml_backend_sched_graph_compute_async
            |          llama_context::graph_compute(ggml_cgraph*, bool)
            |          llama_context::process_ubatch(llama_ubatch const&, llm_graph_type, llama_memory_context_i*, ggml_status&)
            |          llama_context::decode(llama_batch const&)
            |
            |--2.39%--vld1q_u8_x2 (inlined)
            |          ggml_vec_dot_q4_K_q8_K
            |          ggml_compute_forward_mul_mat_one_chunk (inlined)
            |          ggml_compute_forward_mul_mat
            |          ggml_compute_forward (inlined)
            |          ggml_graph_compute_thread (inlined)
            |          ggml_graph_compute
            |          ggml_backend_cpu_graph_compute(ggml_backend*, ggml_cgraph*)
            |          ggml_backend_sched_graph_compute_async
            |          ggml_backend_sched_graph_compute_async
            |          llama_context::graph_compute(ggml_cgraph*, bool)
            |          llama_context::process_ubatch(llama_ubatch const&, llm_graph_type, llama_memory_context_i*, ggml_status&)
            |          llama_context::decode(llama_batch const&)
            |
            |--1.52%--vld1q_s8_x2 (inlined)
            |          ggml_vec_dot_q4_K_q8_K
            |          ggml_compute_forward_mul_mat_one_chunk (inlined)
            |          ggml_compute_forward_mul_mat
            |          ggml_compute_forward (inlined)
            |          ggml_graph_compute_thread (inlined)
            |          ggml_graph_compute
            |          ggml_backend_cpu_graph_compute(ggml_backend*, ggml_cgraph*)
            |          ggml_backend_sched_graph_compute_async
            |          ggml_backend_sched_graph_compute_async
            |          llama_context::graph_compute(ggml_cgraph*, bool)
            |          llama_context::process_ubatch(llama_ubatch const&, llm_graph_type, llama_memory_context_i*, ggml_status&)
            |          llama_context::decode(llama_batch const&)
            |
            |--1.27%--neon_compute_fp16_to_fp32 (inlined)
            |          ggml_vec_dot_q4_K_q8_K
            |          ggml_compute_forward_mul_mat_one_chunk (inlined)
            |          ggml_compute_forward_mul_mat
            |          ggml_compute_forward (inlined)
            |          ggml_graph_compute_thread (inlined)
            |          ggml_graph_compute
            |          ggml_backend_cpu_graph_compute(ggml_backend*, ggml_cgraph*)
            |          ggml_backend_sched_graph_compute_async
            |          ggml_backend_sched_graph_compute_async
            |          llama_context::graph_compute(ggml_cgraph*, bool)
            |          llama_context::process_ubatch(llama_ubatch const&, llm_graph_type, llama_memory_context_i*, ggml_status&)
            |          llama_context::decode(llama_batch const&)
            |
             --1.09%--vset_lane_u32 (inlined)
                       ggml_vec_dot_q4_K_q8_K
                       ggml_compute_forward_mul_mat_one_chunk (inlined)
                       ggml_compute_forward_mul_mat
                       ggml_compute_forward (inlined)
                       ggml_graph_compute_thread (inlined)
                       ggml_graph_compute
                       ggml_backend_cpu_graph_compute(ggml_backend*, ggml_cgraph*)
                       ggml_backend_sched_graph_compute_async
                       ggml_backend_sched_graph_compute_async
                       llama_context::graph_compute(ggml_cgraph*, bool)
                       llama_context::process_ubatch(llama_ubatch const&, llm_graph_type, llama_memory_context_i*, ggml_status&)
                       llama_context::decode(llama_batch const&)

    26.15%  llama-bench  libggml-cpu.so         [.] ggml_vec_dot_q6_K_q8_K
```

Next, I made a flame graph to see as well.
<insert flame graph>

So, clearly a lot of CPU time is being spent in `ggml_vec_dot_q4_K_q8_K`, with a bit over half of the time spent in the function itself and the rest of the time spent in the few functions it is calling.
Next, I need to know what the code is doing, and unfortunately there are a lot of preprocessor feature-based branches, so I gotta figure out exactly what is running and what isn't.
We can first see what is default defined:

```bash
brandonneway@raspberrypi:~/dev/transformers/ggml-neon-opt $ echo | gcc -march=native -E -dM - | grep -e 'ARM_FEATURE_' -e 'NEON' -e 'MATMUL'
#define __ARM_FEATURE_ATOMICS 1
#define __ARM_FEATURE_AES 1
#define __ARM_FEATURE_IDIV 1
#define __ARM_FEATURE_DOTPROD 1
#define __ARM_FEATURE_CRYPTO 1
#define __ARM_FEATURE_FP16_SCALAR_ARITHMETIC 1
#define __ARM_FEATURE_CLZ 1
#define __ARM_FEATURE_QRDMX 1
#define __ARM_FEATURE_FMA 1
#define __ARM_FEATURE_SHA2 1
#define __ARM_FEATURE_FP16_VECTOR_ARITHMETIC 1
#define __ARM_FEATURE_UNALIGNED 1
#define __ARM_FEATURE_CRC32 1
#define __ARM_NEON 1
#define __ARM_FEATURE_NUMERIC_MAXMIN 1
brandonneway@raspberrypi:~/dev/transformers/ggml-neon-opt $
```

So it seems we don't have SVE and we don't have MATMUL_I8. We can do a sanity check by seeing how the processor treats the source file. From `external/llama.cpp/build/compile_commands.json`:
```
{
  "directory": "/home/brandonneway/dev/transformers/ggml-neon-opt/external/llama.cpp/build/ggml/src",
  "command": "/usr/bin/cc -DGGML_BACKEND_BUILD -DGGML_BACKEND_SHARED -DGGML_SCHED_MAX_COPIES=4 -DGGML_SHARED -DGGML_USE_CPU_REPACK -DGGML_USE_LLAMAFILE -DGGML_USE_OPENMP -D_GNU_SOURCE -D_XOPEN_SOURCE=600 -Dggml_cpu_EXPORTS -I/home/brandonneway/dev/transformers/ggml-neon-opt/external/llama.cpp/ggml/src/.. -I/home/brandonneway/dev/transformers/ggml-neon-opt/external/llama.cpp/ggml/src/. -I/home/brandonneway/dev/transformers/ggml-neon-opt/external/llama.cpp/ggml/src/ggml-cpu -I/home/brandonneway/dev/transformers/ggml-neon-opt/external/llama.cpp/ggml/src/../include -O2 -g -DNDEBUG -fPIC -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes -Werror=implicit-int -Werror=implicit-function-declaration -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wdouble-promotion -mcpu=cortex-a76+crypto+dotprod+noi8mm+nosve -fopenmp -std=gnu11 -o CMakeFiles/ggml-cpu.dir/ggml-cpu/arch/arm/quants.c.o -c /home/brandonneway/dev/transformers/ggml-neon-opt/external/llama.cpp/ggml/src/ggml-cpu/arch/arm/quants.c",
  "file": "/home/brandonneway/dev/transformers/ggml-neon-opt/external/llama.cpp/ggml/src/ggml-cpu/arch/arm/quants.c"
},
```
It seems -mcpu also confirms we don't have i8mm or sve. I will modify the command to dump the preprocessed file though via gcc -E.
```bash
brandonneway@raspberrypi:~/dev/transformers/ggml-neon-opt/external/llama.cpp/build/ggml/src $ /usr/bin/cc -DGGML_BACKEND_BUILD -DGGML_BACKEND_SHARED -DGGML_SCHED_MAX_COPIES=4 -DGGML_SHARED -DGGML_USE_CPU_REPACK -DGGML_USE_LLAMAFILE -DGGML_USE_OPENMP -D_GNU_SOURCE -D_XOPEN_SOURCE=600 -Dggml_cpu_EXPORTS -I/home/brandonneway/dev/transformers/ggml-neon-opt/external/llama.cpp/ggml/src/.. -I/home/brandonneway/dev/transformers/ggml-neon-opt/external/llama.cpp/ggml/src/. -I/home/brandonneway/dev/transformers/ggml-neon-opt/external/llama.cpp/ggml/src/ggml-cpu -I/home/brandonneway/dev/transformers/ggml-neon-opt/external/llama.cpp/ggml/src/../include -O2 -g -DNDEBUG -fPIC -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes -Werror=implicit-int -Werror=implicit-function-declaration -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wdouble-promotion -mcpu=cortex-a76+crypto+dotprod+noi8mm+nosve -fopenmp -std=gnu11 -o ~/dev/transformers/ggml-neon-opt/preprocessed_arm_quants.c.txt -E /home/brandonneway/dev/transformers/ggml-neon-opt/external/llama.cpp/ggml/src/ggml-cpu/arch/arm/quants.c
```

The preprocessed file reveals this as the source.
```c
void ggml_vec_dot_q4_K_q8_K(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {

# 2127 "/home/brandonneway/dev/transformers/ggml-neon-opt/external/llama.cpp/ggml/src/ggml-cpu/arch/arm/quants.c" 3 4
   ((void) (0))
# 2127 "/home/brandonneway/dev/transformers/ggml-neon-opt/external/llama.cpp/ggml/src/ggml-cpu/arch/arm/quants.c"
                        ;




# 2131 "/home/brandonneway/dev/transformers/ggml-neon-opt/external/llama.cpp/ggml/src/ggml-cpu/arch/arm/quants.c" 3 4
   ((void) (0))
# 2131 "/home/brandonneway/dev/transformers/ggml-neon-opt/external/llama.cpp/ggml/src/ggml-cpu/arch/arm/quants.c"
                   ;

    (void)(nrc);
    (void)(bx);
    (void)(by);
    (void)(bs);

    const block_q4_K * restrict x = vx;
    const block_q8_K * restrict y = vy;

    const int nb = n / 256;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];
# 2371 "/home/brandonneway/dev/transformers/ggml-neon-opt/external/llama.cpp/ggml/src/ggml-cpu/arch/arm/quants.c"
    const uint8x16_t m4b = vdupq_n_u8(0xf);
    const int32x4_t mzero = vdupq_n_s32(0);

    int8x16x2_t q4bytes;
    int8x16x2_t q8bytes;

    float sumf = 0;

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * neon_compute_fp16_to_fp32(x[i].d);
        const float dmin = y[i].d * neon_compute_fp16_to_fp32(x[i].dmin);

        const int16x8_t q8sums = vpaddq_s16(vld1q_s16(y[i].bsums), vld1q_s16(y[i].bsums + 8));

        memcpy(utmp, x[i].scales, 12);

        uint32x2_t mins8 = { 0 };
        mins8 = vset_lane_u32(utmp[1] & kmask1, mins8, 0);
        mins8 = vset_lane_u32(((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4), mins8, 1);

        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[0] &= kmask1;

        const int16x8_t mins = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(mins8)));
        const int32x4_t prod = vaddq_s32(vmull_s16(vget_low_s16 (q8sums), vget_low_s16 (mins)),
                                         vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins)));
        sumf -= dmin * vaddvq_s32(prod);

        const uint8_t * scales = (const uint8_t *)utmp;

        const uint8_t * restrict q4 = x[i].qs;
        const int8_t * restrict q8 = y[i].qs;

        int32_t sumi1 = 0;
        int32_t sumi2 = 0;

        for (int j = 0; j < 256/64; ++j) {
            const uint8x16x2_t q4bits = vld1q_u8_x2(q4); q4 += 32;

            q8bytes = vld1q_s8_x2(q8); q8 += 32;
            q4bytes.val[0] = vreinterpretq_s8_u8(vandq_u8 (q4bits.val[0], m4b));
            q4bytes.val[1] = vreinterpretq_s8_u8(vandq_u8 (q4bits.val[1], m4b));

            const int32x4_t p1 = vdotq_s32(vdotq_s32(mzero, q4bytes.val[0], q8bytes.val[0]), q4bytes.val[1], q8bytes.val[1]);
            sumi1 += vaddvq_s32(p1) * scales[2*j+0];

            q8bytes = vld1q_s8_x2(q8); q8 += 32;
            q4bytes.val[0] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[0], 4));
            q4bytes.val[1] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[1], 4));

            const int32x4_t p2 = vdotq_s32(vdotq_s32(mzero, q4bytes.val[0], q8bytes.val[0]), q4bytes.val[1], q8bytes.val[1]);

            sumi2 += vaddvq_s32(p2) * scales[2*j+1];
        }

        sumf += d * (sumi1 + sumi2);

    }

    *s = sumf;
# 2490 "/home/brandonneway/dev/transformers/ggml-neon-opt/external/llama.cpp/ggml/src/ggml-cpu/arch/arm/quants.c"
}
```
Comparing with the original source file confirms the `__ARM_FEATURE_MATMUL_INT8` and `__ARM_FEATURE_SVE` branches were omitted and that the only branch hit is the section under `__ARM_NEON`.















TODO MAKE README SHOW COLORS
```bash
brandonneway@raspberrypi:~/dev/transformers/ggml-neon-opt $ sudo perf report --stdio --no-children --symbol-filter ggml_vec_dot_q4_K_q8_K
# To display the perf.data header info, please use --header/--header-only options.
#
#
# Total Lost Samples: 0
#
# Samples: 72K of event 'cycles:P'
# Event count (approx.): 173521193861
#
# Overhead  Command      Shared Object   Symbol
# ........  ...........  ..............  ..........................
#
    66.48%  llama-bench  libggml-cpu.so  [.] ggml_vec_dot_q4_K_q8_K
            |
            |--35.96%--ggml_vec_dot_q4_K_q8_K
            |          ggml_compute_forward_mul_mat_one_chunk (inlined)
            |          ggml_compute_forward_mul_mat
            |          ggml_compute_forward (inlined)
            |          ggml_graph_compute_thread (inlined)
            |          ggml_graph_compute
            |          ggml_backend_cpu_graph_compute(ggml_backend*, ggml_cgraph*)
            |          ggml_backend_sched_graph_compute_async
            |          ggml_backend_sched_graph_compute_async
            |          llama_context::graph_compute(ggml_cgraph*, bool)
            |          llama_context::process_ubatch(llama_ubatch const&, llm_graph_type, llama_memory_context_i*, ggml_status&)
            |          llama_context::decode(llama_batch const&)
            |
            |--11.15%--vdotq_s32 (inlined)
            |          ggml_vec_dot_q4_K_q8_K
            |          ggml_compute_forward_mul_mat_one_chunk (inlined)
            |          ggml_compute_forward_mul_mat
            |          ggml_compute_forward (inlined)
            |          ggml_graph_compute_thread (inlined)
            |          ggml_graph_compute
            |          ggml_backend_cpu_graph_compute(ggml_backend*, ggml_cgraph*)
            |          ggml_backend_sched_graph_compute_async
            |          ggml_backend_sched_graph_compute_async
            |          llama_context::graph_compute(ggml_cgraph*, bool)
            |          llama_context::process_ubatch(llama_ubatch const&, llm_graph_type, llama_memory_context_i*, ggml_status&)
            |          llama_context::decode(llama_batch const&)
            |
            |--8.29%--vandq_u8 (inlined)
            |          ggml_vec_dot_q4_K_q8_K
            |          ggml_compute_forward_mul_mat_one_chunk (inlined)
            |          ggml_compute_forward_mul_mat
            |          ggml_compute_forward (inlined)
            |          ggml_graph_compute_thread (inlined)
            |          ggml_graph_compute
            |          ggml_backend_cpu_graph_compute(ggml_backend*, ggml_cgraph*)
            |          ggml_backend_sched_graph_compute_async
            |          ggml_backend_sched_graph_compute_async
            |          llama_context::graph_compute(ggml_cgraph*, bool)
            |          llama_context::process_ubatch(llama_ubatch const&, llm_graph_type, llama_memory_context_i*, ggml_status&)
            |          llama_context::decode(llama_batch const&)
            |
            |--4.49%--vaddvq_s32 (inlined)
            |          ggml_vec_dot_q4_K_q8_K
            |          ggml_compute_forward_mul_mat_one_chunk (inlined)
            |          ggml_compute_forward_mul_mat
            |          ggml_compute_forward (inlined)
            |          ggml_graph_compute_thread (inlined)
            |          ggml_graph_compute
            |          ggml_backend_cpu_graph_compute(ggml_backend*, ggml_cgraph*)
            |          ggml_backend_sched_graph_compute_async
            |          ggml_backend_sched_graph_compute_async
            |          llama_context::graph_compute(ggml_cgraph*, bool)
            |          llama_context::process_ubatch(llama_ubatch const&, llm_graph_type, llama_memory_context_i*, ggml_status&)
            |          llama_context::decode(llama_batch const&)
            |
            |--2.39%--vld1q_u8_x2 (inlined)
            |          ggml_vec_dot_q4_K_q8_K
            |          ggml_compute_forward_mul_mat_one_chunk (inlined)
            |          ggml_compute_forward_mul_mat
            |          ggml_compute_forward (inlined)
            |          ggml_graph_compute_thread (inlined)
            |          ggml_graph_compute
            |          ggml_backend_cpu_graph_compute(ggml_backend*, ggml_cgraph*)
            |          ggml_backend_sched_graph_compute_async
            |          ggml_backend_sched_graph_compute_async
            |          llama_context::graph_compute(ggml_cgraph*, bool)
            |          llama_context::process_ubatch(llama_ubatch const&, llm_graph_type, llama_memory_context_i*, ggml_status&)
            |          llama_context::decode(llama_batch const&)
            |
            |--1.52%--vld1q_s8_x2 (inlined)
            |          ggml_vec_dot_q4_K_q8_K
            |          ggml_compute_forward_mul_mat_one_chunk (inlined)
            |          ggml_compute_forward_mul_mat
            |          ggml_compute_forward (inlined)
            |          ggml_graph_compute_thread (inlined)
            |          ggml_graph_compute
            |          ggml_backend_cpu_graph_compute(ggml_backend*, ggml_cgraph*)
            |          ggml_backend_sched_graph_compute_async
            |          ggml_backend_sched_graph_compute_async
            |          llama_context::graph_compute(ggml_cgraph*, bool)
            |          llama_context::process_ubatch(llama_ubatch const&, llm_graph_type, llama_memory_context_i*, ggml_status&)
            |          llama_context::decode(llama_batch const&)
            |
            |--1.27%--neon_compute_fp16_to_fp32 (inlined)
            |          ggml_vec_dot_q4_K_q8_K
            |          ggml_compute_forward_mul_mat_one_chunk (inlined)
            |          ggml_compute_forward_mul_mat
            |          ggml_compute_forward (inlined)
            |          ggml_graph_compute_thread (inlined)
            |          ggml_graph_compute
            |          ggml_backend_cpu_graph_compute(ggml_backend*, ggml_cgraph*)
            |          ggml_backend_sched_graph_compute_async
            |          ggml_backend_sched_graph_compute_async
            |          llama_context::graph_compute(ggml_cgraph*, bool)
            |          llama_context::process_ubatch(llama_ubatch const&, llm_graph_type, llama_memory_context_i*, ggml_status&)
            |          llama_context::decode(llama_batch const&)
            |
             --1.09%--vset_lane_u32 (inlined)
                       ggml_vec_dot_q4_K_q8_K
                       ggml_compute_forward_mul_mat_one_chunk (inlined)
                       ggml_compute_forward_mul_mat
                       ggml_compute_forward (inlined)
                       ggml_graph_compute_thread (inlined)
                       ggml_graph_compute
                       ggml_backend_cpu_graph_compute(ggml_backend*, ggml_cgraph*)
                       ggml_backend_sched_graph_compute_async
                       ggml_backend_sched_graph_compute_async
                       llama_context::graph_compute(ggml_cgraph*, bool)
                       llama_context::process_ubatch(llama_ubatch const&, llm_graph_type, llama_memory_context_i*, ggml_status&)
                       llama_context::decode(llama_batch const&)

brandonneway@raspberrypi:~/dev/transformers/ggml-neon-opt $ sudo perf annotate --stdio --print-line --full-paths -s ggml_vec_dot_q4_K_q8_K

Sorted summary for file /home/brandonneway/dev/transformers/ggml-neon-opt/external/llama.cpp/build/bin/libggml-cpu.so
----------------------------------------------

   24.73 /home/brandonneway/dev/transformers/ggml-neon-opt/external/llama.cpp/ggml/src/ggml-cpu/arch/arm/quants.c:2424
   16.16 /usr/lib/gcc/aarch64-linux-gnu/12/include/arm_neon.h:29529
   13.07 /home/brandonneway/dev/transformers/ggml-neon-opt/external/llama.cpp/ggml/src/ggml-cpu/arch/arm/quants.c:2418
   12.47 /usr/lib/gcc/aarch64-linux-gnu/12/include/arm_neon.h:1122
    8.74 /home/brandonneway/dev/transformers/ggml-neon-opt/external/llama.cpp/ggml/src/ggml-cpu/arch/arm/quants.c:2386
    6.76 /usr/lib/gcc/aarch64-linux-gnu/12/include/arm_neon.h:9728
    3.60 /usr/lib/gcc/aarch64-linux-gnu/12/include/arm_neon.h:15441
    1.90 /usr/lib/gcc/aarch64-linux-gnu/12/include/arm_neon.h:15449
    1.64 /usr/lib/gcc/aarch64-linux-gnu/12/include/arm_neon.h:5667
    1.46 /home/brandonneway/dev/transformers/ggml-neon-opt/external/llama.cpp/ggml/src/ggml-cpu/simd-mappings.h:47
    1.36 /home/brandonneway/dev/transformers/ggml-neon-opt/external/llama.cpp/ggml/src/ggml-cpu/arch/arm/quants.c:2381
    0.91 /home/brandonneway/dev/transformers/ggml-neon-opt/external/llama.cpp/ggml/src/ggml-cpu/arch/arm/quants.c:2379
    0.76 /home/brandonneway/dev/transformers/ggml-neon-opt/external/llama.cpp/ggml/src/ggml-cpu/arch/arm/quants.c:2405
 Percent |      Source code & Disassembly of /home/brandonneway/dev/transformers/ggml-neon-opt/external/llama.cpp/build/bin/libggml-cpu.so for cycles:P (48177 samples, percent: loc>
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------>
         :
         :
         :
         : 3     Disassembly of section .text:
         :
         : 5     00000000000512c0 <ggml_vec_dot_q4_K_q8_K>:
         : 6     UNUSED(bs);
         :
         : 8     const block_q4_K * GGML_RESTRICT x = vx;
         : 9     const block_q8_K * GGML_RESTRICT y = vy;
         :
         : 11    const int nb = n / QK_K;
    0.00 :   512c0:  cmp     w0, #0x0
    0.00 :   512c4:  add     w15, w0, #0xff
    0.21 :   512c8:  csel    w15, w15, w0, lt        // lt = tstop
         : 15    ggml_int8x16x2_t q4bytes;
         : 16    ggml_int8x16x2_t q8bytes;
         :
         : 18    float sumf = 0;
         :
         : 20    for (int i = 0; i < nb; ++i) {
    0.00 :   512cc:  cmp     w0, #0xff
    0.00 :   512d0:  b.le    5142c <ggml_vec_dot_q4_K_q8_K+0x16c>
         : 23    float sumf = 0;
    0.35 :   512d4:  movi    v4.2s, #0x0
         :
         : 26    __extension__ extern __inline uint8x16_t
         : 27    __attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
         : 28    vandq_u8 (uint8x16_t __a, uint8x16_t __b)
         : 29    {
         : 30    return __a & __b;
    0.00 :   512d8:  movi    v7.16b, #0xf
         : 32    void ggml_vec_dot_q4_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc)>
    0.00 :   512dc:  sub     sp, sp, #0x10
         :
         : 35    __extension__ extern __inline int32x4_t
         : 36    __attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
         : 37    vdotq_s32 (int32x4_t __r, int8x16_t __a, int8x16_t __b)
         : 38    {
         : 39    return __builtin_aarch64_sdot_prodv16qi (__a, __b, __r);
    0.00 :   512e0:  movi    v6.4s, #0x0
    0.00 :   512e4:  asr     w15, w15, #8
    0.01 :   512e8:  add     x11, x5, #0x4
    0.00 :   512ec:  add     x7, x3, #0x90
         : 44    for (int i = 0; i < nb; ++i) {
    0.00 :   512f0:  mov     w13, #0x0                       // #0
    0.18 :   512f4:  mov     x14, sp
    0.00 :   512f8:  nop
    0.00 :   512fc:  nop
         : 49    const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
         : 50    const float dmin = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);
         :
         : 52    const int16x8_t q8sums = vpaddq_s16(vld1q_s16(y[i].bsums), vld1q_s16(y[i].bsums + 8));
         :
         : 54    memcpy(utmp, x[i].scales, 12);
    1.00 :   51300:  ldur    x2, [x7, #-140] // /home/brandonneway/dev/transformers/ggml-neon-opt/external/llama.cpp/ggml/src/ggml-cpu/arch/arm/quants.c:2386
    1.96 :   51304:  mov     x6, x11
         : 57    vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins)));
         : 58    sumf -= dmin * vaddvq_s32(prod);
         :
         : 60    const uint8_t * scales = (const uint8_t *)utmp;
         :
         : 62    const uint8_t * GGML_RESTRICT q4 = x[i].qs;
    0.04 :   51308:  sub     x5, x7, #0x80
    0.00 :   5130c:  mov     x3, x14
         : 65    const int8_t  * GGML_RESTRICT q8 = y[i].qs;
         :
         : 67    int32_t sumi1 = 0;
         : 68    int32_t sumi2 = 0;
    0.00 :   51310:  mov     w8, #0x0                        // #0
         : 70    memcpy(utmp, x[i].scales, 12);
    0.06 :   51314:  ldur    w10, [x7, #-132]
         : 72    int32_t sumi1 = 0;
    0.76 :   51318:  mov     w4, #0x0                        // #0 // /home/brandonneway/dev/transformers/ggml-neon-opt/external/llama.cpp/ggml/src/ggml-cpu/arch/arm/quants.c:2405
         : 74    return __builtin_aarch64_ld1v8hi ((const __builtin_aarch64_simd_hi *) __a);
    0.17 :   5131c:  ldp     q3, q1, [x11, #256]
         : 76    memcpy(utmp, x[i].scales, 12);
    5.78 :   51320:  lsr     x0, x2, #32 // /home/brandonneway/dev/transformers/ggml-neon-opt/external/llama.cpp/ggml/src/ggml-cpu/arch/arm/quants.c:2386
         : 78    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    0.04 :   51324:  lsr     w9, w2, #2
         : 80    mins8 = vset_lane_u32(((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4), mins8, 1);
    0.01 :   51328:  lsr     w12, w0, #2
         : 82    mins8 = vset_lane_u32(utmp[1] & kmask1, mins8, 0);
    0.00 :   5132c:  and     w0, w0, #0x3f3f3f3f
         : 84    #define GGML_CPU_FP16_TO_FP32(x) GGML_CPU_COMPUTE_FP16_TO_FP32(x)
         :
         : 86    static inline float neon_compute_fp16_to_fp32(ggml_fp16_t h) {
         : 87    __fp16 tmp;
         : 88    memcpy(&tmp, &h, sizeof(ggml_fp16_t));
         : 89    return (float)tmp;
    0.00 :   51330:  ldur    h2, [x7, #-142]
         : 91    mins8 = vset_lane_u32(((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4), mins8, 1);
    0.10 :   51334:  lsr     w16, w10, #4
    0.02 :   51338:  and     w12, w12, #0x30303030
         : 94    memcpy(utmp, x[i].scales, 12);
    0.00 :   5133c:  str     w10, [x14, #8]
         : 96    return __aarch64_vset_lane_any (__elem, __vec, __index);
    1.64 :   51340:  fmov    s0, w0 // /usr/lib/gcc/aarch64-linux-gnu/12/include/arm_neon.h:5667
         : 98    mins8 = vset_lane_u32(((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4), mins8, 1);
    0.03 :   51344:  and     w16, w16, #0xf0f0f0f
         : 100   utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    0.00 :   51348:  and     w9, w9, #0x30303030
         : 102   mins8 = vset_lane_u32(((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4), mins8, 1);
    0.00 :   5134c:  orr     w12, w12, w16
         : 104   const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
    0.00 :   51350:  ldur    s5, [x11, #-4]
         : 106   utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    0.42 :   51354:  and     w10, w10, #0xf0f0f0f
         : 108   utmp[0] &= kmask1;
    0.04 :   51358:  and     w2, w2, #0x3f3f3f3f
         : 110   utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    0.01 :   5135c:  orr     w0, w9, w10
    1.46 :   51360:  ldur    h20, [x7, #-144] // /home/brandonneway/dev/transformers/ggml-neon-opt/external/llama.cpp/ggml/src/ggml-cpu/simd-mappings.h:47
    0.11 :   51364:  fcvt    s2, h2
    0.00 :   51368:  stp     w2, w0, [sp]
         : 115   return __builtin_aarch64_addpv8hi (__a, __b);
    0.01 :   5136c:  addp    v3.8h, v3.8h, v1.8h
         : 117   return __aarch64_vset_lane_any (__elem, __vec, __index);
    0.00 :   51370:  mov     v0.s[1], w12
    0.34 :   51374:  fcvt    s20, h20
         : 120   const float dmin = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);
    0.01 :   51378:  fmul    s2, s2, s5
         : 122   const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
    1.36 :   5137c:  fmul    s20, s20, s5 // /home/brandonneway/dev/transformers/ggml-neon-opt/external/llama.cpp/ggml/src/ggml-cpu/arch/arm/quants.c:2381
         : 124   return __builtin_aarch64_uxtlv8hi_uu (__a);
    0.06 :   51380:  uxtl    v0.8h, v0.8b
         : 126   return __builtin_aarch64_intrinsic_vec_smult_lo_v4hi (__a, __b);
    0.01 :   51384:  smull2  v1.4s, v3.8h, v0.8h
         : 128   return __a + __b;
    0.00 :   51388:  smlal   v1.4s, v3.4h, v0.4h
         : 130   return __builtin_aarch64_reduc_plus_scal_v4si (__a);
    0.00 :   5138c:  addv    s0, v1.4s
         : 132   sumf -= dmin * vaddvq_s32(prod);
    0.34 :   51390:  scvtf   s0, s0
    0.04 :   51394:  fmsub   s21, s0, s2, s4
         : 135   return __builtin_aarch64_ld1x2v16qi (
    1.25 :   51398:  mov     x0, x6 // /usr/lib/gcc/aarch64-linux-gnu/12/include/arm_neon.h:15449
         : 137   return __builtin_aarch64_ld1x2v16qi_us (
    3.60 :   5139c:  ld1     {v0.16b, v1.16b}, [x5], #32 // /usr/lib/gcc/aarch64-linux-gnu/12/include/arm_neon.h:15441
         : 139   q4bytes.val[1] = vreinterpretq_s8_u8(vandq_u8  (q4bits.val[1], m4b));
         :
         : 141   const int32x4_t p1 = ggml_vdotq_s32(ggml_vdotq_s32(mzero, q4bytes.val[0], q8bytes.val[0]), q4bytes.val[1], q8bytes.val[1]);
         : 142   sumi1 += vaddvq_s32(p1) * scales[2*j+0];
         :
         : 144   q8bytes = ggml_vld1q_s8_x2(q8); q8 += 32;
   13.07 :   513a0:  add     x6, x6, #0x40 // /home/brandonneway/dev/transformers/ggml-neon-opt/external/llama.cpp/ggml/src/ggml-cpu/arch/arm/quants.c:2418
         : 146   for (int j = 0; j < QK_K/64; ++j) {
    0.00 :   513a4:  add     x3, x3, #0x2
         : 148   return __builtin_aarch64_ld1x2v16qi (
    0.38 :   513a8:  ld1     {v4.16b, v5.16b}, [x0], #32
         : 150   q4bytes.val[0] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[0], 4));
         : 151   q4bytes.val[1] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[1], 4));
         :
         : 153   const int32x4_t p2 = ggml_vdotq_s32(ggml_vdotq_s32(mzero, q4bytes.val[0], q8bytes.val[0]), q4bytes.val[1], q8bytes.val[1]);
         :
         : 155   sumi2 += vaddvq_s32(p2) * scales[2*j+1];
   24.73 :   513ac:  ldurb   w9, [x3, #-1] // /home/brandonneway/dev/transformers/ggml-neon-opt/external/llama.cpp/ggml/src/ggml-cpu/arch/arm/quants.c:2424
         : 157   return __a & __b;
   10.73 :   513b0:  and     v19.16b, v7.16b, v0.16b // /usr/lib/gcc/aarch64-linux-gnu/12/include/arm_neon.h:1122
         : 159   return __builtin_aarch64_lshrv16qi_uus (__a, __b);
    0.23 :   513b4:  ushr    v18.16b, v0.16b, #4
         : 161   sumi1 += vaddvq_s32(p1) * scales[2*j+0];
    0.04 :   513b8:  ldurb   w10, [x3, #-2]
         : 163   return __a & __b;
    1.74 :   513bc:  and     v17.16b, v7.16b, v1.16b
         : 165   return __builtin_aarch64_lshrv16qi_uus (__a, __b);
    0.00 :   513c0:  ushr    v16.16b, v1.16b, #4
         : 167   return __builtin_aarch64_ld1x2v16qi (
    0.65 :   513c4:  ld1     {v2.16b, v3.16b}, [x0] // /usr/lib/gcc/aarch64-linux-gnu/12/include/arm_neon.h:15449
         : 169   return __builtin_aarch64_sdot_prodv16qi (__a, __b, __r);
   15.23 :   513c8:  mov     v1.16b, v6.16b // /usr/lib/gcc/aarch64-linux-gnu/12/include/arm_neon.h:29529
    0.93 :   513cc:  mov     v0.16b, v6.16b
    0.15 :   513d0:  sdot    v1.4s, v19.16b, v4.16b
    0.01 :   513d4:  sdot    v0.4s, v18.16b, v2.16b
    0.45 :   513d8:  sdot    v1.4s, v17.16b, v5.16b
    0.00 :   513dc:  sdot    v0.4s, v16.16b, v3.16b
         : 176   return __builtin_aarch64_reduc_plus_scal_v4si (__a);
    0.64 :   513e0:  addv    s1, v1.4s // /usr/lib/gcc/aarch64-linux-gnu/12/include/arm_neon.h:9728
    3.96 :   513e4:  addv    s0, v0.4s
    1.16 :   513e8:  fmov    w2, s1
    1.00 :   513ec:  fmov    w0, s0
    0.13 :   513f0:  madd    w4, w10, w2, w4
         : 182   sumi2 += vaddvq_s32(p2) * scales[2*j+1];
    0.01 :   513f4:  madd    w8, w9, w0, w8
         : 184   for (int j = 0; j < QK_K/64; ++j) {
    0.48 :   513f8:  cmp     x5, x7
    0.00 :   513fc:  b.ne    51398 <ggml_vec_dot_q4_K_q8_K+0xd8>  // b.any
         : 187   }
         :
         : 189   sumf += d * (sumi1 + sumi2);
    0.46 :   51400:  add     w4, w4, w8
         : 191   for (int i = 0; i < nb; ++i) {
    0.91 :   51404:  add     w13, w13, #0x1 // /home/brandonneway/dev/transformers/ggml-neon-opt/external/llama.cpp/ggml/src/ggml-cpu/arch/arm/quants.c:2379
    0.38 :   51408:  add     x11, x11, #0x124
    0.07 :   5140c:  add     x7, x7, #0x90
         : 195   sumf += d * (sumi1 + sumi2);
    0.03 :   51410:  scvtf   s0, w4
    0.08 :   51414:  fmadd   s4, s0, s20, s21
         : 198   for (int i = 0; i < nb; ++i) {
    0.00 :   51418:  cmp     w15, w13
    0.41 :   5141c:  b.gt    51300 <ggml_vec_dot_q4_K_q8_K+0x40>
         :
         : 202   }
         :
         : 204   *s = sumf;
    0.37 :   51420:  str     s4, [x1]
         : 206   sumf -= dmin * sumi;
         : 207   }
         : 208   for (int l = 0; l < 8; ++l) sumf += sums[l];
         : 209   *s = sumf;
         : 210   #endif
         : 211   }
    0.13 :   51424:  add     sp, sp, #0x10
    0.00 :   51428:  ret
         : 214   float sumf = 0;
    0.00 :   5142c:  movi    v4.2s, #0x0
         : 216   *s = sumf;
    0.00 :   51430:  str     s4, [x1]
    0.00 :   51434:  ret
brandonneway@raspberrypi:~/dev/transformers/ggml-neon-opt $
```
