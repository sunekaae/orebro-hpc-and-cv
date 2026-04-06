# Mini Project: High-Performance Computing and Computer Vision

# Track C: JIT Compilation

## Introduction

This project investigates the acceleration of a core computation in a computer vision pipeline.

YOLOv8 (Ultralytics implementation) is used to generate object detections on a traffic image dataset, from which bounding boxes are extracted. The focus is on computing pairwise Intersection over Union (IoU) between these boxes; this is a common operation in detection pipelines that is computationally expensive in pure Python due to nested loops.

A baseline Python implementation is compared with an optimized version using Numba JIT compilation.

# Benchmark Methodology

## Objective

The goal of the benchmark is to compare the execution time of a pure Python implementation of pairwise Intersection over Union (IoU) computation with an optimized implementation using Numba JIT compilation. This will include an analysis of the performance gains and trade-offs of applying JIT compilation to this workload.

It is important to also ensure correctness of the optimized computation.

---

## Workload Definition

A traffic dataset containing 160 images from Hugging Face is used. Objection detection using YOLOv8 inference is run on this dataset resulting in approximately 1300 bounding boxes in total.

The workload consists of computing pairwise Intersection over Union (IoU) for these detections.

- Dataset size: 160 images  
- Total detections: ~1300 bounding boxes  
- Average detections per image: ~8  

A single pass over the dataset was found to be too lightweight for stable benchmarking. Therefore, the workload is repeated multiple times.

A single pass over this dataset was found to be insufficiently computationally intensive for reliable benchmarking. To address this, the workload is repeated multiple times over the same set of detections.

- Final configuration: **50 repeated passes** over the same detection set  
- This results in a baseline runtime of approximately 5 seconds  


The same detection data is reused across all repetitions to ensure identical inputs for both baseline and optimized implementations.

---

## Implementations Compared

Two implementations are evaluated:

- **Baseline:** pure Python implementation using nested loops  
- **Optimized:** Numba JIT-compiled implementation using `@njit`

Both implementations compute the same IoU matrix for each image.

---

## Timing Procedure

Execution time is measured using `time.perf_counter()`.

For each benchmark:

- The full repeated workload (all images × repetitions) is timed  
- Only the IoU computation is included in the measurement  
- Dataset loading, image saving, and YOLO inference are excluded  

---

## Warm-up and Compilation Handling

Numba performs Just-In-Time compilation on the first function call.

To ensure a fair comparison:

- A **warm-up call** is executed before timing the Numba implementation  
- The **first-call time (compile + execute)** is measured and reported separately 
- This excludes compilation overhead from the steady-state benchmark  

The Python implementation does not require warm-up, as it does not involve compilation.

---


## Hardware Environment

All experiments are executed in a Google Colab environment using a standard CPU configuration.

---

## Metrics Reported

The following metrics are reported:

- Total execution time (Python vs Numba)  
- Per-image execution time  
- Speedup factor for steady-state and end-to-end
- Numba first-call time (compile + execute)  

These metrics provide both absolute and normalized views of performance.

## Results

- Python baseline runtime: 4.5865 s
- Numba steady-state runtime: 0.0513 s  
- Steady-state speedup: ~88×
- Numba first-run time (compile + execute): 1.7363 s
- End-to-end speedup: 2,57x

## Limitations

- The dataset is relatively small (160 images), requiring repeated passes to create a sufficiently large workload.  
- The benchmark focuses on a custom IoU kernel rather than full object detection, so end-to-end performance gains would be smaller.  

# Correctness Validation

- The outputs of the Python baseline and Numba JIT implementations are compared  
- Equality is verified using `np.allclose` with a tolerance of `1e-6`  
- The maximum (worst-case) absolute difference across all outputs was 0.00000024 s and the mean was 0.00000000.

## Discussion

The large speedup (~88x) is primarily due to the difference between Python's interpreted execution and Numba's compiled code. The baseline implementation relies on nested Python loops, which introduce significant interpreter overhead. Numba eliminates this overhead by compiling the computation into efficient native code.

However, this speedup applies only to the IoU computation kernel. If this IoU calculation workload was seen as a part of an overall pipeline including Yolo inference then the speed-up would of course be much smaller.

The workload is scaled through repeated passes over the same dataset. This approach ensures stable timing and a sufficiently large benchmark, but it it worth noting that the image dataset itself is relatively small (160 images).

## Conclusion

This project demonstrates that JIT compilation using Numba can significantly accelerate a computational kernel in a computer vision pipeline. The IoU computation achieved a steady-state speedup of approximately 88x while maintaining numerical correctness.

The results highlight the importance of targeting performance-critical kernels and show that even simple optimizations can yield substantial gains when applied to loop-heavy numerical workloads.