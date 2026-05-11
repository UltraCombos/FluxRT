# FluxRT Benchmark Report

## Configuration

```json
{
  "default_prompt": "Turn this into art.",
  "default_steps": 2,
  "default_seed": 52,
  "models_path": "FLUX.2-klein-4B",
  "int8_models_path": "FLUX.2-klein-4B-int8",
  "resolution": {
    "height": 320,
    "width": 576
  },
  "compile_models": true,
  "enable_spatial_cache": true,
  "enable_int8_quantization": false,
  "target_fps": null,
  "interpolation_exp": 2,
  "use_reference_image": false,
  "logging": false
}
```

## Hardware Information

```json
{
  "platform": "Linux-6.19.6-arch1-1-x86_64-with-glibc2.43",
  "python": "3.12.13",
  "cpu": "AMD Ryzen 9 9950X 16-Core Processor",
  "cpu_cores_logical": 32,
  "gpu": [
    {
      "name": "NVIDIA GeForce RTX 5090",
      "vram_gb": 31.36,
      "cc": "12.0"
    }
  ]
}
```

## Results

| Dynamic Area | Processing Time (s) | FPS |
|-------------:|--------------------:|----:|
| 0% | 0.0752 | 53.17 |
| 10% | 0.0974 | 43.18 |
| 25% | 0.1163 | 34.95 |
| 50% | 0.1455 | 27.52 |
| 75% | 0.1741 | 23.23 |
| 90% | 0.1772 | 22.62 |
| 100% | 0.1829 | 22.15 |

**End-to-end latency:** 0.2207 s

**Reserved GPU memory:** 22.5254 GB