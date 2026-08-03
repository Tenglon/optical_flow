"""cv2.cuda TV-L1 baseline at 480x854 and 720x1280 (resident + 4 streams).

Run with the CUDA OpenCV env:
  srun -p gpu -w ilps-cn119 --gres=gpu:1 ... \
    /fnwi_fs/ivi/irlab/personal/tlong/conda_envs/opencv-cuda/bin/python cv2_hr.py
"""
import glob
import time

import cv2

PARAMS = dict(tau=0.25, lambda_=0.15, theta=0.3, nscales=5, warps=5,
              epsilon=0.01, iterations=300, scaleStep=0.8, gamma=0.0)
RESOLUTIONS = [(480, 854), (720, 1280)]
N_STREAMS = 4


def load(hw):
    frames = []
    for d in sorted(glob.glob("data/other-data/*"))[:8]:
        f1 = cv2.imread(f"{d}/frame10.png", cv2.IMREAD_GRAYSCALE)
        f2 = cv2.imread(f"{d}/frame11.png", cv2.IMREAD_GRAYSCALE)
        f1 = cv2.resize(f1, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)
        f2 = cv2.resize(f2, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)
        frames.append((f1, f2))
    return frames


def main():
    print(cv2.__version__, "cuda devices:", cv2.cuda.getCudaEnabledDeviceCount())
    for hw in RESOLUTIONS:
        frames = load(hw)
        gpu = []
        for f1, f2 in frames:
            g1, g2 = cv2.cuda_GpuMat(), cv2.cuda_GpuMat()
            g1.upload(f1)
            g2.upload(f2)
            gpu.append((g1, g2))

        # resident, single engine
        eng = cv2.cuda_OpticalFlowDual_TVL1.create(**PARAMS)
        for g1, g2 in gpu[:2]:
            eng.calc(g1, g2, None)
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            for g1, g2 in gpu:
                eng.calc(g1, g2, None)
            ts.append(time.perf_counter() - t0)
        print(f"tvl1 {hw[0]}x{hw[1]} resident : {min(ts) / len(gpu) * 1e3:8.2f} ms/pair", flush=True)

        # 4 streams x 4 engines
        engines = [cv2.cuda_OpticalFlowDual_TVL1.create(**PARAMS) for _ in range(N_STREAMS)]
        streams = [cv2.cuda_Stream() for _ in range(N_STREAMS)]
        for k, (g1, g2) in enumerate(gpu[:N_STREAMS]):
            engines[k].calc(g1, g2, None, stream=streams[k])
        for s in streams:
            s.waitForCompletion()
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            for k, (g1, g2) in enumerate(gpu):
                engines[k % N_STREAMS].calc(g1, g2, None, stream=streams[k % N_STREAMS])
            for s in streams:
                s.waitForCompletion()
            ts.append(time.perf_counter() - t0)
        print(f"tvl1 {hw[0]}x{hw[1]} 4streams : {min(ts) / len(gpu) * 1e3:8.2f} ms/pair", flush=True)


if __name__ == "__main__":
    main()
