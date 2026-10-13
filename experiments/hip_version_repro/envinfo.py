"""Print the environment's versions and check that only GPU 0 is visible."""
import ctypes, os
import torch
import triton

assert os.environ.get("HIP_VISIBLE_DEVICES") == "0", os.environ.get("HIP_VISIBLE_DEVICES")
assert torch.cuda.device_count() == 1, torch.cuda.device_count()
x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
(x @ x).sum().item()
path = next(l.split()[-1] for l in open(f"/proc/{os.getpid()}/maps") if "libhipblaslt.so" in l)
lib = ctypes.CDLL(path)
h, v, buf = ctypes.c_void_p(), ctypes.c_int(), ctypes.create_string_buffer(256)
lib.hipblasLtCreate(ctypes.byref(h)); lib.hipblasLtGetVersion(h, ctypes.byref(v)); lib.hipblasLtGetGitRevision(h, buf)
p = torch.cuda.get_device_properties(0)
print(f"env={os.environ.get('HVR_ENV')} torch={torch.__version__} hip={torch.version.hip} "
      f"hipblaslt={v.value}({buf.value.decode()}) triton={triton.__version__}@{os.path.dirname(triton.__file__)} "
      f"gpu_pci_bus={getattr(p, 'pci_bus_id', 'n/a')} gpu={p.gcnArchName} cus={p.multi_processor_count}")
