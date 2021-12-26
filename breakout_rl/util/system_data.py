import pynvml
import psutil

pynvml.nvmlInit()

def print_gpu_data():
    device_count = pynvml.nvmlDeviceGetCount()
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        print("Device", i, ":", pynvml.nvmlDeviceGetName(handle))

def get_mem_msg():
    mem_info = psutil.virtual_memory()
    used = round(mem_info.used / 1024 ** 3, 2)
    total = round(mem_info.total / 1024 ** 3, 2)
    msg = f'Mem: {used}/{total} GB, GPUs: '

    device_count = pynvml.nvmlDeviceGetCount()
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used = round(mem_info.used / 1024 ** 3, 2)
        total = round(mem_info.total / 1024 ** 3, 2)
        msg += f'{used}/{total} GB, '

    return msg
