import pywintypes
import win32gui
import win32api
import win32process
import ctypes
from ctypes import wintypes
import time

Psapi = ctypes.WinDLL('Psapi.dll')
Kernel32 = ctypes.WinDLL('kernel32.dll')
PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_READ = 0x0010


def EnumProcessModulesEx(hProcess):
    buf_count = 256
    while True:
        LIST_MODULES_ALL = 0x03
        buf = (ctypes.wintypes.HMODULE * buf_count)()
        buf_size = ctypes.sizeof(buf)
        needed = ctypes.wintypes.DWORD()
        if not Psapi.EnumProcessModulesEx(hProcess, ctypes.byref(buf), buf_size, ctypes.byref(needed),
                                          LIST_MODULES_ALL):
            raise OSError('EnumProcessModulesEx failed')
        if buf_size < needed.value:
            buf_count = needed.value // (buf_size // buf_count)
            continue
        count = needed.value // (buf_size // buf_count)
        return map(ctypes.wintypes.HMODULE, buf[:count])


class Hp_getter():
    def __init__(self):
        hd = win32gui.FindWindow(None, "Dead Cells")
        pid = win32process.GetWindowThreadProcessId(hd)[1]
        self.process_handle = win32api.OpenProcess(0x1F0FFF, False,
                                                   pid)  # this is fixed number if environment is windows
        self.kernal32 = ctypes.windll.LoadLibrary(r"C:\\Windows\\System32\\kernel32.dll")

        self.hx = 0
        # get dll address
        hProcess = Kernel32.OpenProcess(
            PROCESS_QUERY_INFORMATION | PROCESS_VM_READ,
            False, pid)
        hModule = EnumProcessModulesEx(hProcess)
        for i in hModule:
            temp = win32process.GetModuleFileNameEx(self.process_handle, i.value)
            # libhl.dll is the base address where we find all the needed data
            if temp[-9:] == "libhl.dll":  # from -9 because 'lebhl.dll' is length 9
                self.libhl = i.value

    def get_self_hp(self):
        base_address = self.libhl + 0x00048184  # this is find from cheat engine
        offset_address = ctypes.c_long()
        offset_list = [0x42C, 0x0, 0x58, 0x64, 0x10C]  # in cheat engine upper offset is closer to the end of the list
        self.kernal32.ReadProcessMemory(int(self.process_handle), base_address, ctypes.byref(offset_address), 4, None)
        for offset in offset_list:
            self.kernal32.ReadProcessMemory(int(self.process_handle), offset_address.value + offset,
                                            ctypes.byref(offset_address), 4, None)  # 4 means 4 bytes data
        return offset_address.value


    def get_self_location(self):
        base_address = self.libhl + 0x00048184
        offset_address = ctypes.c_long()
        offset_list = [0x650, 0x0, 0x18, 0x64, 0x64]  # in cheat engine upper offset is closer to the end of the list
        self.kernal32.ReadProcessMemory(int(self.process_handle), base_address, ctypes.byref(offset_address), 4, None)
        for offset in offset_list:
            self.kernal32.ReadProcessMemory(int(self.process_handle), offset_address.value + offset,
                                            ctypes.byref(offset_address), 4, None)  # 4 means 4 bytes data
        return offset_address.value

    def get_boss_loca(self):
        base_address = self.libhl + 0x00048184
        offset_address = ctypes.c_long()
        offset_list = [0x64C, 0x0, 0x18, 0x104, 0x64]  # in cheat engine upper offset is closer to the end of the list
        self.kernal32.ReadProcessMemory(int(self.process_handle), base_address, ctypes.byref(offset_address), 4, None)
        for offset in offset_list:
            self.kernal32.ReadProcessMemory(int(self.process_handle), offset_address.value + offset,
                                            ctypes.byref(offset_address), 4, None)  # 4 means 4 bytes data
        return offset_address.value

    def get_boss_hp(self):
        base_address = self.libhl + 0x00048184
        offset_address = ctypes.c_long()
        offset_list = [0x64C, 0x0, 0x18, 0x104, 0x10C]  # in cheat engine upper offset is closer to the end of the list
        self.kernal32.ReadProcessMemory(int(self.process_handle), base_address, ctypes.byref(offset_address), 4, None)
        for offset in offset_list:
            self.kernal32.ReadProcessMemory(int(self.process_handle), offset_address.value + offset,
                                            ctypes.byref(offset_address), 4, None)  # 4 means 4 bytes data
        return offset_address.value


cl = Hp_getter()
while (1):
    sh = cl.get_self_hp()
    bh = cl.get_boss_hp()
    sl = cl.get_self_location()
    bl = cl.get_boss_loca()
    print('self location is : {} ; boss location is: {} '.format(sl, bl))
    print('self health is : {} ; boss health is: {} '.format(sh, bh))
    time.sleep(1)
