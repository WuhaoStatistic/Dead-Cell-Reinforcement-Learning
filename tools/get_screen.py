import pywintypes
import cv2
import numpy as np
import win32gui, win32ui, win32con, win32api
import torch
import matplotlib.pyplot as plt
from PIL import Image


def grab_screen(hwnd, region=[10, 25, 1285, 750]):
    if region:
        left, top, x2, y2 = region
        width = x2 - left + 1
        height = y2 - top + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwnd)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)

    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height, width, 4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return img


# the region here needs configuration. This is the best in my computer
# dead cell can not set resolution when in the window form
# regions is the offset from the left top of the window.

#--------------------------------------------------------------------------------
# use code below to get standard parameter
# k = np.empty((10000, 256, 256))
# i = 0
# while i < 10000:
#     hwnd1 = win32gui.FindWindow(None, 'Dead Cells')
#     bmp1 = grab_screen(hwnd1)
#     bmp1 = cv2.resize(cv2.cvtColor(bmp1, cv2.COLOR_BGR2GRAY), (256, 256))
#     k[i, :, :] = bmp1
#     i += 1
#     if i == 9999:
#         break
# k = k.flatten()
# print(np.mean(k), np.std(k))
#--------------------------------------------------------------------------------

