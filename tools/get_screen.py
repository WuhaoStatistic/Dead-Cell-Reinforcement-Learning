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
# hwnd1 = win32gui.FindWindow(None, 'Dead Cells')
# bmp1 = grab_screen(hwnd1)
# bmp1 = cv2.resize(bmp1, (400, 400))
# bmp1 = bmp1[:, :, :3]
# bmp1 = Image.fromarray(cv2.cvtColor(bmp1, cv2.COLOR_BGR2RGB))
# bmp1.show()

# bmp1 = torch.from_numpy(bmp1)
# # bmp1 = bmp1.permute(2, 0, 1)
# bmp1 = np.array(bmp1)
# bmp1.show()
# # cv2.imshow('2', bmp1)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
