# -*- coding: utf-8 -*-
# Citation: Box Of Hats (https://github.com/Box-Of-Hats )

import win32api as wapi
import win32con as wcon
import time

keyList = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'£$/\\":
    keyList.append(char)


def key_check():
    keys = []
    for key in keyList:
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)
    if wapi.GetAsyncKeyState(wcon.VK_UP):
        keys.append('up')
    if wapi.GetAsyncKeyState(wcon.VK_DOWN):
        keys.append('down')
    if wapi.GetAsyncKeyState(wcon.VK_RIGHT):
        keys.append('right')
    if wapi.GetAsyncKeyState(wcon.VK_LEFT):
        keys.append('left')
    if wapi.GetAsyncKeyState(wcon.VK_SPACE):
        keys.append('space')

    return keys


def keys_to_output_movement(keys):
    """
    Convert keys to a ...multi-hot... array
    ['up - 0', 'down - 1', 'left - 2', 'right - 3', 'none - 4']
    """
    output = [0, 0, 0, 0, 0]

    if 'left' in keys:
        output[2] = 1
    elif 'up' in keys:
        output[0] = 1
    elif 'down' in keys:
        output[1] = 1
    elif 'right' in keys:
        output[3] = 1
    else:
        output[4] = 1

    return output


def keys_to_output_action(keys):
    """
    Convert keys to a ...multi-hot... array
    ['shoot - 0 - space', 'pass - 1 - w', 'through - 2 - q', 'cross - 3 - f', 'none - 4']
    """
    output = [0, 0, 0, 0, 0]

    if 'space' in keys:
        output[0] = 1
    elif 'W' in keys:
        output[1] = 1
    elif 'Q' in keys:
        output[2] = 1
    elif 'F' in keys:
        output[3] = 1
    else:
        output[4] = 1

    return output
