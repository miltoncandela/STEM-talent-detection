
# Same subject level, but looking at the FC across time, for each type of class

clusters = {'Novice': ['ST01'],
            'Experienced': ['DT01'],
            'Master': ['MT01']}

# Considering subject ST01

import pyautogui as pt
from time import sleep
import os

def check_measures():
    try:
        while True:
            x, y = pt.position()
            print(x, y)
            sleep(1)
    except KeyboardInterrupt:
        pass


off = 25

def ok():
    position = None
    while position is None:
        position = pt.locateOnScreen('icons/ok.png', confidence=.9)
    x, y = position[0], position[1]
    pt.moveTo(x + 10, y + 10)
    pt.click()
    sleep(1)


def pop_newset():
    position = None
    while position is None:
        position = pt.locateOnScreen('icons/pop_newset.png', confidence=.9)
    x, y = position[0], position[1]
    pt.moveTo(x + 5, y + 5)
    pt.click()
    ok()


def load_file():
    # position = pt.locateOnScreen('icons/matlab.png', confidence=.5)
    # x, y = position[0], position[1]
    # pt.moveTo(x + 750, y + 10)
    # pt.click()

    # sleep(0.5)

    position = pt.locateOnScreen('icons/command.png', confidence=.6)
    x, y = position[0], position[1]
    pt.moveTo(x + 50, y + 50)
    pt.click()

    pt.typewrite('eeglab', interval=.01)
    pt.press('enter')

    position = None
    while position is None:
        position = pt.locateOnScreen('icons/eeglab.png', confidence=.6)
    x, y = position[0], position[1]
    sleep(1)
    pt.moveTo(x, y)

    # Open load box
    pt.moveTo(x + 8, y + 50)
    pt.click()
    pt.moveTo(x + 104, y + 77)
    sleep(0.25)
    pt.moveTo(x + 372, y + 77)
    sleep(0.25)
    pt.moveTo(x + 674, y + 107)
    sleep(0.25)
    pt.click()

    sleep(3)

    position = pt.locateOnScreen('icons/pop_importdata.png', confidence=.6)
    x, y = position[0], position[1]

    pt.moveTo(x + 662, y + 74)
    pt.click()
    pt.moveTo(x + 667, y + 140)
    pt.click()

    pt.moveTo(x + 935, y + 72)
    pt.click()
    pt.typewrite("C:\\Users\\Milton\\PycharmProjects\\BRAIN-MCE\\data\\Frontiers\\EEG_Txt\\{}".format(data), interval=.01)

    pt.moveTo(x + 869, y + 103)
    pt.click()
    pt.typewrite(data.split('.txt')[0], interval=.01)

    pt.moveTo(x + 604, y + 170)
    pt.click()
    pt.press('del')
    pt.typewrite('250', interval=.01)

    pt.moveTo(x + 1177, y + 528)
    pt.click()

    # sleep(10)
    ok()


def assign_channlocs():
    position = pt.locateOnScreen('icons/eeglab.png', confidence=.5)
    x, y = position[0], position[1]
    pt.moveTo(x + 390 - 332, y + 396 - 350)
    pt.click()
    pt.moveTo(x + 480 - 332, y + 500 - 350)
    pt.click()
    ok()

    def insert_channel(ch):
        position = None
        while position is None:
            position = pt.locateOnScreen('icons/chanedit1.png', confidence=.7)
        x, y = position[0], position[1]
        pt.moveTo(x + 10, y + 10)

        if channel != 'FP2':  # First channel
            pt.click()
            pt.typewrite(ch, interval=.01)

            # Channel in data array
            position = pt.locateOnScreen('icons/chanedit1-5.png', confidence=.7)
            x, y = position[0], position[1]
            pt.moveTo(x + 1010 - 454, y + 590 - 578)
            pt.click()
        else:
            pt.doubleClick()
            pt.press('backspace')
            pt.typewrite(ch, interval=.01)

        # Loop up locs
        position = pt.locateOnScreen('icons/chanedit2.png', confidence=.7)
        x, y = position[0], position[1]
        pt.moveTo(x, y)
        pt.click()
        sleep(1)
        ok()

        # If the last channel, do not need to insert it
        if channel != 'FP1':
            position = pt.locateOnScreen('icons/chanedit3.png', confidence=.7)
            x, y = position[0], position[1]
            pt.moveTo(x, y)
            pt.click()

    for channel in ['FP2', 'F4', 'C4', 'Pz', 'C3', 'F3', 'FP1']:
        insert_channel(channel)
        sleep(1)

    ok()


def fir():
    position = pt.locateOnScreen('icons/eeglab.png', confidence=.5)
    x, y = position[0], position[1]
    pt.moveTo(x + 915 - 811, y + 310 - 262)
    pt.click()
    pt.moveTo(x + 1050 - 811, y + 393 - 262)
    sleep(0.25)
    pt.moveTo(x + 1385 - 811, y + 393 - 262)
    pt.click()

    sleep(2)

    position = pt.locateOnScreen('icons/pop_eegfiltnew.png', confidence=.5)
    x, y = position[0], position[1]
    pt.moveTo(x + 1240 - 552, y + 262 - 188)
    pt.click()
    pt.typewrite('0.1', interval=.01)

    pt.moveTo(x + 1243 - 552, y + 294 - 188)
    pt.click()
    pt.typewrite('50', interval=.01)

    pt.moveTo(x + 600 - 552, y + 490 - 188)
    pt.click()

    ok()
    pop_newset()


def prep():
    position = pt.locateOnScreen('icons/eeglab.png', confidence=.5)
    x, y = position[0], position[1]
    pt.moveTo(x + 705 - 598, y + 345 - 290)
    pt.click()
    pt.moveTo(x + 700 - 598, y + 720 - 290)
    pt.click()

    sleep(2)
    ok()
    # sleep(120)
    pop_newset()
    sleep(10)


def asr():
    position = pt.locateOnScreen('icons/eeglab.png', confidence=.5)
    x, y = position[0], position[1]
    pt.moveTo(x + 440 - 333, y + 400 - 348)
    pt.click()
    pt.moveTo(x + 440 - 333, y + 573 - 348)
    pt.click()

    sleep(2)

    position = pt.locateOnScreen('icons/pop_clean_rawdata.png', confidence=.5)
    x, y = position[0], position[1]
    pt.moveTo(x + 605 - 560, y + 167 - 90)
    pt.click()
    pt.moveTo(x + 605 - 560, y + 240 - 90)
    pt.click()
    pt.moveTo(x + 655 - 560, y + 475 - 90)
    pt.click()
    pt.moveTo(x + 605 - 560, y + 515 - 90)
    pt.click()
    pt.moveTo(x + 605 - 560, y + 625 - 90)
    pt.click()

    ok()
    pop_newset()


def ica():
    position = pt.locateOnScreen('icons/eeglab.png', confidence=.5)
    x, y = position[0], position[1]
    pt.moveTo(x + 440 - 333, y + 400 - 348)
    pt.click()
    pt.moveTo(x + 440 - 333, y + 600 - 348)
    pt.click()

    sleep(2)
    ok()

    # sleep(100 + off)

    position = None
    while position is None:
        position = pt.locateOnScreen('icons/doneICA.png', confidence=.5)


def save_file():
    position = pt.locateOnScreen('icons/eeglab.png', confidence=.5)
    x, y = position[0], position[1]
    pt.moveTo(x + 880 - 860, y + 334 - 283)
    pt.click()
    sleep(0.25)
    pt.moveTo(x + 880 - 860, y + 515 - 283)
    pt.click()

    sleep(2)

    position = pt.locateOnScreen('icons/pop_saveset.png', confidence=.5)
    x, y = position[0], position[1]
    pt.moveTo(x + 1595 - 870, y + 785 - 320)
    pt.click()
    pt.typewrite(data[:(len(data) - 4)], interval=.01)
    sleep(1)

    pt.moveTo(x + 1595 - 870, y + 880 - 320)
    pt.click()


def closeEEG():
    position = pt.locateOnScreen('icons/eeglab.png', confidence=.5)
    x, y = position[0], position[1]
    pt.moveTo(x + 1475 - 863, y + 300 - 284)
    pt.click()
    sleep(1)

    pt.moveTo(960, 860)
    pt.click()
    pt.typewrite('clearvars', interval=.01)
    pt.press('enter')
    pt.typewrite('clc', interval=.01)
    pt.press('enter')

    pt.moveTo(960, 400)
    pt.click()


# check_measures()
sleep(5)
skip = True

# Calib
# DJ03_Desi, DJ04_Desi, ST01_Prog

for data in os.listdir('data/Frontiers/EEG_Txt'):
    if data == 'MJ01_Desi.txt':
        skip = False
    if skip:
        continue
    # Abre Matlab y abre eeglab
    load_file()
    sleep(3)
    assign_channlocs()
    sleep(2)

    # PREP
    prep()
    sleep(2)

    # FIR 0.1 - 50 Hz
    fir()
    sleep(2)

    # Drift & ASR
    asr()
    sleep(2)

    # ICA
    ica()
    sleep(2)

    save_file()
    sleep(15)

    closeEEG()
    sleep(2)
