import os
import time
import subprocess

subprocess.call('python code.py',shell=True)

os.remove('1.pdf')
os.remove('2.pdf')
os.remove('3.pdf')
os.remove('4.pdf')
os.remove('5.pdf')
