import subprocess
import os


def run_py_and_exe():
    py_process = subprocess.call(['python', os.path.abspath(os.path.join(os.getcwd(), 'web.py'))])


if __name__ == '__main__':
     run_py_and_exe()