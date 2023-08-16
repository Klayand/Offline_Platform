import subprocess
import os


def run_py_and_exe():
    py_process = subprocess.Popen(['python', os.path.abspath(os.path.join(os.getcwd(), 'run.py'))])
    exe_process = subprocess.Popen([os.path.abspath(os.path.join(os.getcwd(), 'Light/Light.exe'))])

    py_process.wait()

    exe_process.wait()


if __name__ == '__main__':
     run_py_and_exe()

