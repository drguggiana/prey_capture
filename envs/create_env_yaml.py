# script to generate environment.yaml files for all environments involved in the pipeline
import os
import subprocess as sp

# define the environment names
environment_list = [
    'DLC-GPU',
    # 'prey_capture',
    # 'caiman',
    # 'minian',
    # 'vame',
]

# for all the environments
for env in environment_list:
    # assemble the path to the environment file
    out_path = os.path.join(os.getcwd(), env + '_environment.yaml')
    # print(os.getcwd())
    # using a subprocess, activate the environment and create the environment file
    print(out_path)
    # preprocess_sp = sp.Popen([f'conda list & conda env export > "{out_path}"'], stdout=sp.PIPE, shell=True)
    preprocess_sp = sp.Popen([r'%windir%\System32\cmd.exe', '/K', r'C:\ProgramData\Anaconda3\Scripts\activate.bat', r'C:\ProgramData\Anaconda3'], stdout=sp.PIPE, shell=True)

    # preprocess_sp = sp.Popen([r'%windir%\System32\cmd.exe'], stdout=sp.PIPE, shell=True)

    stdout = preprocess_sp.communicate()[0]
    print(stdout.decode())
