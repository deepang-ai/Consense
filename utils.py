import re
import os
import sys
import subprocess
from typing import List, Optional
from crytic_compile import CryticCompile


class Logger(object):
    def __init__(self, file_path='default.log', stream=sys.stdout):
        self.terminal = stream
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.log = open(file_path, 'w')
        self.log = open(file_path, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        pass

def convert_to_int(value) -> int:
    ''' 
    Convert a value to an integer
    defaulting to 0 
    '''
    if value is None:
        return 0
    try:
        safe_value = int(value)
    except ValueError:
        safe_value = 0
    return safe_value

def get_versions_from_file(filePath) -> List[str]:
    '''
    get solc versions from solidity file
    default: [0.4.24, 0.4.5, 0.4.8]
    '''
    versions = []
    f = open(filePath, encoding='utf-8', errors='ignore')
    line = f.readline()
    while line:
        if re.search('pragma', line) != None and re.search('0\.[0-9\.]*', line) != None:
            version = re.search('0\.[0-9\.]*', line).group(0)
            versions.append(version)
        line = f.readline()
    f.close()
    versions.append('0.4.24')
    versions.append('0.4.5')
    versions.append('0.4.8')
    return set(versions)

def switch_solidity_version(solc_version):
    ''' 
    switch solidity version using solc-select 
    '''
    if solc_version in ['0.4.1', '0.4.2', '0.4.3', '0.4.4', '0.4.0']:
        return
    try:
        output = subprocess.check_output(['solc-select', 'versions'], universal_newlines=True)
        if str(solc_version) in output:
            subprocess.run(['solc-select', 'use', solc_version], check=True)
        else:
            print(f'start to install {solc_version}...')
            subprocess.run(['solc-select', 'install', solc_version], check=True)
            subprocess.run(["solc-select", "use", solc_version], check=True)
    except subprocess.CalledProcessError as e:
        print(f'switch version error: {solc_version}')

def check_compile(file_path, solc_versions) -> Optional[str]:
    '''
    try to compile the file with different solc versions
    return the solc version that can compile the file
    '''
    for solc_version in solc_versions:
        try:
            switch_solidity_version(solc_version)
            CryticCompile(file_path)
            return solc_version
        except Exception as e:
            continue
    return None

def find_sol_files(directory):
    '''
    return all sol files in the directory
    including subdirectories
    '''
    sol_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if not file.endswith(".sol"):
                continue
            full_path = os.path.join(root, file)
            sol_files.append(full_path)
    return sol_files

def tokenize_code(string_code):
    '''
    split code into tokens
    hello world my name is 1+3=5(895) ---> 
    ['hello', 'world', 'my', 'name', 'is', '1', '+', '3', '=', '5', '(', '895', ')']
    '''
    text = re.sub(r'([^\w])', r' \1 ', string_code) 
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    tokens = cleaned_text.split()
    return tokens



if  __name__ == "__main__":
    pass