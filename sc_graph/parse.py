import os
import sys
from slither import Slither
from typing import List, Optional
from slither.core.declarations.function_contract import FunctionContract as SlitherFunction

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
from sc_graph.declaration import *


# region analyze
def analyze_call(my_function:Function, function:SlitherFunction):
    '''
    add call information to the function
    '''
    # state variable read
    for variable in function.state_variables_read:
        my_function.add_sv_read(variable.canonical_name)
    # state variable write
    for variable in function.state_variables_written:
        my_function.add_sv_write(variable.canonical_name)
    # inter function call
    for inter_call in function.internal_calls:
        if not hasattr(inter_call, 'canonical_name'):
            continue
        my_function.add_call_function(inter_call.canonical_name)
    # external function call
    for v in function.high_level_calls:
        if type(v[1]).__name__ == 'FunctionContract':   # function
            my_function.add_call_function(v[1].canonical_name)
        else:   # variable
            my_function.add_sv_read(v[1].canonical_name)
    return

def analyze_contract(file_path, solc_version) -> List[Contract]:
    '''
    analyze a solidity file and returns a list of Contract objects
    Arguments:
        file_path: path to the solidity file
        solc_version: version of solc to use
    '''
    filter_function_name = ['slitherConstructorConstantVariables', 'slitherConstructorVariables']
    switch_solidity_version(solc_version)
    slither = Slither(file_path)
    my_contracts = []
    for contract in slither.contracts:
        my_contract = Contract(contract.name, contract.contract_kind)
        # declared variables
        for variable in contract.state_variables_declared:
            my_variable = Variable(variable.canonical_name, True, variable.source_mapping.content)
            my_contract.add_variable(my_variable)
        # inherited variables
        for variable in contract.state_variables_inherited:
            my_variable = Variable(variable.canonical_name, False, variable.source_mapping.content)
            my_contract.add_variable(my_variable)
        # declared functions
        for function in contract.functions_declared:
            if function.name in filter_function_name:
                continue
            my_function = Function(function.canonical_name, True, False, function.source_mapping.content)
            analyze_call(my_function, function)
            my_contract.add_function_declared(my_function)
            my_contract.add_function(my_function)
        # inherited functions
        for function in contract.functions_inherited:
            my_function = Function(function.canonical_name, False, False, function.source_mapping.content)
            my_contract.add_function(my_function)
        # declared modifiers
        for modifier in contract.modifiers_declared:
            my_modifier = Function(modifier.canonical_name, True, True, modifier.source_mapping.content)
            analyze_call(my_modifier, modifier)
            my_contract.add_function_declared(my_modifier)
            my_contract.add_function(my_modifier)
        # inherited modifiers
        for modifier in contract.modifiers_inherited:
            my_modifier = Function(modifier.canonical_name, False, True, modifier.source_mapping.content)
            my_contract.add_function(my_modifier)
        my_contracts.append(my_contract)
    return my_contracts

def test_analyze_contract(file_path, solc_version):
    contracts = analyze_contract(file_path, solc_version)
    for contract in contracts:
        print(f'*********{contract.name} {contract.type}***********')
        for variable in contract.variables:
            if not variable.declared:
                continue
            print('=============')
            print(f'{variable.canonical_name}')
            print('-----------')
            print(variable.content)
        for function in contract.functions:
            if not function.declared:
                continue
            print('=============')
            print(f'{function.canonical_name}, is_modifier: {function.is_modifier}')
            print('-----------')
            print(function.content)

def test_analyze_call(file_path, solc_version):
    contracts = analyze_contract(file_path, solc_version)
    for contract in contracts:
        print(f'*********{contract.name} {contract.type}***********')
        for function in contract.functions_declared:
            if not function.declared:
                continue
            print('=============')
            print(f'{function.canonical_name}, is_modifier: {function.is_modifier}')
            print(f'sv read: {function.sv_read}')
            print(f'sv write: {function.sv_write}')
            print(f'call functions: {function.call_functions}')
            print('------------------')
            # print(function.content)

# end region

#region solidity_parser
class SolidityParser:
    def __init__(self, file_path:str, solc_version:str):
        self._file_path = file_path
        self._solc_version = solc_version
        self._contracts = analyze_contract(file_path, solc_version)

    @property
    def contracts(self) -> List[Contract]:
        return self._contracts

    def get_contract(self, contract_name:str) -> Optional[Contract]:
        for contract in self._contracts:
            if contract.name == contract_name:
                return contract
        return None
    
    def get_state_variable(self, canonical_name:str) -> Optional[Variable]:
        contract_name = canonical_name.split('.')[0]
        contract = self.get_contract(contract_name)
        if contract is None:
            return None
        for variable in contract.variables:
            if variable.canonical_name == canonical_name:
                return variable
        return None
    
    def get_function(self, canonical_name:str) -> Optional[Function]:
        contract_name = canonical_name.split('.')[0]
        contract = self.get_contract(contract_name)
        if contract is None:
            return None
        for function in contract.functions:
            if function.canonical_name == canonical_name:
                return function
        return None
    
    def get_node(self, node_name:str):
        if node_name.endswith(')'):
            return self.get_function(node_name)
        else:
            return self.get_state_variable(node_name)


#region contract_graph
def contract_graph(solidity_parser:SolidityParser, contract_name:str):
    '''
    Obtain graph structure based on contracts, 
    where nodes represent functions、modifiers and State variables. 
    Edge represents the calling relationship between nodes
    Arguments:
        solidity_parser: SolidityParser object
        contract_name: name of the contract
    Returns:
        nodes: list of node names
        edges: list of edge tuples
    '''
    contract = solidity_parser.get_contract(contract_name)
    if contract is None:
        print(f'Contract {contract_name} not found')
        return None
    nodes = []
    edges = []
    # add variable nodes
    for variable in contract.variables:
        variable_node = solidity_parser.get_state_variable(variable.canonical_name)
        if variable_node is None:
            print(f'Variable {variable.canonical_name} not found')
            continue
        nodes.append(variable_node.canonical_name)
    # add function or modifier nodes
    for function in contract.functions:
        function_node = solidity_parser.get_function(function.canonical_name)
        if function_node is None:
            print(f'Function {function.canonical_name} not found')
            continue
        nodes.append(function_node.canonical_name)
    # add edges
    for function in contract.functions_declared:
        # state variable read
        for svr in function.sv_read:
            variable_node = solidity_parser.get_state_variable(svr)
            if variable_node is None:
                print(f'Variable {svr} not found')
                continue
            # supple node
            if variable_node.canonical_name not in nodes:
                nodes.append(variable_node.canonical_name)
            # add edge
            src_id = nodes.index(variable_node.canonical_name)
            dst_id = nodes.index(function.canonical_name)
            edges.append((src_id, dst_id))
        for svw in function.sv_write:
            variable_node = solidity_parser.get_state_variable(svw)
            if variable_node is None:
                print(f'Variable {svw} not found')
                continue
            # supple node
            if variable_node.canonical_name not in nodes:
                nodes.append(variable_node.canonical_name)
            # add edge
            src_id = nodes.index(function.canonical_name)
            dst_id = nodes.index(variable_node.canonical_name)
            edges.append((src_id, dst_id))
        for func in function.call_functions:
            func_node = solidity_parser.get_function(func)
            if func_node == None:
                print(f'Function {func} not exist!')
                continue
            # supple node
            if func_node.canonical_name not in nodes:
                nodes.append(func_node.canonical_name)
            # add edge
            src_id = nodes.index(function.canonical_name)
            dst_id = nodes.index(func_node.canonical_name)
            edges.append((src_id, dst_id))
    return nodes, edges

    
def test_contract_graph(file_path:str, contract_name:str, solc_version:str):
    solidity_parser = SolidityParser(file_path, solc_version)
    res = contract_graph(solidity_parser, contract_name)
    if res is None:
        return None
    print(res[0])
    print(res[1])


#region solidity_graph
def supplement_source(function:Function, solidity_parser:SolidityParser):
    '''
    supplement source code for function
    including called state variables and modifiers
    '''
    node_text = ''
    for svr in function.sv_read:
        variable = solidity_parser.get_state_variable(svr)
        if not variable:
            print(f'Variable {svr} not found')
            continue
        node_text += f'{variable.content}\n'
    for svw in function.sv_write:
        variable = solidity_parser.get_state_variable(svw)
        if not variable:
            print(f'Variable {svw} not found')
            continue
        node_text += f'{variable.content}\n'
    for func in function.call_functions:
        dst_function = solidity_parser.get_function(func)
        if not dst_function:
            print(f'Function {func} not found')
            continue
        if not dst_function.is_modifier:
            continue
        node_text += f'{dst_function.content}\n'
    node_text += f'{function.content}\n'
    return node_text


def solidity_graph(solidity_parser:SolidityParser):
    nodes = []
    edges = []
    source_list = []    # source code
    # add nodes
    for contract in solidity_parser.contracts:
        if contract.type != 'contract':
            continue
        # only function and modifier as nodes
        for function in contract.functions:
            if function.canonical_name in nodes:
                continue
            if function.is_modifier:
                continue
            nodes.append(function.canonical_name)
            source_list.append(supplement_source(function, solidity_parser))
    # add edges
    for contract in solidity_parser.contracts:
        if contract.type != 'contract':
            continue
        for function in contract.functions_declared:
            if function.is_modifier:
                continue
            for func in function.call_functions:
                dst_function = solidity_parser.get_function(func)
                if not dst_function:
                    print(f'Function {func} not found')
                    continue
                if dst_function.is_modifier:
                    continue
                if func not in nodes:
                    nodes.append(func)
                    source_list.append(supplement_source(dst_function, solidity_parser))
                src_id = nodes.index(function.canonical_name)
                dst_id = nodes.index(func)
                edges.append((src_id, dst_id))
    return nodes, edges, source_list


def test_solidity_graph(file_path, solc_version):
    solidity_parser = SolidityParser(file_path= file_path,solc_version=solc_version)
    nodes, edges, sources = solidity_graph(solidity_parser)
    for i in range(len(nodes)):
        print(nodes[i])
        print('----------')
        print(sources[i])
        print('==========================')
    print(f'len(nodes): {len(nodes)}, len(edges): {len(edges)}, len(sources): {len(sources)}')


if __name__ == '__main__':
    # sys.stdout = Logger('log.txt', sys.stdout)
    file_path = "../datasets/scvhunter/source/reentrancy/dependency/0x0000000000b3f879cb30fe243b4dfee438691c04.sol"
    solc_version = '0.4.24'
    contract_name = 'GasToken2'
    # test_analyze_contract(file_path, solc_version)
    # test_analyze_call(file_path, solc_version)
    # test_contract_graph(file_path, contract_name, solc_version)
    test_solidity_graph(file_path, solc_version)
    pass
