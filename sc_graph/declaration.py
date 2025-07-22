from typing import List


# region Variable
###################
class Variable:
    def __init__(self, canonical_name:str, declared:bool, content:str):
        '''
        Arguments:
            canonical_name: {contract.name}.{variable.name}
            declared: whether the variable is declared by this contract
            content: the source code of the variable
        '''
        self._canonical_name = canonical_name
        self._declared = declared
        self._content = content

    @property
    def canonical_name(self) -> str:
        '''
        str: {contract.name}.{variable.name}
        '''
        return self._canonical_name
    

    @property
    def declared(self) -> bool:
        '''returns whether the variable is declared by this contract'''
        return self._declared


    @property
    def content(self) -> str:
        '''returns the source code of the variable'''
        return self._content
# endregion

# region Function
class Function:
    def __init__(self, canonical_name:str, declared:bool, is_modifier:bool, content:str):
        '''
        Arguments:
            canonical_name: {contract.name}.{func_name(type1,type2)}
            declared: whether the function is declared by this contract
            is_modifier: whether the function is a modifier
            content: the source code of the function
        '''
        self._canonical_name = canonical_name
        self._declared = declared
        self._is_modifier = is_modifier
        self._content = content

        self._sv_read = []      # state variable read
        self._sv_write = []     #  state variable write
        self._call_functions = []  # function or modifier call


    @property
    def canonical_name(self) -> str:
        '''
        str: {contract.name}.{func_name(type1,type2)}
        '''
        return self._canonical_name
    
    @property
    def declared(self) -> bool:
        '''returns whether the function is declared by this contract'''
        return self._declared
    
    @property
    def is_modifier(self) -> bool:
        '''returns whether the function is a modifier'''
        return self._is_modifier

    @property
    def content(self) -> str:
        '''returns the source code of the function'''
        return self._content
    
    @property
    def sv_read(self) -> List[str]:
        '''
        returns the state variable read by the function
        each variable is represented by its full name
        '''
        return self._sv_read
    
    @property
    def sv_write(self) -> List[str]:
        '''
        returns the state variable write by the function
        each variable is represented by its full name
        '''
        return self._sv_write
    
    @property
    def call_functions(self) -> List[str]:
        '''
        returns the functions or modifiers called by the function
        each function or modifier is represented by its canonical name
        '''
        return self._call_functions
    
    def add_sv_read(self, v_full_name:str):
        '''adds a state variable read to the function'''
        self._sv_read.append(v_full_name)
    
    def add_sv_write(self, v_full_name:str):
        '''adds a state variable write to the function'''
        self._sv_write.append(v_full_name)

    def add_call_function(self, function_name:str):
        '''adds a function or modifier called by the function'''
        self._call_functions.append(function_name)
# endregion


# region Contract
class Contract:
    def __init__(self, name:str, type:str):
        self._name = name
        self._type = type          # contract, library, interface
        self._functions = []    # including both declared and inherited functions and modifiers
        self._variables = []
        self._functions_declared = []   # only functions or modifiers declared by this contract

    @property
    def name(self) -> str:
        '''returns the name of the contract'''
        return self._name
    
    @property
    def type(self) -> str:
        '''returns the type of the contract'''
        return self._type
    
    @property
    def variables(self) -> List[Variable]:
        '''returns the variables of the contract'''
        return self._variables

    @property
    def functions(self) -> List[Function]:
        '''returns the all functions and modifers of the contract'''
        return self._functions
    
    @property
    def functions_declared(self) -> List[Function]:
        '''returns the functions declared by this contract'''
        return self._functions_declared

    def add_function(self, function:Function):
        '''adds a function or modifier to the contract'''
        self._functions.append(function)

    def add_variable(self, variable:Variable):
        '''adds a variable to the contract'''
        self._variables.append(variable)

    def add_function_declared(self, function:Function):
        '''adds a function or modifier declared by this contract to the contract'''
        self._functions_declared.append(function)
# endregion