import torch as tn
import torchtt as tntt


class functionTT():

    def __init__(self, basis, dofs = None,  variable_names = []) -> None:
        
        self.__d = len(basis)
        self.__basis = basis.copy()
        self.__variable_names = variable_names.copy()
        self.__tt = dofs.clone()

