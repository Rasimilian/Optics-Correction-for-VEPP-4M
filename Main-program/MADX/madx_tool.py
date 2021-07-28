import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from cpymad.madx import Madx

from elements_parser.data_parser import read_elements_from_file, describe_elements


class Structure():
    def __init__(self, structure_file="MADX\\VEPP4M_full.txt", bad_structure_file="MADX\\VEPP4M_full_errors.txt"):
        """
        Initialize class Structure.

        :param str structure_file: file name with structure
        :param str bad_structure_file: file name with bad structure
        """
        self.structure = structure_file
        self.structure_in_lines = open(self.structure).readlines()
        self.bad_structure = bad_structure_file
        self.bad_structure_in_lines = open(self.bad_structure).readlines()

        self.BPMs_number = 54
        # self.elements_number = len(describe_elements(self.structure_in_lines)[0])

        self.twiss_table_4D = self.calculate_structure_4D(self.structure)
        self.twiss_table_6D = self.calculate_structure_6D(self.structure)


    def calculate_structure_4D(self, structure, initial_imperfections=None):
        """
        Calculate TWISS table for 4D beam motion.

        :param str structure: file name with structure
        :param initial_imperfections:
        :return: float TWISS table
        """
        madx = Madx(stdout=False)
        madx.option(echo=False, warn=False, info=False, twiss_print=False)
        madx.call(file=structure)
        madx.input('beam,particle=electron,energy=1.8;')
        madx.input('use,sequence=RING;')

        # TODO add errors
        # self.errors_table = self.add_errors_to_structure(madx, 'quadrupole', value=0.000001, initial_imperfections=initial_imperfections)

        # madx.input('select,flag=twiss,class=monitor;')

        madx.twiss(sequence='RING', centre=True, table='twiss', file="MADX\\log_file.txt")
        madx.input('readtable,file="MADX\\log_file.txt",table=twiss_in_BPMs;')

        twiss_table = madx.table.twiss_in_BPMs
        # madx.quit()

        return twiss_table

    def calculate_structure_6D(self, structure, initial_imperfections=None):
        """
        Calculate TWISS table for 6D beam motion.

        :param str structure: file name with structure
        :param initial_imperfections:
        :return: float TWISS table
        """
        madx = Madx(stdout=False)
        madx.option(echo=False, warn=False, info=False, twiss_print=False)
        madx.call(file=structure)
        madx.input('beam,particle=electron,energy=1.8;')
        madx.input('use,sequence=RING;')

        # TODO add errors
        # self.errors_table = self.add_errors_to_structure(madx, 'quadrupole', value=0.000001, initial_imperfections=initial_imperfections)

        # madx.input('select,flag=twiss,class=monitor;')
        madx.input('ptc_create_universe;ptc_create_layout,model=2,method=2,nst=1;')

        madx.ptc_twiss(icase=6,no=1,center_magnets=True,table='twiss', file="MADX\\log_file.txt",)
        madx.input('readtable,file="MADX\\log_file.txt",table=twiss_in_BPMs;')

        twiss_table = madx.table.twiss_in_BPMs
        # madx.quit()

        return twiss_table

    def change_structure(self, structure, structure_in_lines, parameter_number, variation_step, accumulative_param_additive):
        """
        Change elements parameters in structure.

        :param str structure: file name with structure
        :param list structure_in_lines: opened structure in lines
        :param int parameter_number: number of varying parameter
        :param float variation_step: step to vary elements parameters
        :param float accumulative_param_additive: to accumulate parameters changes after iterations
        :return: float TWISS table
        """
        elements_definition, _ = describe_elements(structure_in_lines)
        number_of_elements = len(elements_definition)

        variations_list = np.zeros(number_of_elements)
        variations_list[parameter_number] = variation_step
        accumulative_param_additive += variations_list

        madx = Madx(stdout=False)
        madx.option(echo=False, warn=False, info=False, twiss_print=False)
        madx.call(file=structure)
        madx.input('beam,particle=electron,energy=1.8;')

        for i in range(len(elements_definition)):
            # TODO add for other elements
            if elements_definition[i][2] == 'quadrupole':
                k1 = float(elements_definition[i][8]) + accumulative_param_additive[i]
                madx.command.quadrupole.clone(elements_definition[i][0], L=float(elements_definition[i][5]), k1=k1)
                # print(str(elements_definition[i][0])+ " K1 changed from " + str(float(elements_definition[i][8])) + " to " + str(k1))

        madx.input('use,sequence=RING;')

        madx.input('select,flag=twiss,class=monitor;')
        # madx.input('ptc_create_universe;ptc_create_layout,model=2,method=2,nst=1;')
        #
        # madx.ptc_twiss(icase=6,no=1,center_magnets=True,table='twiss', file="MADX\\log_file.txt",)
        madx.twiss(sequence='RING', centre=True, table='twiss', file="MADX\\log_file.txt")
        madx.input('readtable,file="MADX\\log_file.txt",table=twiss_in_BPMs;')

        twiss_table = madx.table.twiss_in_BPMs
        # madx.quit()

        return twiss_table

    def measure_response(self, structure, element_definition, variation_step, accumulative_param_additive, madx_compiler=None):
        """
        Measure response of a changed element.

        :param str structure: file name with structure
        :param list element_definition: list with elements
        :param float variation_step: step to vary elements parameters
        :param float accumulative_param_additive: to accumulate parameters changes after iterations
        :param object madx_compiler: MAD-X interpreter
        :return: float TWISS table
        """
        madx = Madx(stdout=False)
        madx.option(echo=False, warn=False, info=False, twiss_print=False)
        madx.call(file=structure)
        madx.input('beam,particle=electron,energy=1.8;')

        if madx_compiler != None:
            madx = madx_compiler

        if element_definition[2] == 'quadrupole':
            madx.command.quadrupole.clone(element_definition[0], L=float(element_definition[5]), k1=float(element_definition[8])+variation_step + accumulative_param_additive)
            print(str(element_definition[0]) + " quadrupole response measured")

        # elif elements_definition[2] == 'hkicker' or elements_definition[2] == 'HKICKER':
        #     if len(elements_definition)<=6:
        #         kick = variation_step + accumulative_param_additive
        #         madx.command.hkicker.clone(elements_definition[0], L=float(elements_definition[5]), kick=kick)
        #         print(str(elements_definition[0])+ " hkicker response measured")
        #     else:
        #         kick = float(elements_definition[8]) + variation_step + accumulative_param_additive
        #         madx.command.hkicker.clone(elements_definition[0], L=float(elements_definition[5]), kick=kick)
        #         print(str(elements_definition[0])+ " hkicker response measured")
        #
        # elif elements_definition[2] == 'vkicker' or elements_definition[2] == 'VKICKER':
        #     if len(elements_definition)<=6:
        #         kick = variation_step + accumulative_param_additive
        #         madx.command.vkicker.clone(elements_definition[0], L=float(elements_definition[5]), kick=kick)
        #         print(str(elements_definition[0])+ " vkicker response measured")
        #     else:
        #         kick = float(elements_definition[8]) + variation_step + accumulative_param_additive
        #         madx.command.vkicker.clone(elements_definition[0], L=float(elements_definition[5]), kick=kick)
        #         print(str(elements_definition[0])+ " vkicker response measured")

        else:
            #madx.command.hkicker.clone(elements_definition[0], L=float(elements_definition[5]), kick=variation_step + accumulative_param_additive)
            print("error: measure_response")

        madx.input('use,sequence=RING;')

        madx.input('select,flag=twiss,class=monitor;')
        madx.input('ptc_create_universe;ptc_create_layout,model=1,method=2,nst=1;')

        madx.ptc_twiss(icase=4,no=1,center_magnets=True,table='twiss', file="MADX\\log_file.txt",)
        madx.input('readtable,file="MADX\\log_file.txt",table=twiss_in_BPMs;')

        twiss_table = madx.table.twiss_in_BPMs
        madx.quit()

        return twiss_table

    def calculate_response_matrix(self, structure, structure_in_lines, variation_step, accumulative_param_additive,accumulative_alignment_additive=None,areErrorsNeeded=None):
        """
        Calculate response matrix.

        :param str structure: file name with structure
        :param list structure_in_lines: opened structure in lines
        :param float variation_step: step to vary elements parameters
        :param float accumulative_param_additive: to accumulate parameters changes after iterations
        :param float accumulative_alignment_additive: to accumulate alignment errors changes after iterations
        :param bool areErrorsNeeded: whether to add errors
        :return: float response matrix
        """
        elements_to_vary, _ = describe_elements(structure_in_lines)
        frames = []

        twiss_in_BPMs = self.change_structure(structure, structure_in_lines, 0, 0, accumulative_param_additive)

        for n, element in enumerate(elements_to_vary):
            now = datetime.now()
            ## Jacobian = (f(x+dx)-f(x))/dx
            ## For f(x+dx)
            twiss_in_BPMs_1 = self.change_structure(structure, structure_in_lines, n, variation_step, accumulative_param_additive)

            ## For f(x)
            # twiss_in_BPMs = self.measure_response(self.structure, self.structure_in_lines, element, variation_step[0],
            #                                     accumulative_param_additive[n])
            # print(len(twiss_in_BPMs))
            # print(len(twiss_in_BPMs_1))
            # assert len(twiss_in_BPMs_1) != len(twiss_in_BPMs)

            df_x = pd.DataFrame((twiss_in_BPMs_1.x - twiss_in_BPMs.x)/variation_step, columns = [element[0]])
            df_y = pd.DataFrame((twiss_in_BPMs_1.y - twiss_in_BPMs.y)/variation_step, columns = [element[0]])
            # df_dx = pd.DataFrame((twiss_in_BPMs_1.dx - twiss_in_BPMs.dx)/variation_step, columns = [element[0]])
            # df_dy = pd.DataFrame((twiss_in_BPMs_1.dy - twiss_in_BPMs.dy)/variation_step, columns = [element[0]])

            df = pd.concat([df_x,df_y], ignore_index = True)
            # df = pd.concat([df_x,df_y,df_dx,df_dy], ignore_index = True)
            frames.append(df)
            print(datetime.now()-now)

        matrix = pd.concat(frames, axis=1)
        print("Response Matrix:\n",matrix)
        # matrix.to_csv('MADX//response_matrix_quad_x.txt',index=False,header=False,sep="\t")

        return matrix


class Imperfection(Structure):
    def __init__(self, madx, types_of_errors, file_with_elements_to_spoil):
        """
        Initialize class Imperfection.

        :param object madx: MAD-X interpreter
        :param list types_of_errors: list of errors types
        :param str file_with_elements_to_spoil: file with listed elements to spoil
        """
        self.madx = madx
        self.elements_list, self.elements_number = read_elements_from_file(file_with_elements_to_spoil)

        if not isinstance(types_of_errors,list): self.types_of_errors = list[types_of_errors]

    def add_errors(self, error_amplitude=0.000001, element_type='quadrupole'):
        """
        Add errors to magnetic elements.

        :param float error_amplitude: magnitude of errors
        :param str element_type: type of elements to add errors to
        :return: float table with created errors
        """
        seed = np.random.randint(0,999999999)
        self.madx.input('eoption,seed=' + str(seed) + ',add=True;')

        for element in elements:
            # TODO check the line below
            element = element.strip()
            madx.input('select,flag=error,class=' + element_type + ',pattern=' + element + ';')

        # TODO add types_of_errors choice
        madx.input('ealign,dx:='+str(error_amplitude)+'*gauss(),dy:='+str(error_amplitude)+'*gauss();')
        madx.input('esave,file="MADX\machine_imperfections";')
        madx.input('select,flag=myerrortable, class=quadrupole;')
        madx.input('etable,table=errors_table;')

        errors_table = madx.table.errors_table

        return errors_table








