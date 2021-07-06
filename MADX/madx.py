from cpymad.madx import Madx,Sequence
import pandas as pd
import numpy as np
from scipy.optimize import lsq_linear
import logging
logger = logging.getLogger(__name__)



accumulative_param_additive = np.zeros(1)
class Structure():
    def __init__(self, structure_file):
        self.structure = structure_file
        self.structure_in_lines = open(self.structure).readlines()
        self.table = self.calculate_structure(self.structure,base_imperfections=False)
        self.twiss = self.table.twiss
        self.twiss_short = self.table.twiss_short
        self.summ_table = self.table.summ

        self.bad_structure = 'MADX\VEPP4M_full_errors.txt'
        self.bad_structure_in_lines = open(self.bad_structure).readlines()
        self.bad_table = self.calculate_structure(self.bad_structure,base_imperfections=True)
        self.bad_twiss = self.bad_table.twiss
        self.bad_twiss_short = self.bad_table.twiss_short
        self.bad_summ_table = self.bad_table.summ

        self.errors_table = 0
        self.quads,self.amount_of_quads = self.read_elements('MADX\quads.txt')
        self.correctors,self.amount_of_correctors = self.read_elements('MADX\correctors.txt')




        logger.info("privet")




        #self.model_twiss = self.change_structure1('CTX', 0.07,0)
        #self.model_twiss = self.change_structure1('NIL1', 0.288, 0.93011966)


    def calculate_structure(self, structure, base_imperfections):
        #madx = Madx(stdout=False)
        #madx.option(echo = False, warn = False, info = False, twiss_print = False)
        madx = Madx()
        madx.call(file=structure)
        madx.input('Beam, particle = electron, energy = 1.8;')
        madx.input('use,sequence=RING;')
        self.errors_table = self.add_errors_to_structure(madx,'quadrupole',error=0.000001,base_imperfections=base_imperfections)
        madx.input('select,flag=twiss,class=monitor;')
        twiss = madx.twiss(sequence='RING',centre = True,table='twiss',file="MADX\\measure_123.txt")
        madx.input('readtable, file="MADX\\measure_123.txt",table=twiss_short;')
        table = madx.table
        #arr = np.zeros((len(twiss.s),1))
        #arr[:,0] = np.reshape(twiss.s,(664,1))[:,0]

        return table


    def define_elements_type(self,structure_in_lines):
        #elements, num = self.read_elements('MADX\quads.txt')
        #elements, num = self.read_elements('MADX\correctors.txt')
        elements, num = self.read_elements('MADX\corrs&quads.txt')
        elements_definition = []
        elements_parameters = []
        for element in elements:
            element = element.split()[0]
            for line in structure_in_lines:
                if line.startswith(element):
                    line = line.replace(',','').replace(";",'').split()
                    # index_l = line.index("L")+2
                    # if line[2] == 'quadrupole':
                    #     index_k1 = line.index("K1")+2
                    # else:
                    #     index_k1 = line.index("KICK")+2
                    elements_definition.append(line)
                    elements_parameters.append(float(line[-1]))

        elements_parameters = np.array(elements_parameters)
        print(elements_definition)
        print(elements_parameters)

        return elements_definition,elements_parameters





    # def add_elements(self,elements,element_length, element_strength):
    #     for i in range(0,len(elements)):
    #         madx.command.quadrupole.clone(elements[i],element_length[i],element_strength[i])

    def change_structure_quad(self, elements, length, strength):
        """
        :param elements: list of elements
        :param length: list of elements length
        :param kicks: list of elements strength
        :return: twiss table
        """
        madx = Madx(stdout=False)
        madx.option(echo = False, warn = False, info = False, twiss_print = False)
        madx.call(file=self.structure)
        madx.input('Beam, particle = electron, energy = 1.8;')
        for i in range(elements):
            madx.command.quadrupole.clone(elements[i], L=length[i], k1=strength[i])
        madx.input('use,sequence=RING;')
        madx.input('select,flag=twiss,class=monitor;')
        twiss = madx.twiss(sequence='RING',centre = True,table='twiss',file="MADX\\measure_123.txt")
        madx.input('readtable, file="MADX\\measure_123.txt",table=twiss_short;')
        table = madx.table
        return table

    def change_structure(self, structure,structure_in_lines, variation_step, accumulative_param_additive, accumulative_aglignment_additive,base_imperfections,areErrorsForSimpleSVD,areErrorsForOptimize):
        """

        :param variation_step: changes in parameters
        :return: twiss table
        """
        elements_definition,_ = self.define_elements_type(structure_in_lines)
        print(elements_definition)
        madx = Madx()
        madx.option(echo = False, warn = False, info = False, twiss_print = False)
        madx.call(file=structure)
        madx.input('Beam, particle = electron, energy = 1.8;')
        for i in range(len(elements_definition)):
            if elements_definition[i][2] == 'quadrupole':
                k1 = float(elements_definition[i][8])+variation_step[i]+accumulative_param_additive[i]
                madx.command.quadrupole.clone(elements_definition[i][0], L=float(elements_definition[i][5]), k1 = k1)
                print(str(elements_definition[i][0])+ " K1 changed from " + str(float(elements_definition[i][8])) + " to " + str(k1))

            elif elements_definition[i][2] == 'hkicker' or elements_definition[i][2] == 'HKICKER':
                if len(elements_definition[i])<=6:
                    kick = variation_step[i]+accumulative_param_additive[i]
                    madx.command.hkicker.clone(elements_definition[i][0], L=float(elements_definition[i][5]), kick=kick)
                    print(str(elements_definition[i][0])+" HKICK changed from " + str(0) + " to " + str(kick))
                else:
                    kick = float(elements_definition[i][8])+variation_step[i]+accumulative_param_additive[i]
                    madx.command.hkicker.clone(elements_definition[i][0], L=float(elements_definition[i][5]), kick=kick)
                    print(str(elements_definition[i][0])+" HKICK changed from " + str(elements_definition[i][8]) + " to " + str(kick))

            elif elements_definition[i][2] == 'vkicker' or elements_definition[i][2] == 'VKICKER':
                if len(elements_definition[i])<=6:
                    kick = variation_step[i]+accumulative_param_additive[i]
                    madx.command.vkicker.clone(elements_definition[i][0], L=float(elements_definition[i][5]), kick=kick)
                    print(str(elements_definition[i][0])+ " VKICK changed from " + str(0) + " to " + str(kick))
                else:
                    kick = float(elements_definition[i][8])+variation_step[i]+accumulative_param_additive[i]
                    madx.command.vkicker.clone(elements_definition[i][0], L=float(elements_definition[i][5]), kick=kick)
                    print(str(elements_definition[i][0])+" VKICK changed from " + str(elements_definition[i][8]) + " to " + str(kick))

            else:
                #madx.command.hkicker.clone(elements_definition[i][0], L=float(elements_definition[i][5]), kick=variation_step[i]+accumulative_param_additive[i])
                print("error change_structure")

        madx.input('use,sequence=RING;')
        if base_imperfections == True:
            madx.input('readtable,file="MADX\machine_imperfections",table=errors_table;"')
            madx.input('seterr,table=errors_table;')


        if areErrorsForSimpleSVD == True:
            errors_values = variation_step[len(elements_definition):]
            print("Amount of alignments: ",len(errors_values))
            parameters = ['dx','dy']
            quads, number = self.read_elements('MADX\quads.txt')
            self.change_structure_with_alignment(madx,quads,parameters,errors_values,base_imperfections=True)

        if areErrorsForOptimize == True:
            parameters = ['dx','dy']
            self.change_structure_with_alignment(madx,self.quads,parameters,accumulative_aglignment_additive,base_imperfections=False)


        madx.input('select,flag=twiss,class=monitor;')
        twiss = madx.twiss(sequence='RING',centre = True,table='twiss',file="MADX\\measure_123.txt")
        madx.input('readtable, file="MADX\\measure_123.txt",table=twiss_short;')
        table = madx.table

        return table, len(elements_definition)

    def change_structure_for_response_calculation(self,structure,structure_in_lines,accumulative_param_additive,accumulative_alignment_additive,areErrorsNeeded):
        elements_definition,_ = self.define_elements_type(structure_in_lines)
        print(elements_definition)
        madx = Madx(stdout=False)
        madx.option(echo = False, warn = False, info = False, twiss_print = False)
        madx.call(file=structure)
        madx.input('Beam, particle = electron, energy = 1.8;')
        for i in range(len(elements_definition)):
            if elements_definition[i][2] == 'quadrupole':
                k1 = float(elements_definition[i][8])+accumulative_param_additive[i]
                madx.command.quadrupole.clone(elements_definition[i][0], L=float(elements_definition[i][5]), k1 = k1)
                print(str(elements_definition[i][0])+ " K1 changed from " + str(float(elements_definition[i][8])) + " to " + str(k1))

            elif elements_definition[i][2] == 'hkicker' or elements_definition[i][2] == 'HKICKER':
                if len(elements_definition[i])<=6:
                    kick = accumulative_param_additive[i]
                    madx.command.hkicker.clone(elements_definition[i][0], L=float(elements_definition[i][5]), kick=kick)
                    print(str(elements_definition[i][0])+" HKICK changed from " + str(0) + " to " + str(kick))
                else:
                    kick = float(elements_definition[i][8])+accumulative_param_additive[i]
                    madx.command.hkicker.clone(elements_definition[i][0], L=float(elements_definition[i][5]), kick=kick)
                    print(str(elements_definition[i][0])+" HKICK changed from " + str(elements_definition[i][8]) + " to " + str(kick))

            elif elements_definition[i][2] == 'vkicker' or elements_definition[i][2] == 'VKICKER':
                if len(elements_definition[i])<=6:
                    kick = accumulative_param_additive[i]
                    madx.command.vkicker.clone(elements_definition[i][0], L=float(elements_definition[i][5]), kick=kick)
                    print(str(elements_definition[i][0])+ " VKICK changed from " + str(0) + " to " + str(kick))
                else:
                    kick = float(elements_definition[i][8])+accumulative_param_additive[i]
                    madx.command.vkicker.clone(elements_definition[i][0], L=float(elements_definition[i][5]), kick=kick)
                    print(str(elements_definition[i][0])+" VKICK changed from " + str(elements_definition[i][8]) + " to " + str(kick))

            else:
                #madx.command.hkicker.clone(elements_definition[i][0], L=float(elements_definition[i][5]), kick=variation_step[i]+accumulative_param_additive[i])
                print("error change_structure")


        madx.input('use,sequence=RING;')
        if areErrorsNeeded == True:
            parameters = ['dx','dy']
            self.change_structure_with_alignment(madx,self.quads,parameters,accumulative_alignment_additive,False)



        return madx

    def response_matrix_calculate_in_changed_structure(self,structure,structure_in_lines, elements_to_vary,elements_amount, BPMs_to_observe,variation_step, accumulative_param_additive,accumulative_alignment_additive,areErrorsNeeded):
        elements_to_vary,_ = self.define_elements_type(structure_in_lines)
        variation_step = np.ones(elements_amount)*variation_step
        BPMs_amount = BPMs_to_observe.shape[0]
        matrix = pd.DataFrame()
        frames = []
        # frames_1 = []

        if len(accumulative_param_additive) != len(elements_to_vary):
            print("Accumulative_param_additive length changed")
            accumulative_param_additive = np.zeros(len(elements_to_vary))

        for n, element in enumerate(elements_to_vary):
            ## Jacobian = (f(x+dx)-f(x))/dx
            ## For f(x+dx)
            madx = self.change_structure_for_response_calculation(structure,structure_in_lines,accumulative_param_additive,accumulative_alignment_additive,areErrorsNeeded)
            table = self.measure_response_in_changed_structure(element,accumulative_param_additive[n],variation_step[0],madx)
            twiss_short = table.twiss_short
            summ_table = table.summ


            #
            # ## For f(x)
            madx_1 = self.change_structure_for_response_calculation(structure,structure_in_lines,accumulative_param_additive,accumulative_alignment_additive,areErrorsNeeded)
            table_1 = self.measure_response_in_changed_structure(element,accumulative_param_additive[n],variation_step=0,madx_compiler=madx_1)
            twiss_short_1 = table_1.twiss_short
            summ_table_1 = table_1.summ

            print("proverka",twiss_short.x)
            print("proverka",twiss_short_1.x)

            print("params", accumulative_param_additive)
            # if accumulative_param_additive[n] != 0:
            #     breakpoint()



            # for lattice correction
            #response_matrix[:,num] = (twiss.betx - self.bad_twiss.betx)/variation_step[0]
            # df_betx = pd.DataFrame((twiss_short.betx - self.twiss_short.betx)/variation_step[0], columns = [element[0]])
            # df_qx = pd.DataFrame((summ_table.q1 - self.summ_table.q1)/variation_step[0], columns = [element[0]])
            # df_bety = pd.DataFrame((twiss_short.bety - self.twiss_short.bety)/variation_step[0], columns = [element[0]])
            # df_qy = pd.DataFrame((summ_table.q2 - self.summ_table.q2)/variation_step[0], columns = [element[0]])

            # df_x = pd.DataFrame((twiss_short.x - self.bad_twiss_short.x)/variation_step[0], columns = [element[0]])
            # df_y = pd.DataFrame((twiss_short.y - self.bad_twiss_short.y)/variation_step[0], columns = [element[0]])

            df_x = pd.DataFrame((twiss_short.x - twiss_short_1.x)/variation_step[0], columns = [element[0]])
            df_y = pd.DataFrame((twiss_short.y - twiss_short_1.y)/variation_step[0], columns = [element[0]])

            # df_dx = pd.DataFrame((twiss_short.dx - self.twiss_short.dx)/variation_step[0], columns = [element[0]])
            # df_dy = pd.DataFrame((twiss_short.dy - self.twiss_short.dy)/variation_step[0], columns = [element[0]])

            #df = pd.concat([df_betx,df_bety,df_qx,df_qy],ignore_index = True) # with betas and tunes
            #df = pd.concat([df_betx,df_bety],ignore_index = True) #with betas
            df = pd.concat([df_x,df_y],ignore_index = True) #with betas and orbits
            # df = pd.concat([df_x,df_y],ignore_index = True) #with orbits

            # df_1 = pd.concat([df_dx,df_dy],ignore_index = True) # with dispersion

            # for orbit correction
            #response_matrix[:,num] = (twiss_short.x - self.bad_twiss.x)/variation_step[0]
            # df_x = pd.DataFrame((twiss_short.x - self.bad_twiss_short.x)/variation_step[0], columns = [element[0]])
            # df_y = pd.DataFrame((twiss_short.y - self.bad_twiss_short.y)/variation_step[0], columns = [element[0]])
            # df = pd.concat([df_x,df_y],ignore_index = True)
            #
            # df_dx = pd.DataFrame((twiss_short.dx - self.bad_twiss_short.dx)/variation_step[0], columns = [element[0]])
            # df_dy = pd.DataFrame((twiss_short.dy - self.bad_twiss_short.dy)/variation_step[0], columns = [element[0]])
            # df_1 = pd.concat([df_dx,df_dy],ignore_index = True)


            frames.append(df)
            # frames_1.append(df_1)


        if areErrorsNeeded == True:
            parameters = ['dx','dy']
            for parameter in parameters:
                for quad in self.quads:
                    print("accumulative_alignment_additive", accumulative_alignment_additive)
                    madx = self.change_structure_for_response_calculation(structure,structure_in_lines,accumulative_param_additive,accumulative_alignment_additive,areErrorsNeeded=True)
                    table = self.response_from_alignment(structure,quad,parameter,variation_step[0],False,madx)
                    print("tabloid", table.errors_table.dx,table.errors_table.dy)
                    twiss_short = table.twiss_short
                    summ_table = table.summ

                    df_betx = pd.DataFrame((twiss_short.betx - self.bad_twiss_short.betx)/variation_step[0], columns = [quad +"_"+ parameter])
                    # df_qx = pd.DataFrame((summ_table.q1 - self.summ_table.q1)/variation_step[0], columns = [element[0]])
                    df_bety = pd.DataFrame((twiss_short.bety - self.bad_twiss_short.bety)/variation_step[0], columns = [quad +"_"+ parameter])
                    # df_qy = pd.DataFrame((summ_table.q2 - self.summ_table.q2)/variation_step[0], columns = [element[0]])

                    df_x = pd.DataFrame((twiss_short.x - self.bad_twiss_short.x)/variation_step[0], columns = [quad +"_"+ parameter])
                    df_y = pd.DataFrame((twiss_short.y - self.bad_twiss_short.y)/variation_step[0], columns = [quad +"_"+ parameter])

                    df = pd.concat([df_x,df_y,df_betx,df_bety],ignore_index = True)

                    print(quad+" quadrupole for "+parameter+" response measured")
                    print("check matrix: ",df)
                    frames.append(df)



        matrix = pd.concat(frames, axis = 1)
        # matrix_1 = pd.concat(frames_1, axis = 1)
        print("Response Matrix:\n",matrix)
        # matrix.to_csv('MADX//response_matrix_quad_x.txt',index=False,header=False,sep="\t")
        # matrix_1.to_csv('MADX//response_matrix_quad_dx.txt',index=False,header=False,sep="\t")


        return matrix


    def measure_response_in_changed_structure(self,elements_definition,accumulative_param_additive, variation_step, madx_compiler):
        """

        :param variation_step: changes in parameters
        :return: twiss table
        """
        madx = madx_compiler

        if elements_definition[2] == 'quadrupole':
            madx.command.quadrupole.clone(elements_definition[0], L=float(elements_definition[5]), k1=float(elements_definition[8])+variation_step + accumulative_param_additive)
            print(str(elements_definition[0]) + " quadrupole response measured")

        elif elements_definition[2] == 'hkicker' or elements_definition[2] == 'HKICKER':
            if len(elements_definition)<=6:
                kick = variation_step + accumulative_param_additive
                madx.command.hkicker.clone(elements_definition[0], L=float(elements_definition[5]), kick=kick)
                print(str(elements_definition[0])+ " hkicker response measured")
            else:
                kick = float(elements_definition[8]) + variation_step + accumulative_param_additive
                madx.command.hkicker.clone(elements_definition[0], L=float(elements_definition[5]), kick=kick)
                print(str(elements_definition[0])+ " hkicker response measured")

        elif elements_definition[2] == 'vkicker' or elements_definition[2] == 'VKICKER':
            if len(elements_definition)<=6:
                kick = variation_step + accumulative_param_additive
                madx.command.vkicker.clone(elements_definition[0], L=float(elements_definition[5]), kick=kick)
                print(str(elements_definition[0])+ " vkicker response measured")
            else:
                kick = float(elements_definition[8]) + variation_step + accumulative_param_additive
                madx.command.vkicker.clone(elements_definition[0], L=float(elements_definition[5]), kick=kick)
                print(str(elements_definition[0])+ " vkicker response measured")

        else:
            #madx.command.hkicker.clone(elements_definition[0], L=float(elements_definition[5]), kick=variation_step + accumulative_param_additive)
            print("error: measure_response")

        madx.input('use,sequence=RING;')
        madx.input('select,flag=twiss,class=monitor;')
        twiss = madx.twiss(sequence='RING',centre = True,table='twiss',file="MADX\\measure_123.txt")
        madx.input('readtable, file="MADX\\measure_123.txt",table=twiss_short;')
        table = madx.table

        return table

    def measure_response(self,structure, elements_definition, variation_step, accumulative_param_additive,madx_compiler=None):
        """

        :param variation_step: changes in parameters
        :return: twiss table
        """
        madx = Madx(stdout=False)
        madx.option(echo = False, warn = False, info = False, twiss_print = False)
        madx.call(file=structure)
        madx.input('Beam, particle = electron, energy = 1.8;')
        if madx_compiler != None:
            madx = madx_compiler

        if elements_definition[2] == 'quadrupole':
            madx.command.quadrupole.clone(elements_definition[0], L=float(elements_definition[5]), k1=float(elements_definition[8])+variation_step + accumulative_param_additive)
            print(str(elements_definition[0]) + " quadrupole response measured")

        elif elements_definition[2] == 'hkicker' or elements_definition[2] == 'HKICKER':
            if len(elements_definition)<=6:
                kick = variation_step + accumulative_param_additive
                madx.command.hkicker.clone(elements_definition[0], L=float(elements_definition[5]), kick=kick)
                print(str(elements_definition[0])+ " hkicker response measured")
            else:
                kick = float(elements_definition[8]) + variation_step + accumulative_param_additive
                madx.command.hkicker.clone(elements_definition[0], L=float(elements_definition[5]), kick=kick)
                print(str(elements_definition[0])+ " hkicker response measured")

        elif elements_definition[2] == 'vkicker' or elements_definition[2] == 'VKICKER':
            if len(elements_definition)<=6:
                kick = variation_step + accumulative_param_additive
                madx.command.vkicker.clone(elements_definition[0], L=float(elements_definition[5]), kick=kick)
                print(str(elements_definition[0])+ " vkicker response measured")
            else:
                kick = float(elements_definition[8]) + variation_step + accumulative_param_additive
                madx.command.vkicker.clone(elements_definition[0], L=float(elements_definition[5]), kick=kick)
                print(str(elements_definition[0])+ " vkicker response measured")

        else:
            #madx.command.hkicker.clone(elements_definition[0], L=float(elements_definition[5]), kick=variation_step + accumulative_param_additive)
            print("error: measure_response")

        madx.input('use,sequence=RING;')
        madx.input('select,flag=twiss,class=monitor;')
        twiss = madx.twiss(sequence='RING',centre = True,table='twiss',file="MADX\\measure_123.txt")
        madx.input('readtable, file="MADX\\measure_123.txt",table=twiss_short;')
        table = madx.table
        return table




    def read_elements(self,elements_file):
        elements = open(elements_file,'r')
        element = elements.readlines()
        num = 0
        elements=[]
        for elem in element:
            num+=1
            elements.append(elem.strip())
        return elements, num

    def read_BPMs(self, BPMs_data):
        data = pd.read_csv(BPMs_data, sep = "\t")
        data_size = data.shape

        # For measure1.txt without s coordinate
        #data = pd.read_csv(BPMs_data)
        #data = data.iloc[7:data.shape[0],0].astype(float)

        dataX = data.iloc[:,1:3].to_numpy()
        dataY = data.iloc[:,1:4:2].to_numpy()
        return dataX, dataY

    #TODO Dobavit' drugie funcsii y,bet,elements to var yispravit
    def response_matrix_calculate(self,structure,structure_in_lines,elements_to_vary,elements_amount, BPMs_to_observe,variation_step,areErrorsNeeded,madx_compiler=None):
        global accumulative_param_additive
        elements_to_vary,_ = self.define_elements_type(structure_in_lines)
        variation_step = np.ones(elements_amount)*variation_step
        BPMs_amount = BPMs_to_observe.shape[0]
        response_matrix = np.zeros((BPMs_amount,elements_amount))
        matrix = pd.DataFrame()
        frames = []
        frames_1 = []
        num = 0
        if len(accumulative_param_additive) != len(elements_to_vary):
            print("Accumulative_param_additive length changed")
            accumulative_param_additive = np.zeros(len(elements_to_vary))

        for n, element in enumerate(elements_to_vary):
            table = self.measure_response(structure,element,variation_step[0], accumulative_param_additive[n],madx_compiler)
            twiss_short = table.twiss_short
            summ_table = table.summ

            # for lattice correction
            #response_matrix[:,num] = (twiss.betx - self.bad_twiss.betx)/variation_step[0]
            # df_betx = pd.DataFrame((twiss_short.betx - self.bad_twiss_short.betx)/variation_step[0], columns = [element[0]])
            # df_qx = pd.DataFrame((summ_table.q1 - self.bad_summ_table.q1)/variation_step[0], columns = [element[0]])
            # df_bety = pd.DataFrame((twiss_short.bety - self.bad_twiss_short.bety)/variation_step[0], columns = [element[0]])
            # df_qy = pd.DataFrame((summ_table.q2 - self.bad_summ_table.q2)/variation_step[0], columns = [element[0]])

            df_x = pd.DataFrame((twiss_short.x - self.bad_twiss_short.x)/variation_step[0], columns = [element[0]])
            df_y = pd.DataFrame((twiss_short.y - self.bad_twiss_short.y)/variation_step[0], columns = [element[0]])

            # df_dx = pd.DataFrame((twiss_short.dx - self.bad_twiss_short.dx)/variation_step[0], columns = [element[0]])
            # df_dy = pd.DataFrame((twiss_short.dy - self.bad_twiss_short.dy)/variation_step[0], columns = [element[0]])

            #df = pd.concat([df_betx,df_bety,df_qx,df_qy],ignore_index = True) # with betas and tunes
            #df = pd.concat([df_betx,df_bety],ignore_index = True) #with betas
            df = pd.concat([df_x,df_y],ignore_index = True) #with betas and orbits
            # df = pd.concat([df_x,df_y],ignore_index = True) #with orbits

            # df_1 = pd.concat([df_dx,df_dy],ignore_index = True) # with dispersion

            # for orbit correction
            #response_matrix[:,num] = (twiss_short.x - self.bad_twiss.x)/variation_step[0]
            # df_x = pd.DataFrame((twiss_short.x - self.bad_twiss_short.x)/variation_step[0], columns = [element[0]])
            # df_y = pd.DataFrame((twiss_short.y - self.bad_twiss_short.y)/variation_step[0], columns = [element[0]])
            # df = pd.concat([df_x,df_y],ignore_index = True)
            #
            # df_dx = pd.DataFrame((twiss_short.dx - self.bad_twiss_short.dx)/variation_step[0], columns = [element[0]])
            # df_dy = pd.DataFrame((twiss_short.dy - self.bad_twiss_short.dy)/variation_step[0], columns = [element[0]])
            # df_1 = pd.concat([df_dx,df_dy],ignore_index = True)


            frames.append(df)
            # frames_1.append(df_1)



        if areErrorsNeeded == True:
            quads, number = self.read_elements('MADX\quads.txt')
            parameters = ['dx','dy']
            for parameter in parameters:
                for quad in quads:
                    quad = quad.strip()
                    variation_step = variation_step[0]
                    table = self.response_from_alignment(self.bad_structure,quad,parameter,variation_step,base_imperfections=True)
                    twiss_short = table.twiss_short
                    summ_table = table.summ

                    df_betx = pd.DataFrame((twiss_short.betx - self.bad_twiss_short.betx)/variation_step, columns = [quad +"_"+ parameter])
                    # df_qx = pd.DataFrame((summ_table.q1 - self.bad_summ_table.q1)/variation_step[0], columns = [element[0]])
                    df_bety = pd.DataFrame((twiss_short.bety - self.bad_twiss_short.bety)/variation_step, columns = [quad +"_"+ parameter])
                    # df_qy = pd.DataFrame((summ_table.q2 - self.bad_summ_table.q2)/variation_step[0], columns = [element[0]])

                    df_x = pd.DataFrame((twiss_short.x - self.bad_twiss_short.x)/variation_step, columns = [quad +"_"+ parameter])
                    df_y = pd.DataFrame((twiss_short.y - self.bad_twiss_short.y)/variation_step, columns = [quad +"_"+ parameter])

                    df = pd.concat([df_x,df_y,df_betx,df_bety],ignore_index = True)

                    print(quad+" quadrupole for "+parameter+" response measured")
                    frames.append(df)


        matrix = pd.concat(frames, axis = 1)
        # matrix_1 = pd.concat(frames_1, axis = 1)
        print("Response Matrix:\n",matrix)
        matrix.to_csv('MADX//response_matrix_test.txt',index=False,header=True,sep="\t")
        # matrix_1.to_csv('MADX//response_matrix_quad_dx.txt',index=False,header=False,sep="\t")

        #return response_matrix
        return matrix

    def correct_lattice(self,response_matrix,inverted_response_matrix,model_structure,real_structure, scale_factor = 1):
        global accumulative_param_additive
        print("accumulative_param_additive", accumulative_param_additive)
        delta = [(model_structure[i] - real_structure[i]) for i in range(len(model_structure))]
        delta = np.array(delta)*scale_factor
        ##TODO proverit' delta. pochemu na krayah bol'shie otkloneniya
        #delta = [-0.5*real_structure[i] for i in range(len(model_structure))]

        #inverse_matrix=np.linalg.pinv(response_matrix)
        print("Inverted response matrix shape:", inverted_response_matrix.shape)
        print(len(delta))
        new_parameters = lsq_linear(response_matrix,delta).x
        print('new parameters without sv choosing:',new_parameters)
        new_parameters = np.matmul(inverted_response_matrix,delta.T)
        print('new parameters:', new_parameters)
        table, amount_of_fields_parameters = self.change_structure(self.bad_structure,self.bad_structure_in_lines,new_parameters, accumulative_param_additive, accumulative_aglignment_additive=0,base_imperfections=True,areErrorsForSimpleSVD=False, areErrorsForOptimize=False)
        accumulative_param_additive += new_parameters[0:amount_of_fields_parameters]
        print("accumulative_param_additive", accumulative_param_additive)

        return table


    def invert_response_matrix(self, response_matrix):
        svd = np.linalg.svd(response_matrix,full_matrices=False)
        u_matrix = svd[0]
        sv_matrix = svd[1]
        v_matrix = svd[2].T
        return u_matrix, sv_matrix, v_matrix

    def make_iteration(self, scale_factor):
        pass

    def add_errors_to_structure(self, madx_compiler, element_type,error,base_imperfections):
        if base_imperfections == True:
            quads, number = self.read_elements('MADX\quads.txt')
            seed = np.random.randint(0,999999999)
            madx_compiler.input('eoption,seed='+str(seed)+',add=True;')

            for quad in quads:
                quad = quad.strip()
                madx_compiler.input('select,flag=error,class='+element_type+',pattern='+quad+';')
            madx_compiler.input('ealign,dx:='+str(error)+'*gauss(),dy:='+str(error)+'*gauss();')



            #madx_compiler.input('efcomp,dkn:={0,'+str(error)+'*ranf()};')

            madx_compiler.input('esave,file="MADX\machine_imperfections";')
            madx_compiler.input('select,flag=myerrortable, class=quadrupole;')
            madx_compiler.input('etable,table=errors_table;')
            #madx_compiler.input('esave,table=errors_table;')

            table = madx_compiler.table.errors_table
            #print(table.dx)


            return table

        else:
            pass

    def response_from_alignment(self, structure, element, parameter,variation_step, base_imperfections,madx_compiler=None):
        madx = Madx(stdout=False)
        madx.option(echo = False, warn = False, info = False, twiss_print = False)
        madx.call(file=structure)
        madx.input('Beam, particle = electron, energy = 1.8;')
        madx.input('use,sequence=RING;')
        if base_imperfections == True:
            madx.input('eoption,add=True;')
            madx.input('readtable,file="MADX\machine_imperfections",table=errors_table;"')
            madx.input('seterr,table=errors_table;')

        ## For Gauss-Newton optimization
        if madx_compiler != None:
            madx = madx_compiler
            madx.input('eoption,add=True;')

        madx.input('select,flag=error,clear;')
        madx.input('select,flag=error,pattern='+element+';')
        madx.input('ealign,'+parameter+'='+str(variation_step)+';')

        ## Check
        madx.input('select,flag=error,clear;')
        for quad in self.quads:
            madx.input('select,flag=error,class=quadrupole,pattern='+quad+';')

        madx.input('select,flag=myerrortable, class=quadrupole;')
        madx.input('etable,table=errors_table;')
        table = madx.table.errors_table
        # errors = np.stack((table.betx,table.bety))
        print("New alignments + response: ",table.dx, table.dy)
        print("Parameter value + response: ", element, parameter,variation_step)

        ##Check

        madx.input('select,flag=twiss,class=monitor;')
        twiss = madx.twiss(sequence='RING',centre = True,table='twiss',file="MADX\\measure_123.txt")
        madx.input('readtable, file="MADX\\measure_123.txt",table=twiss_short;')
        table = madx.table
        #print(table.twiss.betx)

        return table


    def change_structure_with_alignment(self, madx, elements, parameters,parameter_value, base_imperfections):
        if base_imperfections == True:
            madx.input('readtable,file="MADX\machine_imperfections",table=errors_table;"')
            madx.input('seterr,table=errors_table;')

        madx.input('eoption,add=True;')


        k = 0
        for parameter in parameters:
            for i in range(len(elements)):
                madx.input('select,flag=error,clear;')
                madx.input('select,flag=error,class=quadrupole,pattern='+elements[i]+';')
                madx.input('ealign,'+parameter+'='+str(parameter_value[k])+';')
                k += 1

        madx.input('select,flag=error,clear;')
        for i in range(len(elements)):
            madx.input('select,flag=error,class=quadrupole,pattern='+elements[i]+';')

        #madx.input('select,flag=error,class=quadrupole;')
        madx.input('select,flag=myerrortable, class=quadrupole;')
        madx.input('etable,table=errors_table;')
        table = madx.table.errors_table
        # errors = np.stack((table.betx,table.bety))
        print("New alignments: ",table.dx, table.dy)
        print("Parameter value: ", parameter_value)














