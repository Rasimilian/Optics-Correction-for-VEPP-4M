import numpy as np
import MADX.madx as Madx



class Gauss_Newton():
    def __init__(self,structure_file):
        self.ring = Madx.Structure(structure_file)


    def optimize(self,elem, number, dataX,  withErrors,step, tolerance):
        """
        :param: elem - elements to vary, number - number of elements, dataX - BPMs to observe,
        step - variation step in response calc
        :return: optimized parameters of magnetic elements

        See Gauss-Newton optimization in Wikipedia
        new_parameters - {dK1 or dG, dKick}
        accumulative_param_additive - sum of new_parameters after each iteration
        dP - parameter step to calculate Gessian

        """
        ## Shape for Jacobian
        ## Lines: amount of BPMs * number of functions to observe(x,y,betx,bety)
        ## Rows: amount of elements parameters = grads + kicks + quads alignments(dx,dy)
        ## shape = [4*len(dataX), number]
        dP = 0.00001
        accumulative_param_additive = np.zeros(number)

        if withErrors == True:
            shape = [4*len(dataX), number + 2*self.ring.amount_of_quads]
            accumulative_alignment_additive = np.zeros(2*self.ring.amount_of_quads)
        else:
            shape = [2*len(dataX), number]
            accumulative_alignment_additive = None




        response_matrix = self.ring.response_matrix_calculate(self.ring.bad_structure,self.ring.bad_structure_in_lines,elem,number,dataX, step,areErrorsNeeded=withErrors)

        # To compare response_matrix_calculate and response_matrix_calculate_in_changed_structure
        # model_response_matrix = self.ring.response_matrix_calculate(self.ring.structure,self.ring.structure_in_lines,elem,number,dataX,step)

        model_response_matrix = self.ring.response_matrix_calculate_in_changed_structure(self.ring.structure,self.ring.structure_in_lines,elem,number,dataX, step,accumulative_param_additive,accumulative_alignment_additive,withErrors)


        ## initial errors form gradients and kicks
        _,initial_parameters = self.ring.define_elements_type(self.ring.structure_in_lines)

        ## initial errors from quadrupole alignments
        if self.ring.errors_table != 0:
            initial_alignments = np.stack((self.ring.errors_table.dx,self.ring.errors_table.dy))

        ## initial vector of residuals for each BPM (see Gauss-Newton method)
        initial_vector = np.sum(response_matrix-model_response_matrix,1)

        ## sum of squares of vectors of residuals --> scalar
        initial_residuals = np.sum(initial_vector**2)

        ## same as previous for the loop below
        residuals = np.zeros_like(initial_residuals)

        #print("Initial residuals in Least-Squares: ", initial_residuals)

        check=1
        # while np.isclose(initial_residuals,residuals,atol=tolerance)==False:
        while check <= 1:
            ## Calculation of Jacobian(then Gessian): derivative of residual_vector with respect to parameters
            ## It means the derivative of Response Matrix

            J = np.zeros(shape)
            model_response_matrix1 = self.ring.response_matrix_calculate_in_changed_structure(self.ring.structure,self.ring.structure_in_lines,elem,number,dataX, step,accumulative_param_additive, accumulative_alignment_additive,withErrors)
            vector1 = np.sum(response_matrix-model_response_matrix1,1)
            initial_residuals = np.sum(vector1**2)

            k = 0
            for i in range(number):
                accumulative_param_variation = np.zeros(number)
                accumulative_param_variation[i] = dP

                model_response_matrix2 = self.ring.response_matrix_calculate_in_changed_structure(self.ring.structure,self.ring.structure_in_lines,elem,number,dataX, step,accumulative_param_additive+accumulative_param_variation,accumulative_alignment_additive, areErrorsNeeded=withErrors)
                vector2 = np.sum(response_matrix-model_response_matrix2,1)



                J[:,i] = (vector2-vector1)/dP
                k += 1

            if withErrors == True:
                k1 = k
                for i in range(2*self.ring.amount_of_quads):
                    accumulative_alignment_variation = np.zeros(2*self.ring.amount_of_quads)
                    accumulative_alignment_variation[i] = dP

                    model_response_matrix2 = self.ring.response_matrix_calculate_in_changed_structure(self.ring.structure,self.ring.structure_in_lines,elem,number,dataX, step,accumulative_param_additive,accumulative_alignment_additive+accumulative_alignment_variation, areErrorsNeeded=withErrors)

                    vector2 = np.sum(response_matrix-model_response_matrix2,1)



                    J[:,k1] = (vector2-vector1)/dP


                    k1 += 1


            svd = np.linalg.svd(np.matmul(J.T,J),full_matrices=False)
            u,sv,v = svd[0],svd[1],svd[2]
            sv = np.linalg.inv(np.diag(sv))
            for i in range(len(sv)):
                if sv[i,i] > 1:
                    sv[i,i] = 0


            J_new = np.matmul(np.matmul(v,sv),u.T)
            delta = -np.matmul(np.linalg.pinv(np.matmul(J.T,J)),J.T).dot(vector1)
            print("delta",delta)
            # delta = -np.matmul(np.linalg.pinv(np.matmul(J.T,J)),J.T).dot(vector2)
            print("delta",delta)
            # delta = -np.matmul(np.linalg.pinv(np.matmul(J.T,J)),J.T).dot(vector2-vector1)
            print("delta",delta)
            # delta = -np.matmul(J_new,J.T).dot(vector1)
            accumulative_param_additive += delta[:number]
            accumulative_alignment_additive += delta[number:]


            print("Ideal RM:\n", model_response_matrix)
            print("Real RM:\n", response_matrix)
            print("Model RM2:\n", model_response_matrix2)



            print("Iteration: ", check)
            print("Jacobian:\n", J)
            print("Singular values:\n", svd[1])
            check += 1

        fitted_model_response_matrix = self.ring.response_matrix_calculate_in_changed_structure(self.ring.structure,self.ring.structure_in_lines,elem,number,dataX, step,accumulative_param_additive,accumulative_alignment_additive,withErrors)
        final_vector = np.sum(response_matrix-fitted_model_response_matrix,1)
        residuals = final_residuals = np.sum(final_vector**2)

        print("Initial residuals in Least-Squares: ", initial_residuals)
        print("Final residuals in Least-Squares: ", final_residuals)
        print("Final accumulative_param_additive: ", accumulative_param_additive)
        if withErrors == True:
            print("Final accumulative_alignment_additive: ", accumulative_alignment_additive)
        print("Initial grad parameters: ",initial_parameters)
        if self.ring.errors_table != 0:
            print("Initial align parameters: ",initial_alignments)
        #print("Final parameters: ", initial_parameters + accumulative_param_additive)
        print("Ideal RM:\n", model_response_matrix)
        print("Real RM:\n", response_matrix)
        print("Fitted Model RM:\n", fitted_model_response_matrix)
        print("Singular values:\n", svd[1])

        if withErrors == True:
            return accumulative_param_additive, accumulative_alignment_additive
        else:
            return accumulative_param_additive



