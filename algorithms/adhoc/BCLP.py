import numpy as np
import time
import logging
import msvcrt
#from scipy import optimize
import warnings
import matplotlib.pyplot as plt
import vamtoolbox

class LogPerf:
    #Object to store performance metric values and options.
    def __init__(self,options):
        self.options = options
        self.curr_iter = 0
        self.loss = np.zeros(options.n_iter+1)*np.nan #including the init
        
        self.l0 = np.zeros(options.n_iter+1)*np.nan #L0 norm evaluated with equal weights, q=1, eps as in optimization
        self.l1 = np.zeros(options.n_iter+1)*np.nan #L1 norm evaluated with equal weights, q=1, eps as in optimization
        self.l2 = np.zeros(options.n_iter+1)*np.nan #L2 norm evaluated with equal weights, q=1, eps as in optimization
        self.lf = np.zeros(options.n_iter+1)*np.nan #Linf norm evaluated with equal weights, q=1, eps as in optimization

        self.l0_eps0 = np.zeros(options.n_iter+1)*np.nan #L0 norm evaluated with equal weights, q=1, eps=0
        self.l1_eps0 = np.zeros(options.n_iter+1)*np.nan #L1 norm evaluated with equal weights, q=1, eps=0
        self.l2_eps0 = np.zeros(options.n_iter+1)*np.nan #L2 norm evaluated with equal weights, q=1, eps=0
        self.lf_eps0 = np.zeros(options.n_iter+1)*np.nan #Linf norm evaluated with equal weights, q=1, eps=0
        
        self.ver = np.zeros(options.n_iter+1)*np.nan

        self.iter_times = np.zeros(options.n_iter+1)
        self.t0 = np.nan
        self.is_converged = False

    # [UNUSED] Not called in this project: algorithms/adhoc/run.py saves the loss history itself (loss_history.npz).
    def save(self,filename):
        np.savetxt(filename,
                   np.column_stack([range(self.options.n_iter+1),self.iter_times,self.loss,self.l0,self.l1,self.l2,self.lf,self.l0_eps0,self.l1_eps0,self.l2_eps0,self.lf_eps0]),
                   delimiter=",",
                   header="n,compute time,loss,L0 norm,L1 norm,L2 norm,Linf norm,L0 norm eps=0,L1 norm eps=0,L2 norm eps=0,Linf norm eps=0,",
                   fmt=["%5.0i","%10.2f","%10.4f","%10.4f","%10.4f","%10.4f","%10.4f","%10.4f","%10.4f","%10.4f","%10.4f"],
        )

    def startTiming(self):
        self.t0 = time.perf_counter()

    def recordIterTime(self):
        self.iter_times[self.curr_iter] = time.perf_counter() - self.t0

    def show(self, normalation_flags = (True, True, True, True)):
        #Plot cost function and all other metrics
        fig, axs = plt.subplots(1,3) #Create figure
        fig.set_size_inches(12, 8) #Resize figure
        error_plot = vamtoolbox.displaygrayscale.MultiErrorPlot(self.loss,ax=axs[0],fig=fig, title= 'Weighted $L_{p}$ norm cost function', xlabel= 'Iteration', ylabel='Cost')
        error_lp_plot = vamtoolbox.displaygrayscale.MultiErrorPlot(
            self.l0,
			self.l1,
			self.l2,
			self.lf,
			ax=axs[1],
			fig=fig,
			title= r'Other $L_{p}$ norms with original tolerance $\varepsilon$',
			xlabel= 'Iteration',
			ylabel='Normalized norm value',
			legends = ['$L_{0}$','$L_{1}$','$L_{2}$','$L_{\infty}$'],
			normalization_flags= normalation_flags
			)

        error_lp_eps0_plot = vamtoolbox.displaygrayscale.MultiErrorPlot(
            self.l0_eps0,
			self.l1_eps0,
			self.l2_eps0,
			self.lf_eps0,
			ax=axs[2],
			fig=fig,
			title= r'Other $L_{p}$ norms with zero tolerance $\varepsilon = 0$',
			xlabel= 'Iteration',
			ylabel='Normalized norm value',
			legends = ['$L_{0}$','$L_{1}$','$L_{2}$','$L_{\infty}$'],
			normalization_flags= normalation_flags
			)
        
        plt.show()
        return fig, axs, error_plot, error_lp_plot, error_lp_eps0_plot

class BCLPNorm:
    """
    This class provide loss function and gradient methods to external optimizer.
    It also temporarily stores the state of the variables (including error) implicitly during the optimization.
    
    Naming convention:
    Ax=b in algebraic reconstruction corresponds to Pf=g in this code and in the paper.

    Developement notes: 
    This optimization method is written to be as propagator-agnostic as possible.
    (1) Sinogram ndarray returned by propagator is not limited to be in a particular shape.
        The original shape is denoted 'cylindrical' in code. The array is flattened for some operations.
    (2) Dose unit is controlled jointly by propagator and the material response model.
        This code only assumes the propagtor generates output that can be directly feeds to the response model in order to predict response.
        The areal dose or volumetric dose in real space computed by propagator is simply denoted as dose in code.
        The material response is denoted as mapped dose.
    """
    
    def __init__(self, target_geo, proj_geo, options, projector=None):
        self.logger = logging.getLogger(__name__)

        # [W/UW NOTE] Terms from the R2R paper: W (wrapped) is the photoresist curved around the cylindrical roller, in ordinary x-y coordinates about the rotation axis.
        # [W/UW NOTE] UW (unwrapped) is the same photoresist laid flat as a web, in rho-l coordinates (rho = thickness, l = length along the web).
        # [W/UW NOTE] INPUT: target_geo is in whatever domain the projector expects. With the default vamtoolbox projector (projector=None) it is a W-domain / conventional-CAL x-y(-z) grid centered on the rotation axis.
        # [W/UW NOTE] In this project, propagator.minimizeR2RBCLP passes projector = R2RAlgebraicPropagator (the composite P_UW->g, the ray-traced matrix of outputs/matrices/), so target_geo, dose and mapped_dose are all in the UW domain.
        # [W/UW NOTE] This file itself never converts UW <-> W; that is done by the projector.
        self.target_geo = target_geo
        self.target_geo.array = np.atleast_3d(self.target_geo.array) #atleast_3d should be used here when all formulation accepts 2D array as 3D array with third dimension of 1.
        self.proj_geo = proj_geo

        #Initialize performance logger
        self.logs = LogPerf(options)
        self.logs.startTiming()

        #Unpack options
        # self.__dict__.update(options.__dict__) #unpack the option variables to itself

        #Renamed variables has a separate copy
        self.response_model = options.response_model
        self.eps = options.eps
        self.weight = options.weight
        self.p = options.p
        self.q = options.q
        self.learning_rate = options.learning_rate
        self.momentum = options.momentum
        self.regularize_l = options.regularize_l
        self.optim_alg = options.optim_alg
        self.g0 = options.g0
        self.glb = options.blb
        self.gub = options.bub
        self.bit_depth = options.bit_depth
        self.dvol = 1 #differential volume in integration
        self.exit_param = options.exit_param #Convergence tolerance. Expressed as percentage change of loss.
        self.verbose = options.verbose
        self.save_img_path = options.save_img_path

        #State variables, in the order of computation
        self.dose = None
        self.mapped_dose = None
        self.mapped_dose_error_from_f_T = None #error from f_T. Without subtracting epsilon
        self.mapped_dose_error_from_band = None #error from f_T. Without subtracting epsilon
        self.v = None #indicator function of constraint violation
        self.prev_update = 0

        #Iteration index of each variable.
        self.dose_iter = -1
        self.mapped_dose_iter = -1
        self.mapped_dose_error_from_f_T_iter = -1
        self.mapped_dose_error_from_band_iter = -1
        self.v_iter = -1
        self.loss_iter = -1
        self.loss_grad_iter = -1
        self.prev_update_iter = -1

        # [W/UW NOTE] Live plots (verbose 'plot' / 'plot_demo') show target_geo, dose and mapped_dose on the same W-domain x-y grid. They show WRAPPED results; nothing is unrolled for display.
        # [UNUSED] verbose 'plot' / 'plot_demo' (here and in callback, gradientDescent, minimizeBCLP) are never used in this project:
        # [UNUSED] the runs use verbose='time', and vamtoolbox's EvolvingPlot expects a W-grid TargetGeometry, not the UW target.
        if self.verbose == 'plot':
            self.dp = vamtoolbox.displaygrayscale.EvolvingPlot(target_geo,self.logs.options.n_iter+1, self.save_img_path)
        elif self.verbose == 'plot_demo':
            self.dp = vamtoolbox.displaygrayscale.EvolvingPlotDemo(target_geo,self.logs.options.n_iter+1, self.save_img_path)


        #Setup projector
        # [W/UW NOTE] If a projector is given (R2R: R2RAlgebraicPropagator with the paper's composite matrix P_{f_UW -> g}, which combines T: UW -> W with propagation), it is used as is.
        # [W/UW NOTE] Otherwise this is vamtoolbox's standard projector (parallel-ray Radon transform over proj_geo.angles, W-domain tomogram <-> r-theta sinogram), as in conventional CAL.
        if projector is not None:
            self.P = projector
        else:
            self.P = vamtoolbox.projectorconstructor.projectorconstructor(self.target_geo, self.proj_geo)
        self.logger.info(f"Projector: {type(self.P).__name__}")

        # self.custom_fig1, self.custom_ax1 = plt.subplots()
        # self.custom_fig2, self.custom_ax2 = plt.subplots()

        #Check if input f_T in range of response function over non-negative real
        if self.response_model.checkResponseTarget(self.target_geo.array) is False:
            warnings.warn("Target is either out of range of response function over non-negative real dose input, or contains inf/nan.")

        #Obtain initial sinogram (first optimization iterate)
        if self.g0 is None:
            #If it is not externally provided, generate initialization sinogram.
            if callable(getattr(self.P, 'inverseBackward', None)): #if the projector has inverseBackward() method defined, use that to generate initial guess
                self.g0 = self.P.inverseBackward(self.response_model.map_inv(self.target_geo.array), options.filter)
                self.logger.info('Initial sinogram is generated by propagator inverseBackward() method.')

            else: #Generating initial guess using forward propagation.     
                self.g0 = self.P.forward(self.response_model.map_inv(self.target_geo.array))
                self.g0 = vamtoolbox.util.data.filterSinogram(self.g0, options.filter)
                self.g0 *= np.pi/(2*proj_geo.n_angles) #This scales the filtered backpropagation results correctly such that original target scale is maintained.
                self.logger.info('Propagator has no inverseBackward() method. Initialize with forward propagation and filtering in sinogram space.')
 

        # [W/UW NOTE] Careful with terms: the [WRAP NOTE] comments below use "wrap/unwrap" only for array reshaping (cylindrical <-> flattened sinogram).
        # [W/UW NOTE] That reshaping is unrelated to the paper's physical W/UW domains. Reshaping never changes which domain the data is in.
        # [WRAP NOTE] Overview: this file does NOT do angular/phase wrapping (no np.unwrap, no modulo 2*pi, no np.roll).
        # [WRAP NOTE] The only "wrap/unwrap" is a reshape of the sinogram between two layouts:
        # [WRAP NOTE]   - "cylindrical" (wrapped): the native N-D shape the propagator returns, e.g. (n_detector_u, n_angles, n_detector_v).
        # [WRAP NOTE]   - "flattened" (unwrapped): a 1D vector of the same values, used by the gradient-descent update.
        # [WRAP NOTE] Both directions go through checkSinogramShape(). They are lossless because np.reshape only reinterprets the index order (C order) and does not move or change data.
        # [WRAP NOTE] The line below saves the original cylindrical shape. It is the "template" used later to re-wrap any flattened vector.
        self.g0_shape = self.g0.shape
        self.g0 = self.imposeSinogramConstraints(self.g0)

        # #Evaluate performance of initialization --> Turns out it should not be run here. It will be executed in the first iteration anyway.
        # self.updateVariables(self.g0) #update loss and gradient.
        # self.callback(self.g0) #Record other metrics and initialization time as iter 0


    def updateVariables(self, g_iter, update):
        """
        This function checks if the state variables are up-to-date, and update them if not.
        The function ensures the variables are (1) computed in a specified order (to avoid accessing old copy), and (2) computed only once.
        Therefore regardless of when external function query loss or loss gradient, there is no ambiguity of when each variable is updated.
        In the following code, any particular line can safely assume the variables above it are up-to-date.
        """

        if self.dose_iter != self.logs.curr_iter:
            # [WRAP NOTE] WRAP: the optimizer holds g_iter as a flat 1D vector. It is reshaped back to g0_shape here because the propagator's backward() expects the cylindrical sinogram.
            g_iter = self.checkSinogramShape(g_iter, desired_shape = "cylindrical")
            # [W/UW NOTE] backward() returns the dose on the W-domain x-y(-z) grid of target_geo. Every error, loss and gradient below is therefore computed against the target in W.
            self.dose  = self.P.backward(g_iter)
            # [WRAP NOTE] The same idea applies in real space. If the projector returns a flattened reconstruction, it is re-wrapped to the target's 2D/3D shape.
            if self.dose.ndim <= 1: #if the tomogram recon is flattened
                self.dose = self.dose.reshape(self.target_geo.array.shape) #all tomogram space computation happens in 2D/3D
            self.dose_iter = self.logs.curr_iter

        if self.mapped_dose_iter != self.logs.curr_iter:
            self.mapped_dose = self.response_model.map(self.dose)
            self.mapped_dose_iter = self.logs.curr_iter

        if self.mapped_dose_error_from_f_T_iter != self.logs.curr_iter:
            self.mapped_dose_error_from_f_T = self.mapped_dose - self.target_geo.array  #'''When the operands have size mismatch, numpy automatically broadcast the array.''' 
            self.mapped_dose_error_from_f_T_iter = self.logs.curr_iter

        if self.mapped_dose_error_from_band_iter != self.logs.curr_iter:
            #"mapped_dose_error_from_band" is by definition positive in the constraint violation zone, and either zero or negative outside.
            self.mapped_dose_error_from_band = np.abs(self.mapped_dose_error_from_f_T) - self.eps
            self.mapped_dose_error_from_band_iter = self.logs.curr_iter

        if self.v_iter != self.logs.curr_iter:
            self.v = (self.mapped_dose_error_from_band > 0)
            self.v_iter = self.logs.curr_iter

        if self.loss_iter != self.logs.curr_iter:
            self.loss = self.computeLoss()
            self.loss_iter = self.logs.curr_iter

        if self.loss_grad_iter != self.logs.curr_iter:
            self.loss_grad = self.computeLossGradient()
            self.loss_grad_iter = self.logs.curr_iter

        if type(update) == np.ndarray:
            if self.prev_update_iter != self.logs.curr_iter and self.logs.curr_iter != 0:
                self.prev_update = update
                self.prev_update_iter = self.logs.curr_iter

    def computeLoss(self):
        """
        Computation of the loss function
        """

        #The absolute sign around "mapped_dose_error_from_band" (as in the definition of Lp norm) is preserved here, although simiplification in the paper hides it.
        #This absolute sign is to avoid letting the negative values of mapped_dose_error_from_band to create NaNs in **p operation when p<1. The NaNs would avoid the sum to be properly evaluated.
        #These originally negative voxels would not contribute to the loss_integrand due to selection by v. 
        # loss_integrand = self.v*self.weight*np.abs(self.mapped_dose_error_from_band)**self.p
        # [UNUSED] original_method is always False: the first branch below never runs.
        if original_method := False: #set true to use original method, set false to use updated method. TODO
            # Original method with regularization term incorrectly included in the loss_integrand.
            loss_integrand = self.v*self.weight*np.abs(self.mapped_dose_error_from_band)**self.p + self.regularize_l*self.mapped_dose**2
            # regularize_l helps to optimize for low-energy solution 
            # loss = (np.sum(loss_integrand).astype('double')*self.dvol)**(self.q/self.p) #multiply by a constant differential volume, self.dvol. #We may not need the astype('double')
            loss = (np.sum(loss_integrand)*self.dvol)**(self.q/self.p) #multiply by a constant differential volume, self.dvol
        else:    
            # Start of updated method with regularization term correctly evaluated in a separate integral
            loss_integrand = self.v*self.weight*np.abs(self.mapped_dose_error_from_band)**self.p
            # regularize_l helps to optimize for low-energy solution 
            loss = (np.sum(loss_integrand).astype('double')*self.dvol)**(self.q/self.p) #multiply by a constant differential volume, self.dvol. #We may not need the astype('double')
            # loss = (np.sum(loss_integrand)*self.dvol)**(self.q/self.p) #multiply by a constant differential volume, self.dvol

            loss_integrand_reg = self.regularize_l*self.mapped_dose**2 #This is the regularization term, which is not part of the Lp norm.
            loss_reg = (np.sum(loss_integrand_reg)*self.dvol)
            loss += loss_reg #Add the regularization term to the loss
            # End of updated method

        self.logs.loss[self.logs.curr_iter] = loss

        return loss



    def computeLossGradient(self):
        """
        Computation of the loss gradient. Output gradient as a flattened array.
        """

        #The absolute sign around "mapped_dose_error_from_band" (as in the definition of Lp norm) is preserved here, although simiplification in the paper hides it.
        #This absolute sign is to avoid letting the negative values of mapped_dose_error_from_band to create NaNs in **(p-1) operation when p<2. The NaNs would start to spread throughout g_iter can cause errors.
        #These originally negative voxels would not contribute to the operand due to selection by v. 
        # operand = self.v * self.weight * (np.abs(self.mapped_dose_error_from_band)**(self.p-1)) * np.sign(self.mapped_dose_error_from_f_T) * self.response_model.dmapdf(self.dose)
        # [UNUSED] original_method is always False: the first branch below never runs.
        if original_method := False: #set true to use original method, set false to use updated method
            operand = self.v * self.weight * (np.abs(self.mapped_dose_error_from_band)**(self.p-1)) * np.sign(self.mapped_dose_error_from_f_T) * self.response_model.dmapdf(self.dose) + 2*self.regularize_l*self.mapped_dose

            #Computation of gradient happens between the two iterations. Now the index just incremented, but we need to access the loss evaluated in last iteration
            loss_grad =  ( self.q * self.loss**((self.q - self.p)/self.q) ) * self.P.forward(operand) if self.loss != 0.0 else self.P.forward(operand)
        else:
            operand = self.v * self.weight * (np.abs(self.mapped_dose_error_from_band)**(self.p-1)) * np.sign(self.mapped_dose_error_from_f_T) * self.response_model.dmapdf(self.dose)
            loss_grad =  ( self.q * self.loss**((self.q - self.p)/self.q) ) * self.P.forward(operand) if self.loss != 0.0 else self.P.forward(operand)

            loss_grad_reg = 2*self.regularize_l * self.P.forward(self.response_model.dmapdf(self.dose))

            loss_grad += loss_grad_reg #Add the regularization term to the loss gradient
        '''
        Explanation of the conditional evaluation of loss_grad:
        When (self.loss == 0.0) AND (self.q-self.p < 0), the ( self.q * self.loss**((self.q - self.p)/self.q) ) results in infinity because it is a division by 0.
        self.P.forward(operand) is an sinogram array of all zeros.
        Infinity * 0 results in nan.

        To fix this problem, we force the gradient to be zero in this step to prevent the sinogram to be further updated. 
        '''
        # [WRAP NOTE] UNWRAP: P.forward() returns the gradient in cylindrical sinogram shape. It is flattened so it matches the 1D g_iter in the update g_iter - learning_rate * update.
        loss_grad = self.checkSinogramShape(loss_grad, desired_shape= "flattened")

        return loss_grad



    def getLoss(self, g_iter):
        self.updateVariables(g_iter,None)
        return self.loss

    def getLossGrad(self, g_iter):
        self.updateVariables(g_iter,None)
        return self.loss_grad

    def getPrevUpdate(self, g_iter):
        self.updateVariables(g_iter,None)
        return self.prev_update

    def evaluateNormMetrics(self):
        #All following norms are evaluated with eps = original value and eps = 0
        #l0 and l0_eps0

        # l0 = np.sum(self.v).astype('double')*self.dvol #multiply by a constant differential volume, self.dvol
        l0 = np.sum(self.v)*self.dvol #multiply by a constant differential volume, self.dvol
        self.logs.l0[self.logs.curr_iter] = l0
        
        v_eps0 = (np.abs(self.mapped_dose_error_from_f_T) > 0)
        l0_eps0 = np.sum(v_eps0)*self.dvol #multiply by a constant differential volume, self.dvol
        self.logs.l0_eps0[self.logs.curr_iter] = l0_eps0

        #l1 and l1_eps0
        l1 = np.sum(self.v*np.abs(self.mapped_dose_error_from_band))*self.dvol
        self.logs.l1[self.logs.curr_iter] = l1

        l1_eps0 = np.sum(np.abs(self.mapped_dose_error_from_f_T))*self.dvol  #multiplication with v_eps0 is suppressed for performance. Such binary mask is unnecessary in this case. 
        self.logs.l1_eps0[self.logs.curr_iter] = l1_eps0

        #l2 and l2_eps0 
        l2 = np.sqrt(np.sum(self.v*(self.mapped_dose_error_from_band**2))*self.dvol)
        self.logs.l2[self.logs.curr_iter] = l2

        l2_eps0 = np.sqrt(np.sum(self.mapped_dose_error_from_f_T**2)*self.dvol)  #multiplication with v_eps0 is suppressed for performance. Such binary mask is unnecessary in this case.
        self.logs.l2_eps0[self.logs.curr_iter] = l2_eps0

        #lf and lf_eps0 
        #The existance of self.v here is important. Without it, we can't select only the constraint violation only.
        #In the case where all voxels have negative errors (constraint satisfaction), these values could possibly show up either as negative (without abs) and positive (with abs)
        lf = np.amax(self.v*np.abs(self.mapped_dose_error_from_band))  
        self.logs.lf[self.logs.curr_iter] = lf
        
        lf_eps0 = np.amax(v_eps0 * np.abs(self.mapped_dose_error_from_f_T))  
        self.logs.lf_eps0[self.logs.curr_iter] = lf_eps0
        

    def callback(self, g_iter, update):
        #This function evaluate other metrics and record iter time
        #Evaluate other metrics
        self.updateVariables(g_iter, update)
        self.evaluateNormMetrics()

        #record iter time
        self.logs.recordIterTime()
        if self.verbose == 'time' or self.verbose == 'plot' or self.verbose == 'plot_demo':
            self.logger.info(f'Iteration {self.logs.curr_iter: 4.0f} at time: { self.logs.iter_times[self.logs.curr_iter]: 6.1f} s. Loss = {self.logs.loss[self.logs.curr_iter]}')
            
        if self.verbose == 'plot':
            self.dp.update(self.logs.loss, self.dose, self.mapped_dose, [self.logs.l0, self.logs.l1, self.logs.l2, self.logs.lf, self.logs.l0_eps0, self.logs.l1_eps0, self.logs.l2_eps0, self.logs.lf_eps0])
        elif self.verbose == 'plot_demo':
            self.dp.update(self.logs.loss, self.dose, self.mapped_dose)

        self.logs.curr_iter += 1


    def gradientDescent(self): 
        #TODO: using scipy optimize method format such that it could be called by "minimize" front end
        # [WRAP NOTE] UNWRAP: the initial sinogram is flattened once. All of the gradient-descent arithmetic below works on this 1D vector (the format scipy.optimize would also expect).
        g_iter = self.checkSinogramShape(self.g0, desired_shape = 'flattened')

        #Impose constraint on latest g here
        for iter in range(self.logs.options.n_iter): 
            #The first iteration here evaluate the initial solution performance, and generate the next g_iter (for iteration 1).
            #Variables and performance are evaluated at the beginning of each loop iteration (by construction). g_iter generated is for the next iteration.
            #This process is repeated for n_iter times.
            # g_iter = g_iter - self.learning_rate * self.getLossGrad(g_iter)
            if self.momentum == 0.0:
                update = self.getLossGrad(g_iter)
            else:
                update = self.momentum * self.getPrevUpdate(g_iter) + (1-self.momentum) * self.getLossGrad(g_iter)

            g_iter = g_iter - self.learning_rate * update
            g_iter = self.imposeSinogramConstraints(g_iter)
            
            #Evaluate performance
            self.getLoss(g_iter)
            is_converged = self.checkConvergence() #This is where the loss function and gradient is evaluated, but the other auxilliary norms are not evaluated yet.
            self.callback(g_iter,update) #self.logs.curr_iter is updated here
            if is_converged:
                self.logger.warning('Optimization converged or loss value reached zero.')
                self.logs.is_converged = is_converged
                break
            if msvcrt.kbhit():
                if msvcrt.getwche() == 'q':
                    break
            
        self.updateVariables(g_iter,update) #Update the variables according to the last generated g_iter
        self.evaluateNormMetrics() #compute the metrics for the last generated g_iter

        if self.verbose == 'plot':
            self.dp.update(self.logs.loss, self.dose, self.mapped_dose, [self.logs.l0, self.logs.l1, self.logs.l2, self.logs.lf, self.logs.l0_eps0, self.logs.l1_eps0, self.logs.l2_eps0, self.logs.lf_eps0])
        elif self.verbose == 'plot_demo':
            self.dp.update(self.logs.loss, self.dose, self.mapped_dose)

        return g_iter


    def imposeSinogramConstraints(self, g):
        # [WRAP NOTE] This function records whether its input arrived unwrapped (1D) so that it can return the same layout at the end.
        was_flattened = (g.ndim == 1)

        # 1) Always clip (works for 1D or 3D)
        g = np.clip(g, a_min=self.glb, a_max=self.gub)

        # 2) Only discretize sometimes
        if self.bit_depth is not None and ((self.logs.curr_iter+1) % 10 == 0):

            # Ensure 3D for per-b slicing
            # [WRAP NOTE] WRAP (temporary): discretization is done per angle, along axis 1 of the cylindrical sinogram. That axis only exists after re-wrapping to g0_shape.
            # [WRAP NOTE] If g was flat, np.reshape returns a view, so writes into g3 also change the underlying data of g.
            g3 = self.checkSinogramShape(g, desired_shape="cylindrical")

            for i in range(g3.shape[1]):
                slice_i = g3[:, i, :]
                mx = float(np.max(slice_i))
                if mx > 0:
                    g3[:, i, :] = vamtoolbox.util.data.discretize(
                        slice_i,
                        self.bit_depth,
                        [0.0, mx]   # consider [self.glb, self.gub] instead
                    )

            g = g3

            # Re-clip after discretize (optional but usually fine)
            g = np.clip(g, a_min=self.glb, a_max=self.gub)

        # 3) Return in same form as input
        # [WRAP NOTE] UNWRAP (restore): if the caller passed a 1D vector, the result is flattened again so the caller's layout is preserved. A cylindrical input is returned as-is.
        if was_flattened:
            return self.checkSinogramShape(g, desired_shape="flattened")
        else:
            return g



    def checkSinogramShape(self, g, desired_shape = 'flattened'):
        #This function check the shape of sinogram array and reshape it if needed.
        # [WRAP NOTE] This is the single place where wrapping and unwrapping happen. Both branches are idempotent: an array that is already in the requested layout passes through unchanged.
        if desired_shape == 'flattened':
            # [WRAP NOTE] UNWRAP: N-D -> 1D of length prod(shape), in C (row-major) order, so the last axis varies fastest.
            if len(g.shape)> 1:
                g = np.reshape(g, np.prod(g.shape))

        elif desired_shape == 'cylindrical':
            # [WRAP NOTE] WRAP: 1D -> the original g0_shape, using the same C order. This makes it the exact inverse of the unwrap above.
            # [WRAP NOTE] The shape is not inferred. It must come from self.g0_shape, which is recorded in __init__.
            if len(g.shape) == 1:
                g = np.reshape(g, self.g0_shape)
        
        return g
        
    def checkConvergence(self):
        #This function checks convergence by evaluating fractional change of loss from previous to current iteration
        #Slope on the plot Loss vs effort (effort = step size * iteration). This slope is evaluated over one iteration.
        is_loss_zero = True if self.logs.loss[self.logs.curr_iter] == 0.0 else False

        avg_over_n_iter = 5 #Average over n iterations to check convergence. Taken as 5.
        if self.logs.curr_iter >= avg_over_n_iter and self.exit_param is not None: #At least get 6 iterations before checking convergence
            change_of_loss = np.abs(np.diff(self.logs.loss[self.logs.curr_iter-avg_over_n_iter:self.logs.curr_iter+1])) #obtain 5 differences over 6 iterations. +1 is needed to include curr_iter
            avg_change_of_loss_over_n_iter = np.mean(change_of_loss) #average over 5 iterations
            if avg_change_of_loss_over_n_iter <= self.exit_param*self.logs.loss[self.logs.curr_iter]:
                is_converged = True

        # if self.logs.curr_iter > 0 and self.exit_param is not None:
        #     norm_grad = np.linalg.norm(self.loss_grad)
        #     if norm_grad <= self.exit_param:
        #         is_converged = True

        # if self.logs.curr_iter > 0 and self.exit_param is not None:
        #     change_of_loss = np.abs(self.logs.loss[self.logs.curr_iter] - self.logs.loss[self.logs.curr_iter-1])
        #     if change_of_loss <= self.exit_param*self.logs.loss[self.logs.curr_iter-1]*self.learning_rate:
        #         is_converged = True

            else:
                is_converged = False
        else:
            is_converged = False
        
        return is_converged or is_loss_zero

# [UNUSED] Not called in this project: propagator.minimizeR2RBCLP uses BCLPNorm directly with the R2R propagator.
# [UNUSED] This is vamtoolbox's conventional CAL (W domain) front end; its [W/UW NOTE] comments describe that use, not R2R.
def minimizeBCLP(target_geo, proj_geo, options, output = "packaged"):
    """
    Band constraint Lp norm minimization. 

    Parameters
    ----------
    target_geo : geometry.TargetGeometry object

    proj_geo : geometry.ProjectionGeometry object

    options : Options object

    Returns
    -------
    geometry.Sinogram   
        optimized sinogram
    geometry.Reconstruction
        reconstruction from non-filtered backprojection of the 
        optimized reconstructions
    
    References
    ----------

    """
    
    #Initialize norm class

    bclp = BCLPNorm(target_geo, proj_geo, options)
    g_opt = bclp.gradientDescent()
    # [WRAP NOTE] Final WRAP: gradientDescent() returns a flat vector. It is reshaped back to the cylindrical sinogram before being packaged into a vamtoolbox Sinogram.
    g_opt = bclp.checkSinogramShape(g_opt, desired_shape = "cylindrical")

    if bclp.verbose == 'plot':
        bclp.dp.ioff()



    # [W/UW NOTE] OUTPUT domains:
    # [W/UW NOTE]   - g_opt / Sinogram: the r-theta projection space (detector x angle x z). It is neither W nor UW tomogram space.
    # [W/UW NOTE]   - bclp.dose / bclp.mapped_dose / Reconstruction: WRAPPED (W) x-y(-z) grid, the same grid as the input target.
    # [W/UW NOTE] No unwrapped (UW, flat-web) result is produced here. Viewing the print as the flat web would require a separate W -> UW mapping.
    if output == "packaged":
        return vamtoolbox.geometry.Sinogram(g_opt, proj_geo, options), vamtoolbox.geometry.Reconstruction(bclp.dose, proj_geo, options), bclp.logs.loss
    elif output == "full":
        return vamtoolbox.geometry.Sinogram(g_opt, proj_geo, options), vamtoolbox.geometry.Reconstruction(bclp.dose, proj_geo, options), vamtoolbox.geometry.Reconstruction(bclp.mapped_dose, proj_geo, options), bclp.logs
    elif output == "components":
        return g_opt, bclp.dose, bclp.mapped_dose, bclp.logs

    #Future improvement: Parse options and run other optimization methods in Scipy minimize 
    # lbfgs_options = {}
    # lbfgs_options['maxiter'] = options.n_iter
    # lbfgs_options['ftol'] = 1e-12#np.finfo(float).eps
    # # lbfgs_options['gtol'] = np.finfo(float).eps
    # lbfgs_options['disp'] = False
    # lbfgs_options['maxcor'] = 10
    # lbfgs_options['maxls'] = 30

    # result = optimize.minimize(fun=loss, 
    #                         x0=b0, 
    #                         callback=callback,
    #                         method='L-BFGS-B', 
    #                         jac=lossGradient, 
    #                         bounds=bounds, 
    #                         options=lbfgs_options)