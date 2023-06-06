import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import os

"""
The Solver
"""

class LagrangianSolver():
    def __init__(self, name):
        self.name = name
        self.mkdir_if_not_exist(os.path.dirname(__file__)+"/Output/")
        self.mkdir_if_not_exist(os.path.dirname(__file__)+"/Output/pickls/")
        self.mkdir_if_not_exist(os.path.dirname(__file__)+"/Output/png_spam/")
    
    def initialize_sim(self, lagrangian, initial_state, metric, granularity):
        self.dims = np.array(initial_state).shape[2]
        self.lagrangian = lagrangian
        self.metric = metric
        self.positions = np.array(initial_state)[1, :]
        [self.step_count, self.variation_step, self.tolerance] = granularity
        """Note step length is set by distance between initial points"""
        paths_shape = [self.step_count+2]
        for i in range(2): #axis 0 = steps, 1 = subjects, 2 = points
            paths_shape.append(self.positions.shape[i])
        self.paths = np.zeros(paths_shape)
        self.paths[:2] = initial_state

    def mkdir_if_not_exist(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def initialize_postprocess(self, lagrangian, metric, granularity):
        self.lagrangian = lagrangian
        self.metric = metric
        [self.step_count, self.variation_step] = granularity

    def open_old(self):
        with open(os.path.dirname(__file__)+"/Output/pickls/"+self.name+".dat", "rb") as pickle_file:
            data = pickle.load(pickle_file)
        self.paths = data["paths"]
        self.dims = self.paths.shape[2]

    def rename(self, new_name):
        self.name = new_name
        with open(os.path.dirname(__file__)+"/Output/pickls/"+self.name+".dat", "wb") as pickle_file:
            pickle.dump({"paths": self.paths}, pickle_file)

    def lorentz_dot(self, pos, v1, v2):
        v1dotv2 = 0
        metric = self.metric(pos)
        for i in range(self.dims):
            for j in range(self.dims):
                v1dotv2 += v1[i] * metric[i,j] * v2[j]
        return v1dotv2

    def get_L_d(self, pos1, pos2, s):
        delta_tau = np.sqrt(self.lorentz_dot((pos1+pos2)/2, (pos2-pos1), (pos2-pos1))) # technically this is delta_tau^2 but it struggles to converge for close orbits if I sqrt it and this should have the same extremal path
        L_d = self.lagrangian(pos1, pos2, s) * delta_tau
        return L_d

    def get_L_d_derivative(self, fixed_point, variable_point, variable_idx, s):
        variation = np.zeros(self.dims)
        variation[variable_idx] = self.variation_step
        L_d_derivative = (self.get_L_d(fixed_point, variable_point+variation, s) - self.get_L_d(fixed_point, variable_point-variation, s))/(2*self.variation_step)
        return L_d_derivative

    def get_variation_space_grad(self, known_point, test_point, target, s):
        """NOTE the known point is the one varied on individual variations but held fixed between them"""
        grad = np.zeros(self.dims)
        for i in range(self.dims):
            variation = np.zeros(self.dims)
            variation[i] = self.variation_step
            grad[i] = (self.get_L_d_derivative(test_point+variation, known_point, i, s) - self.get_L_d_derivative(test_point-variation, known_point, i, s))/(2*self.variation_step)
            grad[i] += target[i]
        return grad

    def check_guess(self, guess):
        incorrect = False
        for d in range(self.dims):
            if abs(guess[d]) > self.tolerance:
                incorrect = True
        return incorrect

    def find_step(self, s, n=1):
        p_0 = np.array(self.paths[n-2, s])
        p_1 = np.array(self.paths[n-1, s])

        p2_guess = p_1 + (p_1-p_0) #+ np.array([0,0.01,-0.01,0.01])
        dLdp01 = np.zeros(self.dims)
        dLdp12 = np.zeros(self.dims)
        for d in range(self.dims):
            dLdp01[d] = self.get_L_d_derivative(p_0, p_1, d, s)
            dLdp12[d] = self.get_L_d_derivative(p2_guess, p_1, d, s)
        grad = self.get_variation_space_grad(p_1, p2_guess, dLdp01, s)
        dif = dLdp01+dLdp12
        
        i = 0
        while self.check_guess(dif):
            p2_guess -= (dif/grad)*(self.variation_step**(i/500))
            grad = self.get_variation_space_grad(p_1, p2_guess, dLdp01, s)
            for d in range(self.dims):
                dLdp12[d] = self.get_L_d_derivative(p2_guess, p_1, d, s)
            dif = dLdp01+dLdp12
            i += 1
            if i == 10000: 
                print("too many loops forcibly truncating")
                self.log_immediately = True
                break

        p_2 = p2_guess

        return p_2

    def take_steps(self, n=1):
        for s in range(self.positions.shape[0]):
            new_pos = self.find_step(s, n=n)
            self.positions[s] = new_pos

    def run(self):
        self.log_immediately = False
        for n in range(self.step_count):
            if n % (int(self.step_count/10)) == 0:
                print(n)
            self.take_steps(n+2)
            for s in range(self.positions.shape[0]):
                self.paths[n+2, s] = self.positions[s]
            if self.log_immediately == True:
                self.paths = self.paths[:n+2]
                break
        
        with open(os.path.dirname(__file__)+"/Output/pickls/"+self.name+".dat", "wb") as pickle_file:
            pickle.dump({"paths": self.paths}, pickle_file)

    def plot(self, orientation=[30, 220], color_time=True):
        if self.dims == 2:
            vert_num = 1
            horz_num = 1
            gs = gridspec.GridSpec(vert_num, horz_num)
            fig = plt.figure(figsize=(horz_num*3, vert_num*3), dpi=300)
            
            ax = fig.add_subplot(gs[0, 0])
            ax.set_ylabel("t")
            ax.set_xlabel("x")

            for s in range(self.paths.shape[1]):
                ax.plot(self.paths[:,s,1], self.paths[:,s,0], f"C{s}-", marker=".")

            plt.tight_layout()
            plt.savefig(os.path.dirname(__file__)+"/Output/"+self.name+".png")
            plt.close()
        if self.dims == 3:
            vert_num = 1
            horz_num = 1
            gs = gridspec.GridSpec(vert_num, horz_num)
            fig = plt.figure(figsize=(horz_num*3, vert_num*3), dpi=300)
            
            ax = fig.add_subplot(gs[0, 0], projection='3d')
            ax.set_zlabel("t")
            ax.set_ylabel("y")
            ax.set_xlabel("x")
            
            ax.set_xlim([-30, 30])
            ax.set_ylim([-30, 30])
            #ax.set_xlim([-1.1,1.1])
            #ax.set_ylim([-1.1,1.1])

            for s in range(self.paths.shape[1]):
                ax.plot(self.paths[:,s,1], self.paths[:,s,2], self.paths[:,s,0], f"C{s}-")

            maxes = np.zeros(self.paths.shape[1])
            for s in range(self.paths.shape[1]):
                maxes[s] = max(self.paths[:,s,0])
            max_time = max(maxes)

            ax.plot([0,0], [0,0], [0,max_time], c="black")

            ax.view_init(orientation[0], orientation[1])
            #plt.tight_layout()
            plt.savefig(os.path.dirname(__file__)+"/Output/"+self.name+".png")
            plt.close()
        if self.dims == 4:
            maxes = np.zeros([self.paths.shape[1], self.paths.shape[2]])
            upper_bounds = np.zeros([self.paths.shape[2]])
            mins = np.zeros([self.paths.shape[1], self.paths.shape[2]])
            lower_bounds = np.zeros([self.paths.shape[2]])
            bounds = np.zeros(self.paths.shape[2])
            for s in range(self.paths.shape[1]):
                for d in range(self.paths.shape[2]):
                    maxes[s,d] = max(self.paths[:,s,d])
                    mins[s,d] = min(self.paths[:,s,d])
            for d in range(self.paths.shape[2]):
                upper_bounds[d] = max(maxes[:,d])
                lower_bounds[d] = min(mins[:,d])
                bounds[d] = max([abs(upper_bounds[d]), abs(lower_bounds[d])])
            
            if color_time:
                vert_num = 1
                horz_num = 1
                gs = gridspec.GridSpec(vert_num, horz_num)
                fig = plt.figure(figsize=(horz_num*3, vert_num*3), dpi=300)
                
                ax = fig.add_subplot(gs[0, 0], projection='3d')
                ax.set_zlabel("z")
                ax.set_ylabel("y")
                ax.set_xlabel("x")
                
                ax.set_xlim([-1*bounds[1], bounds[1]])
                ax.set_ylim([-1*bounds[2], bounds[2]])
                ax.set_zlim([-1*bounds[3], bounds[3]])
                #ax.set_xlim([-5, 5])
                #ax.set_ylim([-5, 5])
                #ax.set_zlim([-5, 5])

                cmaps = np.array([plt.cm.viridis, plt.cm.plasma])
                for s in range(self.paths.shape[1]):
                    for n in range(self.paths.shape[0]-1):
                        ax.plot(self.paths[n:n+2,s,1]
                                , self.paths[n:n+2,s,2]
                                , self.paths[n:n+2,s,3]
                                , color=cmaps[s](1-(self.paths[n,s,0])/(self.paths[-1,s,0])))

                ax.plot([0], [0], [0], c="black", marker=".")

                ax.view_init(orientation[0], orientation[1])
                #plt.tight_layout()
                plt.savefig(os.path.dirname(__file__)+"/Output/"+self.name+".png")
                plt.close()
            else:
                for n in range(self.paths.shape[0]):
                    vert_num = 1
                    horz_num = 1
                    gs = gridspec.GridSpec(vert_num, horz_num)
                    fig = plt.figure(figsize=(horz_num*3, vert_num*3), dpi=300)
                    
                    ax = fig.add_subplot(gs[0, 0], projection='3d')
                    ax.set_zlabel("z")
                    ax.set_ylabel("y")
                    ax.set_xlabel("x")
                    
                    #ax.set_xlim([-1*bounds[1], bounds[1]])
                    #ax.set_ylim([-1*bounds[2], bounds[2]])
                    #ax.set_zlim([-1*bounds[3], bounds[3]])
                    ax.set_xlim([-20, 20])
                    ax.set_ylim([-20, 20])
                    ax.set_zlim([-1*bounds[3], bounds[3]])

                    for s in range(self.paths.shape[1]):
                        ax.plot(self.paths[:n+1,s,1], self.paths[:n+1,s,2], self.paths[:n+1,s,3], f"C{s}-")
                        ax.plot(self.paths[n,s,1], self.paths[n,s,2], self.paths[n,s,3], f"C{s}-", marker=".")

                    ax.plot([0], [0], [0], c="black", marker=".")

                    ax.view_init(orientation[0], orientation[1])
                    #plt.tight_layout()
                    plt.savefig(os.path.dirname(__file__)+"/Output/png_spam/"+self.name+"%05d.png" % (n))
                    plt.close()
                    print("plotting:", n)
                self.peg()
                self.cleanup_pngs()

    def cleanup_pngs(self):
        os.system("rm -rf "+os.path.dirname(__file__)+"/Output/png_spam/*")
    
    def peg(self, fps=48):
        os.system(f"ffmpeg -framerate {fps} -i "+os.path.dirname(__file__)+"/Output/png_spam/"+self.name+"%05d.png -vf scale=1280:-2 "+os.path.dirname(__file__)+"/Output/"+self.name+".mp4")

    def plot_diagnostic(self):
        proper_time_step = np.zeros([self.paths.shape[0]-1, self.paths.shape[1]])
        fourvel = np.zeros(np.array(self.paths.shape) - [1, 0, 0])
        u_dot_u = np.zeros([self.paths.shape[0]-1, self.paths.shape[1]])
        for s in range(self.paths.shape[1]):
            for n in range(self.paths.shape[0]-1):
                proper_time_step[n, s] = np.sqrt(self.lorentz_dot((self.paths[n,s]+self.paths[n+1,s])/2, (self.paths[n+1,s]-self.paths[n,s]), (self.paths[n+1,s]-self.paths[n,s])))
            for d in range(self.dims):
                fourvel[:,s,d] = (self.paths[1:,s,d] - self.paths[:-1,s,d])/proper_time_step[:,s]
            for n in range(self.paths.shape[0]-1):
                u_dot_u[n] = self.lorentz_dot((self.paths[n+1,s]+self.paths[n,s])/2, fourvel[n,s], fourvel[n,s])

        vert_num = 2
        horz_num = 3
        gs = gridspec.GridSpec(vert_num, horz_num)
        fig = plt.figure(figsize=(horz_num*3, vert_num*3), dpi=300)
        
        axuu = fig.add_subplot(gs[0, 0])
        axuu.set_ylabel(r"$u^{\mu}u_{\mu}$")
        axuu.set_xlabel("steps")
        axuu.set_ylim([1-(1e-14), 1+(1e-14)])

        axtau = fig.add_subplot(gs[1, 0])
        axtau.set_ylabel(r"$\Delta \tau$")
        axtau.set_xlabel("steps")
        
        axp0 = fig.add_subplot(gs[0, 1])
        axp0.set_ylabel(r"E/m")
        axp0.set_xlabel("steps")
        #axp0.set_ylim([0, max(fourvel[0,:,0])*2])

        axp1 = fig.add_subplot(gs[1, 1])
        axp1.set_ylabel(r"${p_x}/m$")
        axp1.set_xlabel("steps")
        #axp1.set_ylim([0, max(fourvel[0,:,1])*2])

        axp2 = fig.add_subplot(gs[0, 2])
        axp2.set_ylabel(r"${p_y}/m$")
        axp2.set_xlabel("steps")
        #axp2.set_ylim([0, max(fourvel[0,:,2])*2])

        axp3 = fig.add_subplot(gs[1, 2])
        axp3.set_ylabel(r"${p_z}/m$")
        axp3.set_xlabel("steps")
        #axp2.set_ylim([0, max(fourvel[0,:,2])*2])

        steps_axis = np.arange(0, (self.paths.shape[0]-1))
        for s in range(self.paths.shape[1]):
            axuu.plot(steps_axis, u_dot_u[:,s], f"C{s}-")#, marker=".")
            axp0.plot(steps_axis, fourvel[:,s,0], f"C{s}-")
            axp1.plot(steps_axis, fourvel[:,s,1], f"C{s}-")
            axp2.plot(steps_axis, fourvel[:,s,2], f"C{s}-")
            axp3.plot(steps_axis, fourvel[:,s,3], f"C{s}-")
            axtau.plot(steps_axis, proper_time_step[:,s], f"C{s}-")

        plt.tight_layout()
        plt.savefig(os.path.dirname(__file__)+"/Output/"+self.name+"_diagnostics.png")
        plt.close()

def smart_newton(func, guess, target=None):
    def derivative(func, value): return (func(value+0.0001)-func(value-0.0001))/0.0002
    if target is None:
        target = 0
    dif = func(guess) - target
    grad = derivative(func, guess)
    n = 0
    tolerance = 1e-6
    i = 0
    while dif >= tolerance:
        guess -= (dif/grad)*(0.0001**(i/1000))
        dif = func(guess) - target
        grad = derivative(func, guess)
        i += 1
        if i == 10000:
            print("failed to converge")
            break
    return guess

"""
Lagrangians
"""

def rel_free_lagrangian(pos1, pos2, s):
    masses = np.array([4, 5])
    mass = masses[s]
    position = (pos1+pos2)/2
    return (-1 * mass*mass)

def class_free_lagrangian(pos1, pos2, s):
    dims = len(pos1)
    masses = np.array([4, 5])
    mass = masses[s]
    velocity_squared = 0
    for i in range(dims - 1):
        velocity_squared += ((pos2[i+1] - pos1[i+1]) / (pos2[0]-pos1[0]))**2
    lagrangian = 0.5 * mass * velocity_squared
    return lagrangian

def class_grav_lagrangian(pos1, pos2, s):
    dims = len(pos1)
    masses = np.array([4, 5])
    mass = masses[s]
    velocity_squared = 0
    r = 0
    M = 0.5
    for i in range(dims - 1):
        velocity_squared += ((pos2[i+1] - pos1[i+1]) / (pos2[0]-pos1[0]))**2
        r += ((pos1[i+1]+pos2[i+1])/2)**2
    r = np.sqrt(r)
    lagrangian = 0.5 * mass * velocity_squared + (M*mass)/r
    return lagrangian

"""
Metrics
"""

def classical_metric(position):
    dims = len(position)
    metric = np.zeros([dims, dims])
    metric[0, 0] = 1
    return metric

def minkowski_metric(position):
    dims = len(position)
    metric = np.zeros([dims, dims])
    for i in range(dims - 1):
        metric[i+1, i+1] = -1
    metric[0, 0] = 1
    """
    metric = np.array([[1, 0, 0, 0]
                      ,[0,-1, 0, 0]
                      ,[0, 0,-1, 0]
                      ,[0, 0, 0,-1]])
    """
    return metric

def schwarzschild_metric(position):
    dims = len(position)
    M = 0.5
    r = 0
    for i in range(dims-1):
        r += position[i+1]*position[i+1]
    R = 2*M
    r = np.sqrt(r)
    spacial_expression = (1+(R/(4*r)))
    temporal_expression = (1-(R/(4*r)))
    metric = np.zeros([dims, dims])
    for i in range(dims-1):
        metric[i+1, i+1] = -(spacial_expression**4)
    metric[0, 0] = (temporal_expression/spacial_expression)**2
    """
    metric = np.array([[ ((1-(R/(4*r)))/(1+(R/(4*r))))**2, 0, 0, 0]
                      ,[0, -(1+(R/(4*r)))**4, 0, 0]
                      ,[0, 0, -(1+(R/(4*r)))**4, 0]
                      ,[0, 0, 0, -(1+(R/(4*r)))**4]])
    """
    return metric

def kerr_metric(position):
    [t, x, y, z] = position
    Gm = 0.5 # schwarz rad = 2M
    a = 0.9
    def func(r): return (((x*x+y*y)/(r*r+a*a))+((z*z)/(r*r))-1)
    r = smart_newton(func, np.sqrt(x*x+y*y+z*z))
    f = ((2*Gm*r*r*r)/((r**4)+(a*a*z*z)))
    k = np.array([1, (r*x+a*y)/(r*r+a*a), (r*y-a*x)/(r*r+a*a), (z/r)])
    nu = np.array([[1, 0, 0, 0]
                  ,[0,-1, 0, 0]
                  ,[0, 0,-1, 0]
                  ,[0, 0, 0,-1]])
    metric = np.zeros([4, 4])
    for i in range(4):
        for j in range(4):
            metric[i,j] = nu[i,j]-f*k[i]*k[j]
    return metric


