import numpy as np
import numdifftools as nd
import matplotlib.pyplot as plt

def parse_config(file='config.yaml'):
    '''
    Parse the configuration args.
    '''
    import yaml
    from yaml.loader import SafeLoader
    with open(file) as f:
        config = yaml.load(f, Loader=SafeLoader)
    
    return config
   

class Particle():
    '''
    The particle to be suspended.
    '''
    def __init__(self, center_pos, V, rho, v_sound):
        # The position of the particle.
        self.pos = np.array(center_pos)
        # The volume of the particle.
        self.volume = V
        # The density of the particle.
        self.density = rho
        # The sonic in such material.
        self.sonic = v_sound
    
    def move_to(self, target):
        '''
        Move the particle to target place.
        '''
        # target should be a np array.
        self.pos = target
        return self
   
class TransPlain():
    '''
    The transducer plain. We abstract the transducer as a point source.
    '''     
    def __init__(self, lt_pos, n_x, n_y, step_x, step_y, frequency, p0, rho, v_sound):
        # lt_pos should be the left-top position <tuple(x, y, z)> of the plain x-y.
        self.lt_pos = np.array([lt_pos])
        # The shape of the plain.
        self.shape = np.array([n_x, n_y])
        # The step of each direction.
        self.step = np.array([step_x, step_y])
        # The direction of the plain is defined as z+. 
        self.norm_vec = np.array([0, 0, 1])
        # The frequency of the transducers.
        self.freq = frequency
        # The constant defined by the power of transducers.
        self.p0 = p0
        # The density of 'air' material.
        self.density = rho
        # The sonic of 'air' material.
        self.sonic = v_sound
        
        # The phase of each transducer, we will set it all to 0 here. Change the logic here and the system should will be OK for phase-different situation.
        self.trans_phase = np.zeros(shape=[n_x, n_y])
        
        # The position of each transducer.
        self.trans_pos = np.array(
                [ np.array(lt_pos) + [i*step_x, j*step_y, 0] for i in range(n_x) for j in range(n_y)]
            )
        self.trans_pos = self.trans_pos.reshape([n_x, n_y, 3])

        # The processed params.
        self.k = self.freq / self.sonic        
        
    
    def cal_angle(self, p:Particle):
        '''
        Function to calculate the angle between particle and the transducers plain.
        '''
        tiny_num = 0.0000000001 # In case of div/0.
        ang = np.arctan(
                np.linalg.norm(self.trans_pos[:, :, 2::1] - p.pos[2::1], axis=2) / 
                (np.linalg.norm(self.trans_pos[:, :, :2] - p.pos[:2], axis=2) + tiny_num)
            )
        return ang
    
    def cal_distance(self, p:Particle):
        '''
        Function to calculate the angle between particle and the transducers plain.
        '''
        dis = np.linalg.norm(self.trans_pos[:, :] - p.pos, axis=2)
        return dis
    
class USSystem():
    '''
    The ultrasonic suspension system parameters including transducer array's and particle'.
    The system only has one particle now.
    '''
    def __init__(self, trans_plain:TransPlain, particle:Particle):
        # The only particle in the system.
        self.p = particle
        # The transducers plain in the system. It should be a numpy array.
        self.tr_pl = trans_plain

        # Generate the relationship between particle and transducers.
        self.theta_func = lambda particle=self.p : self.tr_pl.cal_angle(particle)
        self.dis_func = lambda particle=self.p : self.tr_pl.cal_distance(particle)
        
        self.generate_complex_pressure_func()
        self.generate_GkU_func()
    
    def generate_complex_pressure_func(self):
        '''
        Generate the sum of all complex pressure function.
        '''
        # The complex pressure function of particles position.
        self.cp_func = lambda particle=self.p : self.tr_pl.p0 * np.i0(0) * self.tr_pl.k * np.sin(self.theta_func(particle)) / self.dis_func(particle) * ( np.e ** ((self.tr_pl.k * self.dis_func(particle) + self.tr_pl.trans_phase)* 1j) )
        # The total complex pressure.
        self.cp_all_func = lambda particle=self.p : self.cp_func(particle).sum(axis=(0, 1))
        
    def generate_GkU_func(self):
        k1 = lambda particle=self.p : (1/4) * particle.volume * (
            1 / ( self.tr_pl.density * (self.tr_pl.sonic ** 2) ) -
            1 / ( particle.density * (particle.sonic ** 2) )
        )
        k2 = lambda particle=self.p : (3/4) * particle.volume * (
            (self.tr_pl.density - particle.density) / 
            ( (self.tr_pl.freq ** 2) * self.tr_pl.density * (self.tr_pl.density + 2 * particle.density) )
        )
        self.GkU_func = lambda particle=self.p : (k1(particle) - k2(particle)) * ( np.linalg.norm(self.cp_all_func(particle)) ** 2 )
    
    def get_acoustic_radiation(self, pos):
        f = lambda pos : self.GkU_func(self.p.move_to(pos))
        res = nd.Gradient(f)(pos)
        return res
    
    def scan_space(self, space_shape, line_density):
        '''
        Scan the space and calculate the results.
        Then draw the 3d figure.
        '''
        scan_res = []
        offset = np.array([-space_shape[0]/2, -space_shape[1]/2, -space_shape[2]/2])
        n = np.floor(np.array(space_shape) / line_density).astype('int')
        for i in range(n[0]):
            for j in range(n[1]):
                for k in range(n[2]):
                    pos = offset + np.array([i, j, k]) * line_density
                    arf = self.get_acoustic_radiation(pos)
                    scan_res.append(np.row_stack((pos, arf)))
        scan_res = np.array(scan_res)
        scan_res[:, 1, 0] = np.linalg.norm(scan_res[:, 1, :], axis=1)
        scan_res[:, 1, 0] /= scan_res[:, 1, 0].max()
        
        # Creating dataset
        x = scan_res[:, 0, 0].flatten() 
        y = scan_res[:, 0, 1].flatten() 
        z = scan_res[:, 0, 2].flatten()
        # Change the color distribution if you want.
        colors = [(
                0.8 - 0.8 * scan_res[i, 1, 0] ** 0.75,
                0.8 - 0.8 * scan_res[i, 1, 0] ** 0.75,
                0.8 - 0.8 * scan_res[i, 1, 0] ** 0.75,
                0.4 + 0.6 * scan_res[i, 1, 0]
            ) for i in range(scan_res.shape[0])]
        
        
        # Creating figure
        fig = plt.figure(figsize = (10, 7))
        ax = plt.axes(projection ="3d")
        
        # Creating plot
        ax.scatter3D(x, y, z, c=colors)
        ax.scatter3D(self.tr_pl.trans_pos[..., 0].flatten(), self.tr_pl.trans_pos[...,1].flatten(), self.tr_pl.trans_pos[...,2].flatten(), c='red')
        ax.set_zlabel('z')
        ax.set_ylabel('y')
        ax.set_xlabel('x')
        
        plt.title("sim")
        
        # show plot
        plt.show()
        
def simulate():
    '''
    The main part of the simulator.
    '''
    # Get args.
    conf = parse_config()
    # Construct transducer plain object.
    c = conf['TransPlain']
    tr_pl = TransPlain( c['lt_pos'], c['n_x'], c['n_y'], c['step_x'], c['step_y'], c['frequency'], c['p0'], c['rho'], c['v_sound'] )
    
    # Construct particle object.
    c = conf['Particle']
    particle = Particle( c['center_pos'], c['V'], c['rho'], c['v_sound'] )
    
    # Construct USSystem object.
    uss = USSystem(tr_pl, particle)
    
    # Do space scanning and show figure.
    c = conf['ScanSpace']
    uss.scan_space(c['space_shape'], c['line_density'])
    

if __name__ == "__main__":
    simulate()