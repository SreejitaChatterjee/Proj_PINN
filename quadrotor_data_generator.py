import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class QuadrotorSimulator:
    def __init__(self):
        # Physical parameters from MATLAB code
        self.Jxx = 6.86e-5
        self.Jyy = 9.2e-5  
        self.Jzz = 1.366e-4
        self.m = 0.068
        self.kt = 0.01
        self.kq = 7.8263e-4
        self.b = 0.062/np.sqrt(2)
        self.g = 9.81
        
        # Controller parameters
        self.k2, self.k1, self.ki = 0.1, 1.0, 0.004
        self.k21, self.k11, self.ki1 = 0.1, 1.0, 0.004
        self.k22, self.k12, self.ki2 = 0.1, 1.0, 0.004
        self.kv, self.kz1, self.kz2 = -1.0, 2.0, 0.15
        
        # Control constraints
        self.Tmax = 2.0 * self.m * self.g
        self.nmax = np.sqrt(self.Tmax/(4*self.kt))
        self.txymax = (self.Tmax/4)*2*self.b
        self.tzmax = 2*self.kq*self.nmax**2
        self.th = 0.0000001
        
        # Reference values
        self.phir = 10*np.pi/180
        self.thetar = -5*np.pi/180
        self.psir = 5*np.pi/180
        self.zr = -5.0
        
    def generate_data(self, dt=0.001, tend=5.0, noise_level=0.01):
        """Generate quadrotor flight data with timestamps"""
        
        # Initialize state
        state = np.zeros(15)  # [x,y,z,u,v,w,p,q,r,phi,theta,psi,sump,sumt,sumpsi,sumz]
        state = np.append(state, 0)  # sumz
        
        times = np.arange(0, tend, dt)
        n_steps = len(times)
        
        # Storage arrays
        data = []
        
        for i, t in enumerate(times):
            x, y, z, u, v, w, p, q, r, phi, theta, psi, sump, sumt, sumpsi, sumz = state
            
            # Controller calculations
            sump += (self.phir - phi)
            pr = self.k1*(self.phir - phi) + self.ki*sump*dt
            tx = self.k2*(pr - p)
            tx = np.clip(tx, -self.txymax, self.txymax)
            if abs(tx) < self.th: tx = 0
            
            sumt += (self.thetar - theta)
            qr = self.k11*(self.thetar - theta) + self.ki1*sumt*dt
            ty = self.k21*(qr - q)
            ty = np.clip(ty, -self.txymax, self.txymax)
            if abs(ty) < self.th: ty = 0
            
            sumpsi += (self.psir - psi)
            rref = self.k12*(self.psir - psi) + self.ki2*sumpsi*dt
            tz = self.k22*(rref - r)
            tz = np.clip(tz, -self.tzmax, self.tzmax)
            if abs(tz) < self.th: tz = 0
            
            sumz += (self.zr - z)
            vzr = self.kz1*(self.zr - z) + self.kz2*sumz*dt
            T = self.kv*(vzr - (-w))  # zdot = -w in body frame
            T = np.clip(T, 0.1*self.m*self.g, self.Tmax)
            
            # Add noise to inputs
            if noise_level > 0:
                T += np.random.normal(0, noise_level * T)
                tx += np.random.normal(0, noise_level * abs(tx + 1e-6))
                ty += np.random.normal(0, noise_level * abs(ty + 1e-6))
                tz += np.random.normal(0, noise_level * abs(tz + 1e-6))
            
            # Store current state and controls
            row = {
                'timestamp': t,
                'thrust': T,
                'z': z,
                'torque_x': tx,
                'torque_y': ty, 
                'torque_z': tz,
                'roll': phi,
                'pitch': theta,
                'yaw': psi,
                'p': p,
                'q': q,
                'r': r,
                'vx': u,
                'vy': v,
                'vz': w,
                'mass': self.m,
                'inertia_xx': self.Jxx,
                'inertia_yy': self.Jyy,
                'inertia_zz': self.Jzz
            }
            
            # Dynamics integration
            t1 = (self.Jyy - self.Jzz)/self.Jxx
            t2 = (self.Jzz - self.Jxx)/self.Jyy  
            t3 = (self.Jxx - self.Jyy)/self.Jzz
            
            # Rotational dynamics
            pdot = t1*q*r + tx/self.Jxx - 2*p
            qdot = t2*p*r + ty/self.Jyy - 2*q
            rdot = t3*p*q + tz/self.Jzz - 2*r
            
            p += pdot*dt
            q += qdot*dt
            r += rdot*dt
            
            # Euler angle dynamics
            phidot = p + np.sin(phi)*np.tan(theta)*q + np.cos(phi)*np.tan(theta)*r
            thetadot = np.cos(phi)*q - np.sin(phi)*r
            psidot = np.sin(phi)*q/np.cos(theta) + np.cos(phi)*r/np.cos(theta)
            
            phi += phidot*dt
            theta += thetadot*dt
            psi += psidot*dt
            
            # Angle wrapping
            phi = self.wrap_angle(phi)
            theta = self.wrap_angle(theta)
            psi = self.wrap_angle(psi)
            
            # Translational dynamics
            fz = -T
            udot = r*v - q*w + 0/self.m - self.g*np.sin(theta) - 0.1*u
            vdot = p*w - r*u + 0/self.m + self.g*np.cos(theta)*np.sin(phi) - 0.1*v
            wdot = q*u - p*v + fz/self.m + self.g*np.cos(theta)*np.cos(phi) - 0.1*w
            
            u += udot*dt
            v += vdot*dt
            w += wdot*dt
            
            # Position dynamics
            xdot = (np.cos(psi)*np.cos(theta))*u + (np.cos(psi)*np.sin(theta)*np.sin(phi) - np.sin(psi)*np.cos(phi))*v + (np.sin(psi)*np.sin(phi) + np.cos(psi)*np.sin(theta)*np.cos(phi))*w
            ydot = (np.sin(psi)*np.cos(theta))*u + (np.cos(psi)*np.cos(phi) + np.sin(psi)*np.sin(theta)*np.sin(phi))*v + (np.sin(psi)*np.sin(theta)*np.cos(phi) - np.cos(psi)*np.sin(phi))*w
            zdot = -1*(np.sin(theta)*u - np.cos(theta)*np.sin(phi)*v - np.cos(theta)*np.cos(phi)*w)
            
            x += xdot*dt
            y += ydot*dt
            z += zdot*dt
            
            # Stop if ground hit
            if z > 0:
                break
                
            # Update state
            state = np.array([x, y, z, u, v, w, p, q, r, phi, theta, psi, sump, sumt, sumpsi, sumz])
            data.append(row)
            
        return pd.DataFrame(data)
    
    def wrap_angle(self, angle):
        """Wrap angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2*np.pi
        while angle < -np.pi:
            angle += 2*np.pi
        return angle

# Generate training data
if __name__ == "__main__":
    simulator = QuadrotorSimulator()
    
    # Generate multiple trajectories with different noise levels
    datasets = []
    
    for i in range(10):  # Generate 10 different trajectories
        noise_level = np.random.uniform(0.01, 0.05)
        df = simulator.generate_data(dt=0.001, tend=5.0, noise_level=noise_level)
        df['trajectory_id'] = i
        datasets.append(df)
    
    # Combine all datasets
    full_dataset = pd.concat(datasets, ignore_index=True)
    
    # Save to CSV
    full_dataset.to_csv('quadrotor_training_data.csv', index=False)
    print(f"Generated {len(full_dataset)} data points across {len(datasets)} trajectories")
    print(f"Columns: {list(full_dataset.columns)}")
    print(f"Data shape: {full_dataset.shape}")