import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from collections import deque

class LorenzAttractor:
    def __init__(self, sigma=10.0, rho=28.0, beta=8.0/3.0, dt=0.02):
        """
        Initialize the Lorenz system with classic parameters for chaos.
        
        Parameters:
        - sigma, rho, beta: Lorenz parameters
        - dt: Time step for numerical integration (increased for faster animation)
        """
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.dt = dt
        
        # Initial conditions (slightly perturbed from origin)
        self.x, self.y, self.z = 1.0, 1.0, 1.0
        
        # Storage for trajectory points
        self.max_points = 2000  # Number of points to keep in trail
        self.trajectory = deque(maxlen=self.max_points)
        
        # Animation parameters
        self.angle = 0  # For rotating view
        
    def lorenz_derivatives(self, x, y, z):
        """
        Calculate the derivatives for the Lorenz system.
        """
        dx_dt = self.sigma * (y - x)
        dy_dt = x * (self.rho - z) - y
        dz_dt = x * y - self.beta * z
        return dx_dt, dy_dt, dz_dt
    
    def runge_kutta_step(self):
        """
        Perform one step of 4th-order Runge-Kutta integration.
        """
        # Current state
        x, y, z = self.x, self.y, self.z
        h = self.dt
        
        # k1
        k1_x, k1_y, k1_z = self.lorenz_derivatives(x, y, z)
        
        # k2
        k2_x, k2_y, k2_z = self.lorenz_derivatives(
            x + h*k1_x/2, y + h*k1_y/2, z + h*k1_z/2
        )
        
        # k3
        k3_x, k3_y, k3_z = self.lorenz_derivatives(
            x + h*k2_x/2, y + h*k2_y/2, z + h*k2_z/2
        )
        
        # k4
        k4_x, k4_y, k4_z = self.lorenz_derivatives(
            x + h*k3_x, y + h*k3_y, z + h*k3_z
        )
        
        # Update state
        self.x += h * (k1_x + 2*k2_x + 2*k3_x + k4_x) / 6
        self.y += h * (k1_y + 2*k2_y + 2*k3_y + k4_y) / 6
        self.z += h * (k1_z + 2*k2_z + 2*k3_z + k4_z) / 6
        
        # Store point in trajectory
        self.trajectory.append((self.x, self.y, self.z))
    
    def setup_plot(self):
        """
        Set up the 3D plot for animation.
        """
        self.fig = plt.figure(figsize=(12, 9))
        self.fig.patch.set_facecolor('black')
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('black')
        
        # Set axis limits
        self.ax.set_xlim((-25, 25))
        self.ax.set_ylim((-35, 35))
        self.ax.set_zlim((5, 55))
        
        # Style the axes
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X', color='white', fontsize=12)
        self.ax.set_ylabel('Y', color='white', fontsize=12)
        self.ax.set_zlabel('Z', color='white', fontsize=12)
        
        # Set tick colors
        self.ax.tick_params(colors='white')
        
        # Add title
        self.fig.suptitle('Lorenz Attractor - Deterministic Chaos\n"The Butterfly Effect"\nDesigned by siddharth pathak', 
                         color='white', fontsize=16, y=0.95)
        
        # Initialize empty line objects
        self.line, = self.ax.plot([], [], [], color='cyan', linewidth=0.8, alpha=0.8)
        self.current_point, = self.ax.plot([], [], [], 'o', color='red', markersize=6)
        
        # Add parameter text
        param_text = f'σ={self.sigma}, ρ={self.rho}, β={self.beta:.3f}'
        self.ax.text2D(0.02, 0.95, param_text, transform=self.ax.transAxes, 
                      color='yellow', fontsize=10, verticalalignment='top')
        
        return self.line, self.current_point
    
    def animate(self, frame):
        """
        Animation function called for each frame.
        """
        # Perform more integration steps per frame for faster motion
        for _ in range(5):
            self.runge_kutta_step()
        
        # Update rotating view angle
        self.angle += 0.5
        self.ax.view_init(elev=25, azim=self.angle)
        
        # Get trajectory points
        if len(self.trajectory) > 1:
            traj_array = np.array(self.trajectory)
            x_traj, y_traj, z_traj = traj_array[:, 0], traj_array[:, 1], traj_array[:, 2]
            
            # Create color gradient for the trail (newer points brighter)
            n_points = len(self.trajectory)
            colors = np.linspace(0.2, 1.0, n_points)
            
            # Clear previous plots
            self.ax.clear()
            
            # Redraw axes and labels
            self.ax.set_xlim((-25, 25))
            self.ax.set_ylim((-35, 35))
            self.ax.set_zlim((5, 55))
            self.ax.grid(True, alpha=0.3)
            self.ax.set_xlabel('X', color='white', fontsize=12)
            self.ax.set_ylabel('Y', color='white', fontsize=12)
            self.ax.set_zlabel('Z', color='white', fontsize=12)
            self.ax.tick_params(colors='white')
            self.ax.set_facecolor('black')
            
            # Plot trajectory with gradient effect
            for i in range(1, len(x_traj)):
                alpha = colors[i] * 0.8
                color_intensity = colors[i]
                self.ax.plot(x_traj[i-1:i+1], y_traj[i-1:i+1], z_traj[i-1:i+1], 
                           color=(0, color_intensity, 1), alpha=alpha, linewidth=1.0)
            
            # Plot current point
            self.ax.scatter([self.x], [self.y], [self.z], color='red', s=50, alpha=1.0)
            
            # Add parameter text
            param_text = f'σ={self.sigma}, ρ={self.rho}, β={self.beta:.3f}'
            self.ax.text2D(0.02, 0.95, param_text, transform=self.ax.transAxes, 
                          color='yellow', fontsize=10, verticalalignment='top')
            
            # Add point count
            point_text = f'Points: {len(self.trajectory)}'
            self.ax.text2D(0.02, 0.90, point_text, transform=self.ax.transAxes, 
                          color='cyan', fontsize=10, verticalalignment='top')
        
        return []
    
    def run_animation(self):
        """
        Start the animation loop.
        """
        self.setup_plot()
        
        # Create animation with faster frame rate
        anim = animation.FuncAnimation(
            self.fig, self.animate, frames=None, interval=30, blit=False, repeat=True
        )
        
        # Show plot
        plt.tight_layout()
        plt.show()
        
        return anim

def main():
    """
    Main function to run the Lorenz Attractor animation.
    """
    print("🦋 Starting Lorenz Attractor Animation...")
    print("   The strange attractor will emerge as chaos unfolds!")
    print("   Close the window to stop the animation.")
    print()
    print("📊 System Parameters:")
    print("   σ (sigma) = 10.0   - Prandtl number")
    print("   ρ (rho)   = 28.0   - Rayleigh number") 
    print("   β (beta)  = 8/3    - Geometric parameter")
    print()
    print("🎯 The trajectory will never repeat, yet remains bounded!")
    
    # Create and run the Lorenz attractor
    lorenz = LorenzAttractor()
    animation_obj = lorenz.run_animation()
    
    # Keep reference to animation to prevent garbage collection
    return animation_obj

if __name__ == "__main__":
    # Run the animation

    anim = main()
