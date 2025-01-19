import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plots
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2
import math

###############################
# 1. CIRCLE (BALL) DETECTION  #
###############################
def detect_targets(image):
    """
    Detect circular targets in the image using OpenCV's HoughCircles.

    Parameters:
        image (numpy.ndarray): Input image in BGR format.

    Returns:
        targets (list of tuples): Detected targets as (x, y) coordinates.
    """
    # Convert to grayscale if necessary
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Apply Gaussian blur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(
        blurred,              # Input image
        cv2.HOUGH_GRADIENT,   # Detection method
        dp=1,                 # Inverse ratio of accumulator resolution
        minDist=50,           # Minimum distance between detected centers
        param1=50,            # Higher threshold for Canny edge detector
        param2=30,            # Accumulator threshold for circle detection
        minRadius=20,         # Minimum circle radius
        maxRadius=100         # Maximum circle radius
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        targets = [(c[0], c[1]) for c in circles]
        return targets
    return []

###############################
# 2. SHOOTING METHOD FOR      #
#    TRAJECTORY CALCULATION   #
###############################
def shooting_method_2d(target, origin=(0.0, 0.0), g=9.81,
                       v0_range=(1.0, 100.0), theta_range_right=(5, 80),
                       theta_range_left=(100, 175),
                       steps_v0=200, steps_theta=200):
    """
    Find the initial velocity (v0) and angle (theta) to hit the target from the origin.

    Parameters:
        target (tuple): Target coordinates as (x_t, y_t).
        origin (tuple): Origin coordinates as (x0, y0).
        g (float): Acceleration due to gravity.
        v0_range (tuple): Range of initial velocities to search.
        theta_range_right (tuple): Launch angle range for targets to the right.
        theta_range_left (tuple): Launch angle range for targets to the left.
        steps_v0 (int): Number of steps for initial velocity.
        steps_theta (int): Number of steps for launch angle.

    Returns:
        best_v0 (float): Best initial velocity found.
        best_theta_deg (float): Best launch angle in degrees.
    """
    x_t, y_t = target
    x0, y0 = origin

    delta_x = x_t - x0
    delta_y = y_t - y0

    # Determine shooting direction based on delta_x
    if delta_x >= 0:
        theta_min, theta_max = theta_range_right
    else:
        theta_min, theta_max = theta_range_left

    best_v0, best_theta_deg = None, None
    min_dist = float('inf')

    v0_values = np.linspace(v0_range[0], v0_range[1], steps_v0)
    theta_values = np.linspace(theta_min, theta_max, steps_theta)

    for v0 in v0_values:
        for theta_deg in theta_values:
            theta = math.radians(theta_deg)
            cos_theta = math.cos(theta)
            sin_theta = math.sin(theta)
            if abs(cos_theta) < 1e-6:
                continue  # Avoid division by zero
            t_impact = delta_x / (v0 * cos_theta)
            if t_impact <= 0:
                continue  # Projectile does not reach target in positive time
            y_pred = y0 + v0 * sin_theta * t_impact - 0.5 * g * (t_impact ** 2)
            dist = abs(y_pred - y_t)
            if dist < min_dist:
                min_dist = dist
                best_v0 = v0
                best_theta_deg = theta_deg

    # Optional: Set a maximum acceptable distance
    max_acceptable_dist = 0.5  # Adjust as needed
    if min_dist > max_acceptable_dist:
        return None, None

    return best_v0, best_theta_deg

###############################
# 3. ANIMATION (SEQUENTIAL)   #
###############################
class Animator:
    def __init__(self, fig, ax, trajectories, origin, interval=30):
        """
        Initialize the Animator class.

        Args:
            fig (matplotlib.figure.Figure): The figure object.
            ax (matplotlib.axes.Axes): The axes to plot on.
            trajectories (list): List of (x_vals, y_vals) tuples for each trajectory.
            origin (tuple): Shooting origin coordinates.
            interval (int): Time interval between frames in milliseconds.
        """
        self.fig = fig
        self.ax = ax
        self.trajectories = trajectories
        self.origin = origin
        self.interval = interval

        # Initialize markers and lines
        self.projectile_marker, = ax.plot([], [], 'bo', markersize=6, label='Projectile')
        self.trajectory_line, = ax.plot([], [], 'g-', linewidth=2, label='Trajectory')

        # Initialize generator
        self.gen = self.frame_generator()

        # Initialize animation
        self.anim = FuncAnimation(
            fig, self.update, frames=self.gen,
            init_func=self.init,
            interval=self.interval,
            blit=True, repeat=False,
            save_count=1000  # Adjust as needed
        )

    def init(self):
        """Initialize the animation."""
        self.projectile_marker.set_data([], [])
        self.trajectory_line.set_data([], [])
        return self.projectile_marker, self.trajectory_line

    def frame_generator(self):
        """Generator that yields (x, y) for each projectile sequentially."""
        for traj in self.trajectories:
            if traj is None:
                continue
            x_vals, y_vals = traj
            for x, y in zip(x_vals, y_vals):
                yield (x, y)
            # Pause at the end of trajectory
            for _ in range(10):
                yield (x_vals[-1], y_vals[-1])

    def update(self, data):
        """Update function for the animation."""
        x, y = data
        # Update projectile marker position
        self.projectile_marker.set_data([x], [y])

        # Update trajectory line from origin to current position
        self.trajectory_line.set_data([self.origin[0], x], [self.origin[1], y])

        return self.projectile_marker, self.trajectory_line

def animate_projectiles_sequential(fig, ax, targets, solutions, origin, g=9.81, interval=30):
    """
    Animate projectiles moving to each target one by one.

    Args:
        fig (matplotlib.figure.Figure): The figure object.
        ax (matplotlib.axes.Axes): The axes to plot on.
        targets (list): List of target coordinates.
        solutions (list): List of (v0, theta) tuples for each target.
        origin (tuple): Shooting origin coordinates.
        g (float): Acceleration due to gravity.
        interval (int): Time interval between frames in milliseconds.
    """
    # Prepare trajectories
    trajectories = []
    for (v0, theta_deg), target in zip(solutions, targets):
        if v0 is None or theta_deg is None:
            trajectories.append(None)
            continue
        theta = math.radians(theta_deg)
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        x0, y0 = origin
        t_impact = (target[0] - x0) / (v0 * cos_theta)
        if t_impact <= 0:
            trajectories.append(None)
            continue
        t_vals = np.linspace(0, t_impact, 100)
        x_vals = x0 + v0 * cos_theta * t_vals
        y_vals = y0 + v0 * sin_theta * t_vals - 0.5 * g * (t_vals ** 2)
        trajectories.append((x_vals, y_vals))
        # Plot trajectory as dashed line
        ax.plot(x_vals, y_vals, linestyle='--', color='blue', alpha=0.5)

    # Initialize the Animator class and keep a reference to prevent garbage collection
    global animator  # Use a global variable to keep the animator alive
    animator = Animator(fig, ax, trajectories, origin, interval=interval)

###############################
# 4. MAIN EXECUTION           #
###############################
def main():
    print("======= Projectile Motion & Target Detection Program =======")

    # 1. Grid sizing
    try:
        grid_w = float(input("Enter grid width (e.g. 10): "))
        grid_h = float(input("Enter grid height (e.g. 10): "))
    except ValueError:
        print("Invalid numeric input. Using default grid 10x10.")
        grid_w, grid_h = 10.0, 10.0

    # 2. Image input for target detection
    use_image = input("Use image for target detection? (y/n): ").strip().lower().startswith('y')
    targets = []
    image_rgb = None

    if use_image:
        image_path = input("Enter image path (e.g. 'image.png'): ").strip()
        try:
            # Load image using OpenCV
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                print(f"Failed to load image '{image_path}'. Exiting.")
                return
            # Detect circles
            targets_circles = detect_targets(image_bgr)  # list of (x, y)
            if not targets_circles:
                print("No circles detected in the image. Exiting.")
                return
            print(f"Detected {len(targets_circles)} circle(s).")
            # Map to grid coordinates
            # Assuming image pixels correspond to grid dimensions
            # Need to map image coordinates to grid
            # Image dimensions
            img_h, img_w = image_bgr.shape[:2]
            # Map circles to grid
            targets = []
            for (x, y) in targets_circles:
                x_grid = (x / img_w) * grid_w
                y_grid = (1.0 - (y / img_h)) * grid_h  # invert Y
                targets.append((x_grid, y_grid))
            # For plotting, load image in RGB
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading or processing image: {e}")
            print("Falling back to manual target input.")
            use_image = False

    # 3. Manual input if not using image
    if not use_image:
        try:
            n_targets = int(input("How many targets to input manually? "))
            for i in range(n_targets):
                x_str = input(f"  Target {i+1} X coordinate on grid: ")
                y_str = input(f"  Target {i+1} Y coordinate on grid: ")
                x_val = float(x_str)
                y_val = float(y_str)
                targets.append((x_val, y_val))
        except ValueError:
            print("Invalid manual input. No valid targets.")
            return

    if not targets:
        print("No targets specified. Exiting.")
        return

    print("=== Detected / Entered Targets ===")
    for i, (x, y) in enumerate(targets, start=1):
        print(f"  Target #{i}: (x={x:.2f}, y={y:.2f})")

    # 4. Display image and set shooting origin via mouse click
    fig, ax = plt.subplots()

    if image_rgb is not None:
        # Display the image with extent mapped to grid size
        ax.imshow(image_rgb, extent=[0, grid_w, 0, grid_h])
    else:
        # Set grid limits
        ax.set_xlim(0, grid_w)
        ax.set_ylim(0, grid_h)

    ax.set_title("Click to set shooting origin")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Plot the targets as circles
    target_patches = []
    for i, (tx, ty) in enumerate(targets):
        circ = plt.Circle((tx, ty), 0.2, color='red', fill=False, linewidth=2, label=f"Target {i+1}" if i == 0 else "")
        ax.add_patch(circ)
        target_patches.append(circ)

    # Initialize origin marker
    origin_marker, = ax.plot([], [], 'go', markersize=8, label='Origin')

    # To store origin
    origin = [None, None]
    solutions = []

    def on_click(event):
        """
        Handle mouse click to set shooting origin and start animation.
        """
        # Ignore clicks outside the axes
        if not event.inaxes:
            return
        origin_x, origin_y = event.xdata, event.ydata
        origin[0], origin[1] = origin_x, origin_y
        print(f"Origin set to: ({origin_x:.2f}, {origin_y:.2f})")

        # Plot origin marker
        origin_marker.set_data([origin_x], [origin_y])  # Pass as lists to avoid RuntimeError
        fig.canvas.draw()

        # Compute shooting solutions for each target
        solutions.clear()
        for (tx, ty) in targets:
            v0, theta = shooting_method_2d(
                target=(tx, ty),
                origin=(origin_x, origin_y),
                g=9.81,
                v0_range=(1.0, 100.0),  # Increased upper limit for better coverage
                theta_range_right=(5, 80),
                theta_range_left=(100, 175),
                steps_v0=200,  # Increased resolution
                steps_theta=200
            )
            solutions.append((v0, theta))
            if v0 is not None and theta is not None:
                print(f"  Target at ({tx:.2f}, {ty:.2f}): v0={v0:.2f} m/s, theta={theta:.2f} degrees")
            else:
                print(f"  Target at ({tx:.2f}, {ty:.2f}): No solution found.")

        # Disconnect the click event to prevent multiple origins
        fig.canvas.mpl_disconnect(cid)

        # Start the sequential animation
        animate_projectiles_sequential(fig, ax, targets, solutions, origin, g=9.81, interval=30)

    # Connect the click event
    cid = fig.canvas.mpl_connect('button_press_event', on_click)

    # Show legend
    ax.legend(loc='upper right')
    plt.show()

if __name__ == "__main__":
    main()
