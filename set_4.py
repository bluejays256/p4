import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

############### PROBLEM 1 ################
# part a #

def logistic_map(x, r):
    return r * x * (1 - x)

def logistic_derivative(x, r):
    return r * (1 - 2 * x)

def fixed_points(r):
    """Returns the fixed points x0 and x1 for a given r."""
    x0 = 0
    x1 = (r - 1) / r if r != 0 else None  # Avoid division by zero
    return x0, x1

def stability(r):
    """Determines stability of fixed points for a given r."""
    x0, x1 = fixed_points(r)
    
    stability_x0 = abs(logistic_derivative(x0, r)) < 1
    stability_x1 = abs(logistic_derivative(x1, r)) < 1 if x1 is not None else False
    
    return stability_x0, stability_x1

r_values = np.linspace(1, 4, 100)
stable_x1 = []
unstable_x1 = []

for r in r_values:
    _, stable_x1_flag = stability(r)
    if stable_x1_flag:
        stable_x1.append((r, (r - 1) / r))
    else:
        unstable_x1.append((r, (r - 1) / r))

stable_x1 = np.array(stable_x1)
unstable_x1 = np.array(unstable_x1)

plt.figure(figsize=(8, 5))
if len(stable_x1) > 0:
    plt.scatter(stable_x1[:, 0], stable_x1[:, 1], color='blue', label='Stable Fixed Point')
if len(unstable_x1) > 0:
    plt.scatter(unstable_x1[:, 0], unstable_x1[:, 1], color='red', label='Unstable Fixed Point')

plt.axvline(3, color='black', linestyle='--', label='Bifurcation at r=3')
plt.xlabel("r")
plt.ylabel("Fixed Point x")
plt.title("Stability of Fixed Points in the Logistic Map")
plt.legend()
plt.show()

# part b # 

import numpy as np
import matplotlib.pyplot as plt

def logistic_map(x, r):
    return r * x * (1 - x)

def logistic_derivative(x, r):
    return r * (1 - 2 * x)

def fixed_points(r):
    """Returns the fixed points x0 and x1 for a given r."""
    x0 = 0
    x1 = (r - 1) / r if r != 0 else None  # Avoid division by zero
    return x0, x1

def stability(r):
    """Determines stability of fixed points for a given r."""
    x0, x1 = fixed_points(r)
    
    stability_x0 = abs(logistic_derivative(x0, r)) < 1
    stability_x1 = abs(logistic_derivative(x1, r)) < 1 if x1 is not None else False
    
    return stability_x0, stability_x1

r_values = np.linspace(1, 4, 100)
stable_x1 = []
unstable_x1 = []

for r in r_values:
    _, stable_x1_flag = stability(r)
    if stable_x1_flag:
        stable_x1.append((r, (r - 1) / r))
    else:
        unstable_x1.append((r, (r - 1) / r))

stable_x1 = np.array(stable_x1)
unstable_x1 = np.array(unstable_x1)

plt.figure(figsize=(8, 5))
if len(stable_x1) > 0:
    plt.scatter(stable_x1[:, 0], stable_x1[:, 1], color='blue', label='Stable Fixed Point')
if len(unstable_x1) > 0:
    plt.scatter(unstable_x1[:, 0], unstable_x1[:, 1], color='red', label='Unstable Fixed Point')

plt.axvline(3, color='black', linestyle='--', label='Bifurcation at r=3')
plt.xlabel("r")
plt.ylabel("Fixed Point x")
plt.title("Stability of Fixed Points in the Logistic Map")
plt.legend()
plt.show()

# Iterate the logistic map for specific r values with separate plots
r_list = [2, 3, 3.5, 3.8, 4.0]
x0 = 0.2
n_iterations = 100

for r in r_list:
    x_values = [x0]
    x = x0
    for _ in range(n_iterations):
        x = logistic_map(x, r)
        x_values.append(x)
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(n_iterations + 1), x_values, label=f'r = {r}', color='blue')
    plt.xlabel("Iterations")
    plt.ylabel("x value")
    plt.title(f"Logistic Map Iterations for r = {r}")
    plt.legend()
    plt.show()

    #Part c #
r_list = [2, 3, 3.5, 3.8, 4.0]
x0_values = [0.1, 0.3, 0.5]
n_iterations = 100

for r in r_list:
    plt.figure(figsize=(8, 5))
    for x0 in x0_values:
        x_values = [x0]
        x = x0
        for _ in range(n_iterations):
            x = logistic_map(x, r)
            x_values.append(x)
        plt.plot(range(n_iterations + 1), x_values, label=f'x_0 = {x0}')
    
    plt.xlabel("Iterations")
    plt.ylabel("x value")
    plt.title(f"Logistic Map Iterations for r = {r}")
    plt.legend()
    plt.show()

#Part d #

def logistic_map(x, r):
    return r * x * (1 - x)

def logistic_derivative(x, r):
    return r * (1 - 2 * x)

def fixed_points(r):
    """Returns the fixed points x0 and x1 for a given r."""
    x0 = 0
    x1 = (r - 1) / r if r != 0 else None  # Avoid division by zero
    return x0, x1

def stability(r):
    """Determines stability of fixed points for a given r."""
    x0, x1 = fixed_points(r)
    
    stability_x0 = abs(logistic_derivative(x0, r)) < 1
    stability_x1 = abs(logistic_derivative(x1, r)) < 1 if x1 is not None else False
    
    return stability_x0, stability_x1

r_values = np.linspace(1, 4, 100)
stable_x1 = []
unstable_x1 = []

for r in r_values:
    _, stable_x1_flag = stability(r)
    if stable_x1_flag:
        stable_x1.append((r, (r - 1) / r))
    else:
        unstable_x1.append((r, (r - 1) / r))

stable_x1 = np.array(stable_x1)
unstable_x1 = np.array(unstable_x1)

plt.figure(figsize=(8, 5))
if len(stable_x1) > 0:
    plt.scatter(stable_x1[:, 0], stable_x1[:, 1], color='blue', label='Stable Fixed Point')
if len(unstable_x1) > 0:
    plt.scatter(unstable_x1[:, 0], unstable_x1[:, 1], color='red', label='Unstable Fixed Point')

plt.axvline(3, color='black', linestyle='--', label='Bifurcation at r=3')
plt.xlabel("r")
plt.ylabel("Fixed Point x")
plt.title("Stability of Fixed Points in the Logistic Map")
plt.legend()
plt.show()

# Iterate the logistic map for specific r values and different initial conditions
r_list = [2, 3, 3.5, 3.8, 4.0]
x0_values = [0.1, 0.3, 0.5]
n_iterations = 100

for r in r_list:
    plt.figure(figsize=(8, 5))
    for x0 in x0_values:
        x_values = [x0]
        x = x0
        for _ in range(n_iterations):
            x = logistic_map(x, r)
            x_values.append(x)
        plt.plot(range(n_iterations + 1), x_values, label=f'x_0 = {x0}')
    
    plt.xlabel("Iterations")
    plt.ylabel("x value")
    plt.title(f"Logistic Map Iterations for r = {r}")
    plt.legend()
    plt.show()

# Generate bifurcation diagram
r_values = np.linspace(2.5, 4, 500)
n_iterations = 1000
last_n = 100
x0 = 0.2
bifurcation_r = []
bifurcation_x = []

for r in r_values:
    x = x0
    for _ in range(n_iterations):
        x = logistic_map(x, r)
        if _ >= n_iterations - last_n:  # Store only the last n values
            bifurcation_r.append(r)
            bifurcation_x.append(x)

plt.figure(figsize=(10, 6))
plt.scatter(bifurcation_r, bifurcation_x, s=0.1, color='black')
plt.xlabel("r")
plt.ylabel("x_n")
plt.title("Bifurcation Diagram of the Logistic Map")
plt.show()

############ PROBLEM 2  ##################
#Part a #

def julia_set(xmin, xmax, ymin, ymax, width, height, c, max_iter):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    
    img = np.zeros(Z.shape, dtype=int)
    
    for i in range(max_iter):
        mask = np.abs(Z) < 2
        img[mask] = i
        Z[mask] = Z[mask] ** 2 + c
    
    return img

# Parameters
xmin, xmax = -1.5, 1.5
ymin, ymax = -1, 1
width, height = 800, 800
c = complex(-0.7, 0.356)
max_iter = 256

# Generate Julia set
julia = julia_set(xmin, xmax, ymin, ymax, width, height, c, max_iter)

# Plot the Julia set
plt.figure(figsize=(8, 8))
plt.imshow(julia, extent=(xmin, xmax, ymin, ymax), cmap='inferno', origin='lower')
plt.colorbar(label="Iterations")
plt.title("Julia Set for c = -0.7 + 0.356i")
plt.xlabel("Re(x)")
plt.ylabel("Im(y)")
plt.show()




def julia_set(xmin, xmax, ymin, ymax, width, height, c, max_iter):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    
    img = np.zeros(Z.shape, dtype=int)
    
    for i in range(max_iter):
        mask = np.abs(Z) < 2
        img[mask] = i
        Z[mask] = Z[mask] ** 2 + c
    
    return img, X, Y

# Parameters
xmin, xmax = -1.5, 1.5
ymin, ymax = -1, 1
width, height = 800, 800
c = complex(-0.7, 0.356)
max_iter = 256

# Generate Julia set
julia, X, Y = julia_set(xmin, xmax, ymin, ymax, width, height, c, max_iter)

# Extract boundary points for convex hull
points = np.column_stack((X[julia < max_iter].flatten(), Y[julia < max_iter].flatten()))
convex_hull = ConvexHull(points)

# Plot the Julia set with convex hull
plt.figure(figsize=(8, 8))
plt.imshow(julia, extent=(xmin, xmax, ymin, ymax), cmap='inferno', origin='lower')
plt.colorbar(label="Iterations")
plt.title("Julia Set for c = -0.7 + 0.356i with Convex Hull")
plt.xlabel("Re(x)")
plt.ylabel("Im(y)")

# Plot convex hull
for simplex in convex_hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'r-', linewidth=1, color ='red')

plt.show()

# Box Counting Dimension Estimation
def box_counting_dimension(img, max_box_size=100):
    sizes = np.logspace(1, np.log2(max_box_size), num=20, base=2, dtype=int)
    counts = []
    
    for size in sizes:
        new_shape = (img.shape[0] // size, size, img.shape[1] // size, size)
        reduced = img[:new_shape[0] * size, :new_shape[2] * size].reshape(new_shape)
        count = np.count_nonzero(np.any(reduced < max_iter, axis=(1, 3)))
        counts.append(count)
    
    sizes = 1 / sizes
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return coeffs[0]

# Compute and print box-counting dimension
dim = box_counting_dimension(julia)
print(f"Estimated fractal dimension: {dim:.4f}")

### PROBLEM 3 ###########
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import cv2

# Lorenz system parameters
sigma = 10
rho = 48
beta = 3

def lorenz_system(state, t):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Time array
t = np.linspace(0, 12, 3000)

# Initial condition
initial_state = [1.0, 1.0, 1.0]

# Solve the Lorenz system
solution = scipy.integrate.odeint(lorenz_system, initial_state, t)
x, y, z = solution.T

# Create video
video_name = "lorenz_attractor.mp4"
fps = 30
frame_size = (800, 600)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(video_name, fourcc, fps, frame_size)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-20, 20])
ax.set_ylim([-30, 30])
ax.set_zlim([0, 50])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

for i in range(1, len(x)):
    ax.plot(x[:i], y[:i], z[:i], color='b', alpha=0.6)
    plt.savefig("frame.png", dpi=100)
    frame = cv2.imread("frame.png")
    frame = cv2.resize(frame, frame_size)
    video.write(frame)
    ax.cla()
    ax.set_xlim([-20, 20])
    ax.set_ylim([-30, 30])
    ax.set_zlim([0, 50])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

cv2.destroyAllWindows()
video.release()
print("Video saved as", video_name)
