import numpy as np
from transforms3d import axangles
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

def expCoordToRotation(expCoords):
    angle = np.linalg.norm(expCoords)
    axis = expCoords / angle

    R = axangles.axangle2mat(axis, angle)
    return R

def rotationToExpCoords(R):
    axis, angle = axangles.mat2axangle(R)
    expCoords = axis * angle
    return expCoords

def relativeRotation(expA, expB):
    """
    Returns rotation from RotationA to RotationB
    expA is the exponential coordinates of A
    """
    Ra = expCoordToRotation(expA)
    Rb = expCoordToRotation(expB)
    Rab = np.dot(Ra.transpose(), Rb)
    return Rab

def angleDistance(expA, expB):
    R = relativeRotation(expA, expB)
    theta = np.arccos((np.trace(R) - 1)/2.0)
    return theta

def skewSymmetric(v):
    v = v.reshape(-1)
    v1 = v[0]
    v2 = v[1]
    v3 = v[2]

    V = np.array([[0, -v3, v2], [v3, 0, -v1], [-v2, v1, 0]])
    return V

def rotationDerivative(R):
    """ Returns derivative of RotationMatrix(expCoods) with respect to each coordinate
    """
    expCoords = rotationToExpCoords(R)
    angle = np.linalg.norm(expCoords)
    axis = expCoords / angle
    skew = skewSymmetric(axis)
    deriv = []
    identity = np.identity(3)
    for i, vi in enumerate(axis.reshape(-1)):
        ei = identity[:,i]
        deriv += [vi * np.cos(angle) * skew \
                + vi * np.sin(angle) * np.dot(skew, skew) \
                + np.sin(angle) / angle * skewSymmetric(ei - vi * axis)
                + (1-np.cos(angle))/angle \
                    * (np.dot(ei, axis.transpose()) + np.dot(axis, ei.transpose()) \
                            - 2*vi * np.dot(axis, axis.transpose()))]
    return deriv

def angleDistanceDerivative(expA, expB):
    from math import sqrt
    angle = angleDistance(expA, expB)
    derivOfRb = rotationDerivative(expCoordToRotation(expB))
    Ra = expCoordToRotation(expA)
    deriv = np.zeros(3)
    const = -1/(2.0* sqrt(1.0 - np.cos(angle)**2))
    for i, d in enumerate(derivOfRb):
        deriv[i] = np.trace(np.dot(Ra.transpose(), d)) * const
    return deriv

def optimize(target, x0):
    from scipy.optimize import minimize as minimize
    res = minimize(lambda x: angleDistance(target, x), x0,
            jac = lambda x: angleDistanceDerivative(target, x))
    return res

def draw(target, ax, fig, linestyle='-'):

    R = expCoordToRotation(target)
    V = [np.vstack([(0,0,0),r]) for r in R.transpose()]

    lines = []
    for v, c in zip(V, ["r", "g", "b"]):
        lines += [ax.plot(v[:,0], v[:,1], v[:,2], c=c, linestyle=linestyle)[0]]

    # draw axis of rotation
    lines += [ax.plot([0,target[0]], [0, target[1]], [0, target[2]], linestyle=linestyle, c='k')[0]]
    return fig, ax, lines

def animate(target, vectors, write=True):

    def update_lines(num, datalines, lines):
        dataline = datalines[num]
        for line, data in zip(lines, dataline):
            line.set_data(data[:,0], data[:,1])
            line.set_3d_properties(data[:,2])

        return lines

    def convertToLines(target):
        R = expCoordToRotation(target)
        V = [np.vstack([(0,0,0),r]) for r in R.transpose()]
        V += [np.vstack([(0,0,0), target])]
        return V

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    fig, ax, lines = draw(vectors[0], ax, fig, '-')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_aspect('equal')
    ax.set_xlim3d(-2,2)
    ax.set_ylim3d(-2,2)
    ax.set_zlim3d(-2,2)

    data = [convertToLines(v) for v in vectors]
    line_animation = animation.FuncAnimation(fig,
            update_lines, len(data), fargs=(data, lines), interval=50, blit=False,
            repeat=False)

    if write:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, bitrate=1800)
        line_animation.save("test.mp4", writer=writer)

    else:
        plt.show()

if __name__ == "__main__":
    axis1 = np.array([1,1,1])
    axis1 = axis1 / np.linalg.norm(axis1)
    expA = axis1 * 1e-3
    expB = np.array([1,1,1]) * np.pi/2

    angle =  angleDistance(expA, expB)

    x0 = expB
    target = expA
    stepsize = 0.01
    maxIteration = 500
    x0s = [np.copy(x0)]
    for i in range(maxIteration):
        deriv = angleDistanceDerivative(target, x0)
        x0 = x0 - deriv * stepsize
        x0s += [np.copy(x0)]

    animate(target, x0s, False)

