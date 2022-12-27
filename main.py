# https://github.com/spmvg/nfoursid/blob/master/examples/Overview.ipynb
# https://nfoursid.readthedocs.io/en/latest/
# https://pypi.org/project/nfoursid/
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from nfoursid.kalman import Kalman
from nfoursid.nfoursid import NFourSID
from nfoursid.state_space import StateSpace

pd.set_option('display.max_columns', None)
np.random.seed(0)  # reproducable results

NUM_TRAINING_DATAPOINTS = 1000  # create a training-set by simulating a state-space model with this many datapoints
NUM_TEST_DATAPOINTS = 20  # same for the test-set
INPUT_DIM = 3
OUTPUT_DIM = 2
INTERNAL_STATE_DIM = 4  # actual order of the state-space model in the training- and test-set
NOISE_AMPLITUDE = .1  # add noise to the training- and test-set
FIGSIZE = 8

# define system matrices for the state-space model of the training- and test-set
A = np.array([
    [1,  .01,    0,   0],
    [0,    1,  .01,   0],
    [0,    0,    1, .02],
    [0, -.01,    0,   1],
]) / 1.01
B = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 1, 1],
]
) / 3
C = np.array([
    [1, 0, 1,  1],
    [0, 0, 1, -1],
])
D = np.array([
    [1, 0, 1],
    [0, 1, 0]
]) / 10

state_space = StateSpace(A, B, C, D)
for _ in range(NUM_TRAINING_DATAPOINTS):
    input_state = np.random.standard_normal((INPUT_DIM, 1))
    noise = np.random.standard_normal((OUTPUT_DIM, 1)) * NOISE_AMPLITUDE
    _ = state_space.step(input_state, noise)

figsize = (1.3 * FIGSIZE, FIGSIZE)
fig = plt.figure(figsize=figsize)
state_space.plot_input_output(fig)  # the state-space model can plot its inputs and outputs
fig.tight_layout()
plt.show()


nfoursid = NFourSID(
    state_space.to_dataframe(),  # the state-space model can summarize inputs and outputs as a dataframe
    output_columns=state_space.y_column_names,
    input_columns=state_space.u_column_names,
    num_block_rows=10
)
nfoursid.subspace_identification()
fig, ax = plt.subplots(figsize=figsize)
nfoursid.plot_eigenvalues(ax)
fig.tight_layout()
plt.show()

ORDER_OF_MODEL_TO_FIT = 4
state_space_identified, covariance_matrix = nfoursid.system_identification(
    rank=ORDER_OF_MODEL_TO_FIT
)

kalman = Kalman(state_space_identified, covariance_matrix)
state_space = StateSpace(A, B, C, D)  # new data for the test-set
for _ in range(NUM_TEST_DATAPOINTS):  # make a test-set
    input_state = np.random.standard_normal((INPUT_DIM, 1))
    noise = np.random.standard_normal((OUTPUT_DIM, 1)) * NOISE_AMPLITUDE
    
    y = state_space.step(input_state, noise)  # generate test-set
    _ = kalman.step(y, input_state)  # the Kalman filter sees the output and input, but not the actual internal state

fig = plt.figure(figsize=figsize)
kalman.plot_filtered(fig)
fig.tight_layout()
plt.show()

fig = plt.figure(figsize=figsize)
kalman.plot_predicted(fig)
fig.tight_layout()
plt.show()

kalman.to_dataframe()

