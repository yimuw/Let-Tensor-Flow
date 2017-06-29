import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class VehicleModel:
    """
    Naive vehicle model for slam
    States: (x, y)
    """

    def __init__(self, state0, motion_commands):
        """
        :param state0: Initial state
        :param motion_commands: list of motion command
        """
        # just x, y
        self.state_odometry = state0.copy()
        self.state_gt = state0.copy()

        # A list of (dx, dy)
        self.motion_noise_std = 0.1
        # The graph slam cost use l2 loss, which assume gaussian noise.
        # motion bias should break the current slam assumption
        self.motion_noise_bias = np.array([0.1, 0.])
        self.motion_commands = list(motion_commands)

    def __command_noise(self):
        """
        Command noise
        :return: command noise, ndarray of size (2,)
        """
        noise = np.random.randn(2) * self.motion_noise_std + self.motion_noise_bias
        return noise

    def compute_gt_trajectory(self):
        """
        Generate ground truth trajectory (trajectory with motion noise and motion command)
        :return: generator of (x, y)
        """
        # first state not noise
        yield self.state_gt.copy()
        for command in self.motion_commands:
            self.state_gt += command + self.__command_noise()
            yield self.state_gt.copy()

    def compute_odometry_trajectory(self):
        """
        Generate trajectory by motion command
        :return: generator of (x, y)
        """
        yield self.state_odometry.copy()
        for command in self.motion_commands:
            self.state_odometry += command
            yield self.state_odometry.copy()


class LandMarkObservation:
    """
    Landmark observation in vehicle coordinate
    """

    def __init__(self, landmark_relative_vector, landmark_idx):
        """
        :param landmark_relative_vector: relative position of landmark to vehicle
        :param landmark_idx: idx of the landmark
        """
        self.landmark_relative_vector = landmark_relative_vector
        self.idx = landmark_idx

    def __repr__(self):
        """
        :return: string repr
        """
        return 'landmark offset:{}  landmark idx:{}'.format(self.landmark_relative_vector, self.idx)


class ObservationModel:
    def __init__(self):
        """
        Just init...
        """
        # Assume gaussian noise
        self.observation_noise_std = 0.05

    def observe(self, noisy_gt_states, landmarks):
        """
        Given gt vehicle states, and landmarks, compute expected observations of landmarks
        :param noisy_gt_states: list of ground truth vehicle state
        :param landmarks: list of landmarks
        :return: list of list of observations
        """
        # Only observe landmark within
        MAX_OBSER_DISTANCE = 2.0

        for state_xy in noisy_gt_states:
            observations = []
            for landmark_idx, landmark_location in enumerate(landmarks):
                if np.linalg.norm(state_xy - landmark_location) < MAX_OBSER_DISTANCE:
                    # Compute expected observation
                    landmark_relative_vector = landmark_location - state_xy
                    landmark_relative_vector += self.__observation_noise()
                    observations.append(LandMarkObservation(landmark_relative_vector.astype('float32'), landmark_idx))
            yield observations

    def __observation_noise(self):
        """
        Observation noise
        :return: (noise_x, noise_y)
        """
        return np.random.randn(2) * self.observation_noise_std


class GraphSlam:
    """
    Graph Slam Algorithm
    Explanation:
        Probabilistic Robotics page: 338
    Next step:
        Add submap offset to cost
        Probabilistic Robotics page: 373
    """

    def __init__(self):
        pass

    def construct_graph(self, odometry_states, motion_commands, observations_per_state):
        """
        Construct tensorflow graph
        :param odometry_states: Vehicle states given by odometry
        :param motion_commands: Motion commands given to vehicle
        :param observations_per_state: Observations for each vehicle state
        :return: 1. tf training op 2. tf loss variable 3. tf vehicle state variables 4. tf landmark variables
        """
        assert len(odometry_states) == len(motion_commands) + 1

        # Init state tf variables
        num_state = len(odometry_states)
        all_state_vars = [tf.Variable(s, name='state{}'.format(i)) for i, s in enumerate(odometry_states)]

        # States loss
        state0_var = all_state_vars[0]
        state0_odometry = odometry_states[0]
        # State variable should close to what odometry predict
        # your first state should be accurate
        FIRST_STATE_WEIGHT = 100
        state_loss = FIRST_STATE_WEIGHT * tf.square(state0_var - state0_odometry)

        # Estimate motion bias
        motion_bias_var = tf.Variable(np.array([0, 0.], dtype='float32'), name='motion_bias')

        # Construct state graph
        for i in range(1, num_state):
            previous_state = all_state_vars[i - 1]
            command = motion_commands[i - 1]
            this_state = all_state_vars[i]
            # State variable should close to what odometry predict. l2 loss assume gaussian noise
            # Where chain rule (aka back propagation) comes
            state_loss += tf.square(tf.subtract(previous_state + tf.constant(command) + motion_bias_var, this_state))

        # Init landmark by odometry information
        all_landmark_vars, landmark_idx_to_landmark_var_map = self.__init_landmark_tf_varibles(odometry_states,
                                                                                               observations_per_state)
        # Construct landmark graph
        landmark_loss = 0.
        for idx, (state_var, observations) in enumerate(zip(all_state_vars, observations_per_state)):
            for observation in observations:
                landmark_var = landmark_idx_to_landmark_var_map[observation.idx]
                predict_landmark = state_var + tf.constant(observation.landmark_relative_vector)
                # Landmark should close to what state predict
                landmark_loss += tf.square(predict_landmark - landmark_var)

        LANDMARK_LOSS_WEIGHT = 1.
        # The total loss
        loss = state_loss + LANDMARK_LOSS_WEIGHT * landmark_loss

        # Minimize it!
        loss = tf.reduce_mean(loss)
        # Hacky gradient descent :)
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
        training_operation = optimizer.minimize(loss)

        return training_operation, loss, all_state_vars, all_landmark_vars

    def graph_slam_slove(self, training_operation, loss, all_state_vars, all_landmark_vars,
                         odometry_states, noise_gt_states, gt_landmarks):
        """
        Solve the optimization problem
        :param training_operation: tf training op
        :param loss: tf loss variable
        :param all_state_vars: tf vehicle state variables
        :param all_landmark_vars: tf landmark variables
        :param odometry_states: Vehicle states given by odometry. Only for visualization
        :param noise_gt_states: Ground truth vehicle states. Only for visualization
        :param gt_landmarks: Only for visualization
        """
        assert len(odometry_states) == len(noise_gt_states)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.graph.finalize()

            NUM_ITER = 300
            for i in range(NUM_ITER):
                loss_val, _ = sess.run([loss, training_operation])
                print('iteration:{} loss: {:2.4f}'.format(i, loss_val))

                self.__plot_current_result(sess, all_state_vars, all_landmark_vars, odometry_states, noise_gt_states,
                                           gt_landmarks)

            # Print result
            print('Result:')
            all_trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            for var in all_trainable_vars:
                var_name = var.name
                var_value = sess.run(var)
                print(var_name, ' value: ', var_value)

    def __init_landmark_tf_varibles(self, odometry_states, observations_per_state):
        """
        Initialize landmark tf variables.
        TODO: this is hacky
        :param odometry_states: list of vehicle state given by odometry
        :param observations_per_state: list of list of observations
        :return: tf landmark variables
        """
        landmark_idx_to_init_location_map = {}
        for state, observations in zip(odometry_states, observations_per_state):
            for observation in observations:
                # Just overwrite
                landmark_idx_to_init_location_map[observation.idx] = state + observation.landmark_relative_vector
        all_landmark_vars = []
        landmark_idx_to_landmark_var_map = {}
        for landmark_idx, landmark_init_location in landmark_idx_to_init_location_map.items():
            landmark_var = tf.Variable(landmark_init_location, name='landmark{}'.format(landmark_idx))
            all_landmark_vars.append(landmark_var)
            landmark_idx_to_landmark_var_map[landmark_idx] = landmark_var

        return all_landmark_vars, landmark_idx_to_landmark_var_map

    def __plot_current_result(self, sess, all_state_vars, all_landmark_vars, odometry_states, noise_gt_states,
                              gt_landmarks):
        """
        Visualization
        """
        state_results = np.vstack(sess.run(all_state_vars))
        landmark_results = np.vstack(sess.run(all_landmark_vars))
        plt.plot(odometry_states[:, 0], odometry_states[:, 1], 'bo--', label='States from odometry')
        plt.plot(noise_gt_states[:, 0], noise_gt_states[:, 1], 'ro--', label='Ground truth States')
        plt.plot(state_results[:, 0], state_results[:, 1], 'go--', label='Current tf state variable')
        plt.scatter(landmark_results[:, 0], landmark_results[:, 1], marker='x', label='Current ft landmark variable')
        plt.scatter(gt_landmarks[:, 0], gt_landmarks[:, 1], marker='X', label='Ground truth Landmarks')
        plt.legend()
        plt.axis('equal')
        plt.draw()
        plt.pause(0.05)
        plt.clf()


def plot_generated_data(odometry_states, gt_states, observations_per_state, gt_landmarks):
    """
    Plot generated data for visualization
    :param odometry_states:
    :param gt_states:
    :param observations_per_state:
    :param gt_landmarks:
    :return:
    """
    plt.scatter(gt_landmarks[:, 0], gt_landmarks[:, 1], marker='X', label='Landmarks')
    plt.plot(odometry_states[:, 0], odometry_states[:, 1], 'bo--', label='States from odometry')
    plt.plot(gt_states[:, 0], gt_states[:, 1], 'ro--', label='Ground truth States')

    # TODO: Plot vectors
    line = None
    for noisy_state, observations in zip(gt_states, observations_per_state):
        for observation in observations:
            line = np.vstack([noisy_state, noisy_state + observation.landmark_relative_vector])
            plt.plot(line[:, 0], line[:, 1], 'y-')
    # Only for label
    plt.plot(line[:, 0], line[:, 1], 'y-', label='Observation vectors')

    plt.legend()
    plt.axis('equal')
    plt.show()


def main():
    """
    Better scope
    """
    # Loop
    INIT_STATE = np.array([0., 0.])
    MOTION_COMMANDS = np.array([
        [1, 0],
        [1, 1],
        [0, 1],
        [-1, 1],
        [-1, 0],
        [-1, -1],
        [0, -1],
        [0.5, -0.5]
    ], dtype='float32')

    GT_LANDMARKS = np.array([
        [0, -0.5],
        [1, 1.],
        [0, 1.5],
        [2, 2.5],
    ], dtype='float32')

    vehicle_model = VehicleModel(state0=INIT_STATE, motion_commands=MOTION_COMMANDS)

    # Generate fake data
    gt_states = np.vstack(vehicle_model.compute_gt_trajectory()).astype('float32')
    odometry_states = np.vstack(vehicle_model.compute_odometry_trajectory()).astype('float32')
    observations_per_state = list(ObservationModel().observe(gt_states, GT_LANDMARKS))

    plot_generated_data(odometry_states, gt_states, observations_per_state, GT_LANDMARKS)

    slam = GraphSlam()
    # Two function for visualization
    training_operation, loss, all_state_vars, all_landmark_vars = slam.construct_graph(odometry_states, MOTION_COMMANDS,
                                                                                       observations_per_state)
    slam.graph_slam_slove(training_operation, loss, all_state_vars, all_landmark_vars, odometry_states, gt_states,
                          GT_LANDMARKS)


if __name__ == '__main__':
    main()
