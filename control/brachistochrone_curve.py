import tensorflow as tf
import collections
import matplotlib.pyplot as plt

Point = collections.namedtuple('Point', ['x', 'y'])

VERBOSE = False
# Gravity
G = 9.8
PI = 3.1415926


class BrachistochroneCurve:
    """
    https://en.wikipedia.org/wiki/Brachistochrone_curve
    Solve this problem by optimization. Otherwise you need to study 4 years of math.
    TODO: test it :) not sure it is correct or not.
    """
    def __init__(self, start_point, end_point):
        assert start_point.x < end_point.x and start_point.y > end_point.y

        self.dx = 0.05
        self.num_vars = int((end_point.x - start_point.x) / self.dx) + 1
        self.dy = (end_point.y - start_point.y) / (self.num_vars - 1)

        self.Points = []
        # Set start point
        x = tf.Variable(start_point.x, trainable=False, name='px_start')
        y = tf.Variable(start_point.y, trainable=False, name='py_start')
        self.Points.append([x, y])

        # Init variables
        # ONLY y value for points are optimization variables.
        for i in range(1, self.num_vars - 1):
            x = tf.Variable(start_point.x + i * self.dx, trainable=False, name='px{}'.format(i))
            # TODO: Find a local minimal :)
            # y = tf.Variable(end_point.y, name='py{}'.format(i))
            y = tf.Variable(start_point.y + i * self.dy, name='py{}'.format(i))
            self.Points.append([x, y])
        # Set end point
        x = tf.Variable(end_point.x, trainable=False, name='px_end')
        y = tf.Variable(end_point.y, trainable=False, name='py_end')
        self.Points.append([x, y])

        # Acceleration for each line segment
        self.Accelerations = []
        for i in range(1, self.num_vars):
            self.Accelerations.append(tf.identity(self.compute_acceleration_line_direction(self.Points[i - 1], self.Points[i]),
                                                  name='acc{}'.format(i)))
        assert len(self.Accelerations) == self.num_vars - 1

        # Compute velocity, traveling time and distance for each line segment
        vel0 = tf.Variable(0., trainable=False, name='vel0')
        self.Velocities = [vel0]
        self.Times = []
        for i in range(1, self.num_vars):
            time, next_vel = self.compute_time_and_next_vel(self.Points[i - 1], self.Points[i],
                                                            self.Velocities[i - 1],
                                                            self.Accelerations[i - 1], is_first=i == 1)
            # Save those variables for debug
            self.Velocities.append(tf.identity(next_vel, name='vel{}'.format(i)))
            self.Times.append(tf.identity(time, name='time{}'.format(i)))

        assert len(self.Points) == self.num_vars
        assert len(self.Accelerations) == self.num_vars - 1
        assert len(self.Velocities) == self.num_vars
        assert len(self.Times) == self.num_vars - 1

    def compute_acceleration_line_direction(self, point, point_next):
        """
        Compute acceleration due to gravity along line direction
        :param point: first point of line
        :param point_next: second point of line
        :return: Acceleration
        """
        x, y = point
        x_next, y_next = point_next

        # Draw a picture, and you got it
        slop = tf.atan((y_next - y) / self.dx)
        acceleration = - G * tf.cos(PI / 2. - slop)

        tf.assert_less_equal(acceleration, G)
        tf.assert_greater_equal(acceleration, -G)

        return acceleration

    def compute_time_and_next_vel(self, point, point_next, vel, acc, is_first=False):
        """
        Give line segment, acceleration and initial speed, compute the total time traveled, and
        the speed when leaving the line segment.
        :param point: First point of the line
        :param point_next: Second point of the line
        :param vel: init velocity
        :param acc: acceleration
        :param is_first: If true, handle with quadratic model. o/w linear model
        :return: Time spend in the line segment, speed when leaving the line segment
        """
        x, y = point
        x_next, y_next = point_next

        distance = tf.sqrt(tf.reduce_mean(tf.square(x - x_next) +
                                          tf.square(y - y_next)))

        # This is mathematically hacky...
        # Use linear model for most of line segments
        # Only use quadratic dynamic model for first line segment, since it can handle velocity close to 0.
        if is_first:
            # Quadratic formula, 0.5 * acc * t^2 + vel * t - distance = 0
            # NOTE: Can't use quadratic model for all segments,
            # it makes the optimization numerical unstable. (try it by yourself)
            time = (-vel + tf.sqrt(vel * vel + 2 * acc * distance)) / acc
            next_vel = vel + acc * time
        else:
            # linear dynamic model. It is Stable by can't handle velocity close to 0.
            time_est = distance / vel
            next_vel_est = vel + acc * time_est
            # I guess a while loop can make it converge...
            average_vel = (vel + next_vel_est) / 2
            time = distance / average_vel
            next_vel = vel + acc * time

        return time, next_vel

    def loss(self):
        """
        Loss for Brachistochrone Curve
        Sum of times. NOTE: I didn't constrain the curve to be continuous
        :return: scale loss tensor
        """
        loss = 0.
        for t in self.Times:
            loss += t

        return tf.reduce_mean(loss)


def BrachistochroneCurveDemo():
    """
    Demo with a plot
    """
    START_POINT = Point(x=0., y=1.)
    END_POINT = Point(x=3., y=0.)
    b_curve = BrachistochroneCurve(start_point=START_POINT, end_point=END_POINT)

    loss = b_curve.loss()
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
    training_operation = optimizer.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.graph.finalize()

        NUM_ITERS = 1000
        for i in range(NUM_ITERS):
            x_variable = [x for x, y in b_curve.Points]
            y_variable = [y for x, y in b_curve.Points]
            _, loss_val, times, X, Y, vels, Acc \
                = sess.run([training_operation, loss, b_curve.Times, x_variable, y_variable, b_curve.Velocities, b_curve.Accelerations])
            print('iters:{} loss_val:{}'.format(i, loss_val))

            if VERBOSE:
                for x, y in zip(X, Y):
                    print('Point:', x, y)
                for var in times:
                    print('time:', var)
                for var in vels:
                    print('vel:', var)
                for var in Acc:
                    print('acc:', var)

            plt.plot(X, Y)
            plt.title('Brachistochrone Curve')
            plt.draw()
            plt.axis('equal')
            plt.pause(0.01)
            plt.clf()


def main():
    """
    Better scope
    """
    BrachistochroneCurveDemo()


if __name__ == '__main__':
    main()
