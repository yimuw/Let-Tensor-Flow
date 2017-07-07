"""
Solve optimal control by Tensorflow

Optimal Control problem:

Define: dx(t) = dynamic_t( x(t), u(t) ), where dx(t) means dx/dt evaluate at t
        x(0) = constant

Goal:   Minimize  J = Sum_over_t( h_t( x(t), u(t) ), where h_t(*) is a function output a scale
"""

import tensorflow as tf


class InvestmentOptimalControl:
    """
    Investment system

    State vector: Money, Investment
    Control: Put money to investment. Negative mean get money from investment
    Constrain: money >= 0,  investment >= 0

    Goal: Maximize money in the final state
    """

    def __init__(self, init_money):
        """
        Init variables
        """
        # 10 years
        self.horizon = 10

        self.X = []
        self.U = []

        # Init controls
        for i in range(self.horizon):
            self.U.append(tf.Variable(0., name='u{}'.format(i)))

        # Init states
        x0 = [tf.Variable(init_money, trainable=False, name='x0_money'),
              tf.Variable(0., trainable=False, name='x0_investment')]
        self.X.append(x0)

        # Construct the system
        for i in range(self.horizon - 1):
            assert len(self.X) == i + 1
            money_next, investment_next = self.dynamic(self.X[i], self.U[i])
            xi = [tf.identity(money_next, name='x{}_money'.format(i + 1)),
                  tf.identity(investment_next, name='x{}_investment'.format(i + 1))]
            self.X.append(xi)

    def dynamic(self, x, u):
        """
        Time independent dynamic function
        If you want to learn it, that is Reinforcement learning
        :param x: this state
        :param u: this control
        :return: next state
        """
        money, total_investment = x
        new_investment = u

        # Linear
        INVESTMENT_GAIN_RATE = 0.05
        # None linear
        INVESTMENT_COST_RATE = 0.02

        money_next = money - new_investment \
                     + INVESTMENT_GAIN_RATE * total_investment \
                     - INVESTMENT_COST_RATE * tf.abs(new_investment)  # Not sure how tf handle it...Sub Gradient?
        total_investment_next = total_investment + new_investment

        return [money_next, total_investment_next]

    def money_in_the_end(self):
        """
        Money in the end
        :return: money in the final state
        """
        money, total_investment = self.X[-1]
        return money

    def loss(self):
        """
        Loss function, which is money
        :return: loss tensor
        """
        reward = self.money_in_the_end()

        # Handle constrains
        for x in self.X:
            money, total_investment = x
            # Using log barrier to handle inequality constrains
            LOG_BARRIER_T = 2.
            reward += 1 / LOG_BARRIER_T * tf.log(total_investment + 0.1)
            reward += 1 / LOG_BARRIER_T * tf.log(money + 0.1)

        # Planing to machine learning
        loss = -reward

        return tf.reduce_mean(loss)


def optimal_control_demo():
    """
    Run the optimal control
    """
    money_system = InvestmentOptimalControl(init_money=100.)
    loss = money_system.loss()

    # Hacky gradient descent :)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-1)
    training_operation = optimizer.minimize(loss)

    total_money = money_system.money_in_the_end()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.graph.finalize()

        NUM_ITER = 5000
        for i in range(NUM_ITER):
            _, loss_val, current_total_money = sess.run([training_operation, loss, total_money])
            print('loss_val: ', loss_val)
            print('total_money: ', current_total_money)

        print('Result:')
        for u in money_system.U:
            var_name = u.name
            var_value = sess.run(u)
            print(var_name, ' value: ', var_value)

        for x_vec in money_system.X:
            for x in x_vec:
                var_name = x.name
                var_value = sess.run(x)
                print(var_name, ' value: ', var_value)


if __name__ == '__main__':
    optimal_control_demo()
