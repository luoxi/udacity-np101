from numpy import *

def compute_error_for_line_given_points(b, m, points):
    totalError = 0
     
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
         
        totalError += (y - m * x - b) **2
    
    return totalError / float(len(points))

def gradient_descent_runner(starting_b, starting_m, points, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    
    #gradient descet
    for i in range(num_iterations):
        #update b and m with the new more accurate b and m 
        #by performing this gradient step
        b, m = step_gradient(b, m, points, learning_rate)
    
    return [b, m]
    
def step_gradient(current_b, current_m, points, learning_rate):
    
    #starting point for gradients
    gradient_b = 0
    gradient_m = 0
    
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        
        #direction with respect to b and m
        gradient_b += -(2/float(len(points))) * (y - (current_m * x + current_b))
        gradient_m += -(2/float(len(points))) * x * (y - (current_m * x + current_b))
    #update b and m using partial derivatives
    new_b = current_b - (learning_rate * gradient_b)
    new_m = current_m - (learning_rate * gradient_m)
    return [new_b, new_m]

def run():
    
    #collect data
    points = genfromtxt('data.csv', delimiter=',')

    #define hyperparamters
    learning_rate = 0.0001
    #y = mx + b
    initial_m = 0
    initial_b = 0
    num_iterations = 1000
    
    #train model
    print "staring gradient descet at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points))
    [b, m] = gradient_descent_runner(initial_b, initial_m, points, learning_rate, num_iterations)
    print "ending point at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(b, m, points))

if __name__ == '__main__':
    run()
    
