from numpy import *
def run():

    points=genfromtxt('data.csv',delimiter=',')

    learning_rate=0.0001
    ini_b=0
    ini_m=0
    num_iter=1000

    print ("starting gradient descent at b={0},m={1}, error = {2}".format(ini_b,ini_m,compute_error_for_line_given_points(ini_b,ini_m,points)))
    print ("Running...")
    [b,m]=(gradient_descent_runner(points,ini_b,ini_m,learning_rate,num_iter))
    print("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iter,b,m,compute_error_for_line_given_points(b,m,points)))
def compute_error_for_line_given_points(b,m,points):
    totalError=0
    for i in range(0,len(points)):
        x=points[i,0]
        y=points[i,1]
        totalError+=(y-(m*x+b))**2
    return totalError/float(len(points))
def gradient_descent_runner(points,start_b,start_m,learning_rate,num_iter):
    b=start_b
    m=start_m
    for i in range(num_iter):
        b,m=step_gradient(b,m,array(points),learning_rate)
    return [b,m]
def step_gradient(b_current,m_current,points,learning_rate):
    b_gradient=0
    m_gradient=0
    n = float(len(points))
    for i in range(0,len(points)):
        x=points[i,0]
        y=points[i,1]
        b_gradient+=-(2/n)*(y-((m_current*x)+b_current))
        m_gradient+=-(2/n)*x*(y-((m_current*x)+b_current))
    new_b=b_current-(learning_rate*b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_b,new_m]
if __name__ == '__main__':
    run()

