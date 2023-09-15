from flask import *
import numpy as np

#Defining the activation function for the neural network
def relu(x):
    return max(0.0, x)

#Defining the derivative of the ReLu function in python
def der(x):
    return 1 * (x > 0)

def model_train():
    #Assign the input value
    input_value = np.array([[1, 0, 1, 1, 0, 1, 1, 0],        #Each [] corresponds to a 8-D vector containing whether the student has answered a particular question right (1) or wrong (0)
                       [1, 0, 0, 1, 0, 1, 0, 0], 
                       [0, 0, 1, 0, 1, 1, 1, 1], 
                       [1, 0, 0, 1, 1, 1, 1, 0], 
                       [0, 0, 1, 0, 0, 1, 1, 0], 
                       [1, 0, 1, 0, 1, 0, 0, 0], 
                       [0, 1, 1, 0, 0, 1, 0, 1], 
                       [1, 0, 0, 1, 0, 1, 0, 1], 
                       [1, 0, 0, 1, 1, 1, 1, 0], 
                       [1, 1, 1, 0, 1, 1, 1, 0],
                       [0, 1, 1, 0, 0, 0, 1, 1],
                       [1, 0, 0, 1, 0, 1, 0, 0],
                       [0, 1, 1, 1, 0, 1, 1, 0],
                       [0, 1, 0, 1, 1, 0, 1, 1],
                       [0, 1, 1, 0, 1, 1, 0, 1]])      
    
    #assign the output value of the training set
    output = np.array([2, 0, 3, 3, 2, 2, 1, 2, 3, 3, 0, 0, 3, 1, 2])
    output = output.reshape(15, 1)

    #Assigning the weights
    weights = np.array([[[-0.6337704248273285], 
                         [-0.16160213496596795], 
                         [0.10334436176901375], 
                         [-0.3041908294647837], 
                         [1.029064662636711], 
                         [0.6439121063305244], 
                         [0.2406496000008942], 
                         [-0.8442210942534811]]])  
    #assigning bias
    bias = 0.1


    #The training phase of the neural network
    for epoch in range(1):
        input_arr = input_value

        weighted_sum = np.dot(input_arr, weights) + bias
        #print(weighted_sum)
        first_output = relu(weighted_sum.any())

        error = first_output - output           #Check the logic of codes between breakpoints
        total_error = np.square(np.subtract(first_output, output)).mean()

        first_derivative = error
        second_derivative = der(first_output)
        derivative = first_derivative * second_derivative

        t_input = input_value.T
        final_derivative = np.dot(t_input, derivative)

        #update weights
        weights = weights - (0.05 * final_derivative)

        #update bias
        for i in derivative:
            bias = bias - (0.05 * i)    
        
    return weights, bias

#Predictions of test input
def predict_CoC(bL, weights, bias):
    pred = np.array(bL)
    result = np.dot(pred, weights) + bias
    res = relu(result)
    CoC = round(res[0][0] - 1.3)        #Clarity of Concept (CoC) metric

    return CoC

#The flask app to provide a GUI frontend

app = Flask(__name__)

@app.route("/result1")
def render_coc1():
    return render_template("coc1.html")

@app.route("/result0")
def render_coc0():
    return render_template("coc0.html")

@app.route("/result2")
def render_coc2():
    return render_template("coc2.html")

@app.route("/result3")
def render_coc3():
    return render_template("coc3.html")

@app.route("/evaluate", methods = ["POST"])
def evaluate():
    binaryL = []
    ans1 = int(request.form['ans1'])
    if(ans1 == 8):
        binaryL.append(1)
    else:
        binaryL.append(0)
    
    ans2 = int(request.form['ans2'])
    if(ans2 == 42):
        binaryL.append(1)
    else:
        binaryL.append(0)
    
    ans3 = int(request.form['ans3'])
    if(ans3 == 124):
        binaryL.append(1)
    else:
        binaryL.append(0)
    
    ans4 = int(request.form['ans4'])
    if(ans4 == 4):
        binaryL.append(1)
    else:
        binaryL.append(0)
    
    ans5 = int(request.form['ans5'])
    if(ans5 == 4480):
        binaryL.append(1)
    else:
        binaryL.append(0)
    
    ans6 = int(request.form['ans6'])
    if(ans6 == 20736):
        binaryL.append(1)
    else:
        binaryL.append(0)
    
    ans7 = int(request.form['ans7'])
    if(ans7 == -32):
        binaryL.append(1)
    else:
        binaryL.append(0)
    
    ans8 = int(request.form['ans8'])
    if(ans8 == 74):
        binaryL.append(1)
    else:
        binaryL.append(0)

    print(binaryL)
    
    weights, bias = model_train()
    
    coc = predict_CoC(binaryL, weights, bias)

    print(coc)
    return render_template("home.html", coc=coc)

@app.route('/', methods = ["GET"])
def getvalue():
    coc=None
    print(coc)
    return render_template("home.html", coc='none')

if __name__ == '__main__':
    app.run(debug=True)
        #return redirect("/")

#__main__#

#WEIGHT, BIAS = model_train()

#getvalue(WEIGHT, BIAS)


