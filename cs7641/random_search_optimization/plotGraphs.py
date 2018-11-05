import csv
import matplotlib.pyplot as plt
import numpy as np

def readData(file_name):
    with open(file_name, 'r', encoding="utf8") as file:
        reader = csv.reader(file,delimiter=',')
        csv_list = list(reader)
    print('Read in file -> {}'.format(file_name))

    return csv_list

def plotIt(file_name,text_str,algo):
    print("Plotting graph....")
    data = readData(file_name)
    iterations = []
    train_scores = []
    test_scores = []
    cv_scores = []
    train_mse = []
    test_mse = []
    cv_mse = []
    train_times = []

    for pt in data:
        iterations.append(int(pt[0])) 
        train_scores.append(float(pt[1]))
        #test_scores.append(float(pt[2]))
        #cv_scores.append(float(pt[3]))

        train_mse.append(float(pt[2]))
        #test_mse.append(float(pt[5]))
        #cv_mse.append(float(pt[6])) 
        train_times.append(float(pt[3]))
    '''
    b: blue
    g: green
    r: red
    c: cyan
    m: magenta
    y: yellow
    k: black
    w: white
    '''

    # scores
    plt.figure()
    line_width = 1.5
    plt.plot(iterations, train_scores, 'r',label='Training Score',linewidth=line_width)
    #plt.plot(iterations, test_scores, 'b',label='Testing Score',linewidth=line_width)
    #plt.plot(iterations, cv_scores, 'r',label='CV Score',linewidth=line_width)

    xlabel = 'Iteration Number'
    ylabel = 'Accuracy' 
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #plt.ylim([0,1.01])
    plt.suptitle('{}: {} Vs. {}'.format(algo,ylabel,xlabel), fontsize=15)
    plt.title('({})'.format(text_str),fontsize = 8)
    plt.legend()
    plt.grid()
    plt.draw()
    plt.savefig('./outputs/{}_{}.png'.format(algo,ylabel))
    # mean square error
    plt.figure()
    plt.plot(iterations, train_mse, 'r',label='Training MSE',linewidth=line_width)
    #plt.plot(iterations, test_mse, 'b',label='Testing MSE',linewidth=line_width)
    #plt.plot(iterations, cv_mse, 'r',label='CV MSE',linewidth=line_width)

    xlabel = 'Iteration Number'
    ylabel = 'Mean Square Error' 

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #plt.ylim([0,0.])
    plt.suptitle('{}: {} Vs. {}'.format(algo,ylabel,xlabel), fontsize=15)
    plt.title('({})'.format(text_str),fontsize = 8)
    plt.legend()
    plt.grid()
    plt.draw()
    plt.savefig('./outputs/{}_{}.png'.format(algo,ylabel))
    # train time
    plt.figure()
     
    plt.plot(iterations, train_times, 'b',label='Train Time',linewidth=line_width)
    
    xlabel = 'Iteration Number'
    ylabel = 'Time (seconds)' 
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #plt.ylim([0,1.01])
    plt.suptitle('{}: {} Vs. {}'.format(algo,ylabel,xlabel), fontsize=15)
    plt.title('({})'.format(text_str),fontsize = 8)
    
    plt.legend()
    plt.grid()
    plt.draw()
    plt.savefig('./outputs/{}_{}.png'.format(algo,ylabel))

def getdata(data_input):
    iterations = []
    train_scores = []
    train_mse = []
    train_times = []
    for pt in data_input:
        iterations.append(int(pt[0])) 
        train_scores.append(float(pt[1]))
        train_mse.append(float(pt[2]))
        train_times.append(float(pt[3]))
    return iterations,train_scores,train_mse,train_times

def plotSA(file_name,text_str,algo):
    print("Plotting graph....")
    data10 = readData("./outputs/SA_10")
    data25 = readData("./outputs/SA_25")
    data50 = readData("./outputs/SA_50")
    data75 = readData("./outputs/SA_75")
    data95 = readData("./outputs/SA_95")

    iterations = []
    train_scores = []
    train_mse = []
    train_times = []
    


    '''
    b: blue
    g: green
    r: red
    c: cyan
    m: magenta
    y: yellow
    k: black
    w: white
    '''

    # scores
    plt.figure()
    line_width = 1.5
    iterations,train_scores,train_mse,train_times = getdata(data10)
    plt.plot(iterations, train_scores, 'b',label='CR=10',linewidth=line_width)

    iterations,train_scores,train_mse,train_times = getdata(data25)
    plt.plot(iterations, train_scores, 'g',label='CR=25',linewidth=line_width)

    iterations,train_scores,train_mse,train_times = getdata(data50)
    plt.plot(iterations, train_scores, 'r',label='CR=50',linewidth=line_width)

    iterations,train_scores,train_mse,train_times = getdata(data75)
    plt.plot(iterations, train_scores, 'y',label='CR=75',linewidth=line_width)

    iterations,train_scores,train_mse,train_times = getdata(data95)
    plt.plot(iterations, train_scores, 'k',label='CR=95',linewidth=line_width)
    xlabel = 'Iteration Number'
    ylabel = 'Accuracy' 
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #plt.ylim([0,1.01])
    plt.suptitle('{}: {} Vs. {}'.format(algo,ylabel,xlabel), fontsize=15)
    plt.title('({})'.format(text_str),fontsize = 8)
    plt.legend()
    plt.grid()
    plt.draw()
    plt.savefig('./outputs/{}_{}.png'.format(algo,ylabel))
    # mean square error
    plt.figure()
    iterations,train_scores,train_mse,train_times = getdata(data10)
    plt.plot(iterations, train_mse, 'b',label='CR=10',linewidth=line_width)

    iterations,train_scores,train_mse,train_times = getdata(data25)
    plt.plot(iterations, train_mse, 'g',label='CR=25',linewidth=line_width)

    iterations,train_scores,train_mse,train_times = getdata(data50)
    plt.plot(iterations, train_mse, 'r',label='CR=50',linewidth=line_width)

    iterations,train_scores,train_mse,train_times = getdata(data75)
    plt.plot(iterations, train_mse, 'y',label='CR=75',linewidth=line_width)

    iterations,train_scores,train_mse,train_times = getdata(data95)
    plt.plot(iterations, train_mse, 'k',label='CR=95',linewidth=line_width)

    xlabel = 'Iteration Number'
    ylabel = 'Mean Square Error' 

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #plt.ylim([0,0.])
    plt.suptitle('{}: {} Vs. {}'.format(algo,ylabel,xlabel), fontsize=15)
    plt.title('({})'.format(text_str),fontsize = 8)
    plt.legend()
    plt.grid()
    plt.draw()
    plt.savefig('./outputs/{}_{}.png'.format(algo,ylabel))
    # train time
    plt.figure()
     
    plt.plot(iterations, train_times, 'b',label='Train Time',linewidth=line_width)
    
    xlabel = 'Iteration Number'
    ylabel = 'Time (seconds)' 
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #plt.ylim([0,1.01])
    plt.suptitle('{}: {} Vs. {}'.format(algo,ylabel,xlabel), fontsize=15)
    plt.title('({})'.format(text_str),fontsize = 8)
    
    plt.legend()
    plt.grid()
    plt.draw()
    plt.savefig('./outputs/{}_{}.png'.format(algo,ylabel))

file_path = "./"
run_rhc = 0
run_sa = 0
run_ga = 1
if run_rhc:
    file_name = './outputs/{}/RHC'.format(file_path)
    plotIt(file_name,"RHC: Hidden layer=2. Neuron=3","RHC")

if run_sa:
    file_name = './outputs/{}/SA_95'.format(file_path)
    #plotIt(file_name,"SA: Hidden layer=2. Neuron=3","SA")
    plotSA(file_name,"SA: Hidden layer=2. Neuron=3","SA")

if run_ga:
    file_name = './outputs/{}/GA'.format(file_path)
    plotIt(file_name,"GA: Hidden layer=2. Neuron=3","GA")
plt.show()