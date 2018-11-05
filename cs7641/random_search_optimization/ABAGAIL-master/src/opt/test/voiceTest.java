package opt.test;

import dist.*;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;

import java.util.*;
import java.io.*;
import java.text.*;

/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network that is classifying abalone as having either fewer 
 * or more than 15 rings. 
 *
 * @author Hannah Lau
 * @version 1.0
 */
public class voiceTest {
    private static String train_data_file = new String("../dataset/voice.txt");
    //private static String test_data_file = new String("../dataset/voice_test_data.txt");
    //private static String cv_data_file = new String("../dataset/voice_cv_data.txt");
 
    private static int all_inst_sizse = 3169;
    private static Instance[] train_instances = initializeInstances(all_inst_sizse,train_data_file);
    //private static Instance[] test_instances = initializeInstances(all_inst_sizse,test_data_file);
    //private static Instance[] cv_instances = initializeInstances(all_inst_sizse,cv_data_file);

    private static int verbose = 0; 
    private static int inputLayer = 20, hiddenLayer1 = 3, outputLayer = 1, trainingIterations = 100;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
    
    private static ErrorMeasure measure = new SumOfSquaresError();
    private static DataSet set = new DataSet(train_instances);

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    private static String[] oaNames = {"RHC", "SA", "GA"};
    private static DecimalFormat df = new DecimalFormat("0.000");

    private static Vector<Double> train_accu = new Vector<Double>();
    private static Vector<Double> train_mse = new Vector<Double>();
    /*
    private static Vector<Double> test_accu = new Vector<Double>();
    private static Vector<Double> test_mse = new Vector<Double>();
    private static Vector<Double> cv_accu = new Vector<Double>();
    private static Vector<Double> cv_mse = new Vector<Double>();
    
    */
    private static Vector<Double> train_times = new Vector<Double>();
    //private static double cv_accuracy = 0;
    //private static double cv_mean_square_error = 0;
    private static int step = 10;
    //private static int k = 0;
    
    public static void main(String[] args) {
        if (args.length != 3) {
            System.out.println("Error: 3 arguements are required:training iterations,step,verbose"); 
            System.exit(0);
        } else {
            trainingIterations = Integer.parseInt(args[0]);
            verbose = Integer.parseInt(args[2]); 
            step = Integer.parseInt(args[1]);
            System.out.println("Info: training iterations: "+trainingIterations);
            System.out.println("Info: verbose: "+verbose);
            System.out.println("Info: step: "+step);
        }
         
        for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                new int[] {inputLayer,hiddenLayer1,hiddenLayer1,
                    outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
        }
        oa[0] = new RandomizedHillClimbing(nnop[0]);
        oa[1] = new SimulatedAnnealing(1E12, .10,nnop[1]);
        oa[2] = new StandardGeneticAlgorithm(100, 25, 10, nnop[2]);

        for(int i = 2; i < oa.length; i++) {

            double start = System.nanoTime(), end, trainingTime, trainTestingTime, testTestingTime, correct = 0, incorrect = 0;
            train(oa[i], networks[i], oaNames[i]); //trainer.train();
            end = System.nanoTime();

            trainingTime = end - start;
            trainingTime /= Math.pow(10,9);

            Instance optimalInstance = oa[i].getOptimal();
            networks[i].setWeights(optimalInstance.getData());
            double predicted, actual, train_score, test_score, cv_score;
            String train_results = "";
            String test_results = "";

            for(int j = 0; j < train_instances.length; j++) {
                networks[i].setInputValues(train_instances[j].getData());
                networks[i].run();

                predicted = Double.parseDouble(train_instances[j].getLabel().toString());
                actual = Double.parseDouble(networks[i].getOutputValues().toString());

                double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

            }
            end = System.nanoTime();
            trainTestingTime = end - start;
            trainTestingTime /= Math.pow(10,9);
            train_score = correct/(correct+incorrect);
            train_results +=  "\nTrain: Results for " + oaNames[i] + ": \nCorrectly classified " + correct + " instances." +
                        "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                        + df.format(train_score*100) + "%\nTraining time: " + df.format(trainingTime)
                        + " seconds\nTesting time: " + df.format(trainTestingTime) + " seconds\n";

            /* Test score
            correct = 0;
            incorrect = 0;
            start = System.nanoTime();
            for(int j = 0; j < test_instances.length; j++) {
                networks[i].setInputValues(test_instances[j].getData());
                networks[i].run();

                predicted = Double.parseDouble(test_instances[j].getLabel().toString());
                actual = Double.parseDouble(networks[i].getOutputValues().toString());

                double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
            }
            end = System.nanoTime();
            testTestingTime = end - start;
            testTestingTime /= Math.pow(10,9);
            test_score = correct/(correct+incorrect);
            test_results +=  "\nTest: Results for " + oaNames[i] + ": \nCorrectly classified " + correct + " instances." +
                        "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                        + df.format(test_score*100) + "%\nTraining time: " + df.format(trainingTime)
                        + " seconds\nTesting time: " + df.format(testTestingTime) + " seconds\n";
            */
            if (verbose == 1) {
                System.out.println(train_results);
                System.out.println(test_results);
            }

            //writeFile(train_accu,test_accu,cv_accu,train_mse,test_mse,cv_mse,train_times,oaNames[i]);
            writeFile(train_accu,train_mse,train_times,oaNames[i]);
            train_accu.clear();
            train_mse.clear();
            train_times.clear();
            /*
            test_accu.clear();
            cv_accu.clear();
            
            test_mse.clear();
            cv_mse.clear();
            */

        }
    }
    private static void getDataError(OptimizationAlgorithm oa,BackPropagationNetwork network,Instance[] instances, String data_type) {
        // Train score
        double actual, predicted, accuracy,mean_square_error,error = 0;
        double[] current_weights = network.getWeights();
        int correct = 0;
        int incorrect = 0;
        Instance optimalInstance = oa.getOptimal();
        network.setWeights(optimalInstance.getData());

        for(int j = 0; j < instances.length; j++) {
            network.setInputValues(instances[j].getData());
            network.run();
           
            // compute accuracy
            actual = Double.parseDouble(network.getOutputValues().toString());
            predicted = Double.parseDouble(instances[j].getLabel().toString());
            if (Math.abs(predicted - actual) < 0.5) {
                correct++;
            } else {
                incorrect++;
            }
            
            // compute error
            Instance output = instances[j].getLabel(), example = new Instance(network.getOutputValues());
            example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
            error += measure.value(output, example);
        }
        // final results
        mean_square_error = error/(double)instances.length;
        accuracy = correct/(double)(correct+incorrect);
        train_accu.addElement(accuracy);
        train_mse.addElement(mean_square_error); 
        /*
        if (Objects.equals("train", data_type)) {
           train_accu.addElement(accuracy);
           train_mse.addElement(mean_square_error); 
        } else if (Objects.equals("test", data_type)) {
            test_accu.addElement(accuracy);
            test_mse.addElement(mean_square_error);
        } else {
            cv_accu.addElement(accuracy);
            cv_mse.addElement(mean_square_error);
            k++;
            cv_accuracy = (cv_accuracy*k+accuracy)/(k+1);
            cv_mean_square_error = (cv_mean_square_error*k+accuracy)/(k+1); 
        }*/
        network.setWeights(current_weights);
    }
 
    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName) {
        System.out.println("\nTraining with " + oaName + " Total iteration:" + trainingIterations);
        double percent_completed = 0;
        double start, end,tempTime,trainingTime = 0;
        for(int i = 0; i < trainingIterations; i++) {
            // train
            start = System.nanoTime();
            oa.train();
            end = System.nanoTime();
            tempTime = end - start;
            tempTime /= Math.pow(10,9);
            trainingTime += tempTime;
            
            getDataError(oa,network,train_instances,"train");
            //getDataError(oa,network,test_instances,"test");
            //getDataError(oa,network,cv_instances,"cv");              
            train_times.addElement(trainingTime);
            if (verbose == 1) {
                percent_completed = (i/(double)trainingIterations)*100;
                if (percent_completed % 10 == 0) {
                    System.out.println(percent_completed + "%..."); 
                }
            }
        }
        System.out.println("100%...Done!"); 
    }

    private static Instance[] initializeInstances(int max_inst_num, String fname) {

        //For voice data set, there are total instance 3168.
        double[][][] attributes = new double[max_inst_num][][];
        int total_instances = 0;
        try {
            BufferedReader br = new BufferedReader(new FileReader(new File(fname)));
            Scanner scan = null;
            String next_line = null;
            
            for(int i = 0; i < attributes.length; i++) {
                next_line = br.readLine();
                // if split the data, then it may be less than 3168
                if (next_line == null) {
                    total_instances = i;
                    break;
                }
                scan = new Scanner(next_line);
                scan.useDelimiter(",");
                attributes[i] = new double[2][];
                attributes[i][0] = new double[20]; // 20 attributes
                attributes[i][1] = new double[1];

                for(int j = 0; j < 20; j++)
                    attributes[i][0][j] = Double.parseDouble(scan.next());
                attributes[i][1][0] = Double.parseDouble(scan.next());
            }

            scan.close();
            br.close();
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[total_instances];
        for(int i = 0; i < total_instances; i++) {
            instances[i] = new Instance(attributes[i][0]);
            // classifications range from 0 to 30; split into 0 - 14 and 15 - 30
            // 0 -> male voice, 1 -> female voice
            instances[i].setLabel(new Instance(attributes[i][1][0] < 1 ? 0 : 1));
        }
        System.out.println("Parsed dataset:"+fname+". Initialized "+total_instances+" instances."); 
        return instances;
    }
    //public static void writeFile(Vector<Double> train_accu,Vector<Double> test_accu,Vector<Double> cv_accu,Vector<Double> train_mse,
    //                             Vector<Double> test_mse,Vector<Double> cv_mse,Vector<Double> train_times,String fname) {

    public static void writeFile(Vector<Double> train_accu,Vector<Double> train_mse,Vector<Double>train_times,String fname) {                        
        PrintStream out = null;
        try {
            System.out.println("Start writing to file.");
            out = new PrintStream(new FileOutputStream("../outputs/"+fname));
            for (int i = 0; i < train_accu.size(); i++)
                //out.println((i+1)+","+train_accu.elementAt(i)+","+test_accu.elementAt(i)+","+cv_accu.elementAt(i)+","+train_mse.elementAt(i)+","
                //+test_mse.elementAt(i)+","+cv_mse.elementAt(i)+","+train_times.elementAt(i));
                out.println((i+1)+","+train_accu.elementAt(i)+","+train_mse.elementAt(i)+","+train_times.elementAt(i));
        } catch (ArrayIndexOutOfBoundsException e) {
            System.err.println("ArrayIndexOutOfBoundsException Error:" +
                 e.getMessage());
        } catch (IOException e) {
            System.err.println("IOException: " + e.getMessage());
        } finally {
            if (out != null) {
                System.out.println("PrintStream");
                out.close();
            } else {
                System.out.println("Couldn't open connection");
            }
        }
    }
}
