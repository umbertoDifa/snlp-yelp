/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package homework1;

import static java.lang.Math.*;
import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author Umberto
 */
public class BackPropNN {

    private List<Layer> layerList;
    private double LEARNING_RATE = 0.5;

    public void addLayer(int numberOfOutputs, int numberOfInputs) {
        //create layer
        Layer tmpLay = new Layer(numberOfOutputs, numberOfInputs);
        tmpLay.setIndex(layerList.size());
        //if layerlist has already layers adjust connections
        if (!layerList.isEmpty()) {
            Layer prevLayer = layerList.get(layerList.size() - 1);
            prevLayer.setNextLayer(tmpLay);
            tmpLay.setPreviousLayer(prevLayer);
        }
        layerList.add(tmpLay);
    }

    public BackPropNN() {
        layerList = new ArrayList<>();
    }

    public void train(double[][] input, int numberOfDataset,
                      double[][] datatrue, int numberOfEpochs, boolean silent) {
        double min = 1000000; //min error achieved
        double sumError = 0.0;
        float success = 0;

        for (int epoch = 0; epoch < numberOfEpochs; epoch++) {
            //feedForward phase
            //for each dataset
            if (!silent) {
                // System.out.printf("epoch:%d\n", epoch);
            }
            sumError = 0.0;
            success = 0;
            for (int i = 0; i < numberOfDataset; i++) {
               // System.out.printf("\ninput n: %d/%d\n", i, numberOfDataset);
                layerList.get(0).feedForward(extractRow(input, i));

                //error detection
                double[] error = new double[datatrue[0].length];
                double[] output;

                output = layerList.get(layerList.size() - 1).getOutput();

                //System.out.printf("ERROR is: [");
                for (int j = 0; j < datatrue[0].length; j++) {
                   // System.out.printf("%f\n",output[j]);
                    error[j] = datatrue[i][j] - output[j];
                    sumError += pow(error[j], 2);
                   // System.out.printf("(%f)", error[j]);
                }
                //System.out.print("]");
                boolean correctOutput = true;
                //if all the outputs are belowe 0.1 error
                for (int l = 0; l < output.length; l++) {
                    if (abs(error[l]) >= 0.1) {
                        correctOutput = false;
                        break;
                    }

                }
                if (correctOutput) {
                    System.out.println("--RECOGNIZED--");
                    success++;
                } else {
                    System.out.println("--NOT RECOGNIZED--");
                }

                //backpropagation
                layerList.get(layerList.size() - 1).backProp(extractRow(datatrue, i));

                //update weights
                layerList.get(0).updateWeights(LEARNING_RATE);
            }

            min = min(sumError / 2, min);
            //updateLearningRate(epoch);
        }

        if (!silent) {
            System.out.printf("Train Success is:%f / %d =  %f\n", success, numberOfDataset, (success / numberOfDataset));
            // System.out.printf("Average error is:%f\n", sumError / 2);
        }

    }

    public float test(double[][] input, int numberOfTestset, double[][] datatrue) {
        float sum = 0;
        for (int i = 0; i < numberOfTestset; i++) {
            //System.out.printf("\nTest n: %d\n", i);
            double[] row = extractRow(input, i);
            sum += singleTest(row, i, datatrue);
        }
        System.out.printf("Test Success is:%f / %d =  %f\n", sum, numberOfTestset, (sum / numberOfTestset));
        return sum / numberOfTestset;
    }

    private int singleTest(double[] input, int indexOfTestToRun, double[][] datatrue) {
        layerList.get(0).feedForward(input);

        //error detection
        double[] error = new double[datatrue[0].length];
        double[] output = new double[datatrue[0].length];

        output = layerList.get(layerList.size() - 1).getOutput();

        //   System.out.printf("ERROR is: [");
        for (int j = 0; j < datatrue[0].length; j++) {
            error[j] = datatrue[indexOfTestToRun][j] - output[j];
            // System.out.printf("(%f)", error[j]);
        }
        //  System.out.println("]");
        boolean correctOutput = true;
        //if all the outputs are belowe 0.1 error
        for (int l = 0; l < output.length; l++) {
            if (abs(error[l]) >= 0.1) {
                correctOutput = false;
                break;
            }

        }
        if (correctOutput) {
            //      System.out.println("--RECOGNIZED--");
            //success++;
            return 1;
        } else {
            //      System.out.println("--NOT RECOGNIZED--");
            return 0;
        }

    }

    //**UTILITY FUNCTIONS **//
    private double[] extractRow(double[][] dataset, int row) {
        //System.out.println("Extracting row...\n");
        double[] tmpArray = new double[dataset[row].length];
        for (int i = 0; i < dataset[row].length; i++) {
            tmpArray[i] = dataset[row][i];
            //System.out.printf("%f|",tmpArray[i]);
        }
        return tmpArray;
    }

}
