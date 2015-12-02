package homework1;

import homework1.utility.RandomGenerator;

/**
 *
 * @author Umberto
 */
public class Layer {

    private int numberOfNeurons;
    private int numberOfInputs;
    /**
     * (j,i) j are the next neurons, i are the previous neurons * inputs
     */
    private double[][] weight;
    private double[] x; //inputs to the layer
    private double[] z; //result of linear combination of inputs
    private double[] y; //result of activation function on z
    private double[] phi; //error

    private int index; //index of the layer in the nn

    private Layer previousLayer = null; //initially no previousLayer
    private Layer nextLayer = null; //initially no nextLayer

    public Layer(int numberOfNeurons, int numberOfInputs) {
        this.numberOfInputs = numberOfInputs;
        this.numberOfNeurons = numberOfNeurons;
        //the number of weights is the product because the net if fully connected
        this.weight = new double[numberOfNeurons][numberOfInputs];
        this.x = new double[numberOfInputs];
        this.z = new double[numberOfNeurons];
        this.y = new double[numberOfNeurons];
        this.phi = new double[numberOfNeurons];
        this.initializeWeights();
        this.reset();
    }

    private void reset() {
        //reset the value of each neuron
        for (int i = 0; i < numberOfNeurons; i++) {
            z[i] = 0.0;
            y[i] = 0.0;
            phi[i] = 0.0;
        }
    }

    public double[] getOutput() {
        return y;
    }

    public void feedForward(double[] input) {
        //save input so that later we can update the weights
        System.arraycopy(input, 0, this.x, 0, input.length);

        if (input.length != numberOfInputs) {
            System.out.printf("Expected input at layer %d were %d received are %d, abort.\n", this.index, numberOfInputs, input.length);
            return;
        }
        this.reset();
        //combine the inputs
        for (int i = 0; i < numberOfNeurons; i++) {
            for (int j = 0; j < numberOfInputs; j++) {
                z[i] += weight[i][j] * input[j];
            }
            //activation function, sigmoid
            y[i] = sigmoidActivation(z[i]);
        }

        //now if the layer is the last the function ends otherwise
        //the output of this layer is forwarded into the next one
        if (this.nextLayer == null) {
            return;
        } else {
            this.nextLayer.feedForward(y);
        }
    }

    public void backProp(double targetOrPhi[]) {
        //phi = dE/dy * derivative of activation function 
        //in case last layer dE/dy = d-y where d is the target value
        //in case of hidden layer dE/dy = (phi*w)of succesive layer
        //System.out.printf("Layer %d backprop\n", this.index);

        //if this is the last layer
        if (this.nextLayer == null) {
            for (int i = 0; i < numberOfNeurons; i++) {
                //System.out.printf("%d-Calculating %f * %f\n", i, sigmoideDerivative(z[i]), targetOrPhi[i] - y[i]);
                phi[i] = (sigmoideDerivative(z[i])) * (targetOrPhi[i] - y[i]);
            }
        } else {
            for (int i = 0; i < numberOfNeurons; i++) {
                double sumPhi;
                sumPhi = 0.0;
                for (int j = 0; j < targetOrPhi.length; j++) {
                    //System.out.printf("1-Calculating %f * %f\n", targetOrPhi[j], this.nextLayer.weight[j][i]);
                    sumPhi += targetOrPhi[j] * this.nextLayer.weight[j][i];
                }
                //System.out.printf("2-Calculating %f * %f\n", sumPhi, sigmoideDerivative(z[i]));
                phi[i] = sumPhi * sigmoideDerivative(z[i]);
            }
        }

        if (this.previousLayer == null) {
            //this was the first layer of the nework
            return;
        } else {
            this.previousLayer.backProp(this.phi);
        }
    }

    public void updateWeights(double learningRate) {
        //System.out.printf("Weights layer %d updated\n",this.index);
        for (int i = 0; i < numberOfNeurons; i++) {
            for (int j = 0; j < numberOfInputs; j++) {
                //System.out.printf("updating w[%d][%d] -= %f* %f * %f\n",i,j,learningRate,phi[i],x[j]);
                weight[i][j] += learningRate * phi[i] * x[j];
            }
        }

        if (this.nextLayer == null) {
            //if last layer
            return;
        } else {
            //forward weights update
            this.nextLayer.updateWeights(learningRate);
        }
    }

    private double sigmoidActivation(double val) {
        return 1 / (1 + Math.exp(-val));
    }

    private double sigmoideDerivative(double val) {
        return sigmoidActivation(val) * (1 - sigmoidActivation(val));
    }

    //give a random value between -0.5 and 0.5
    private void initializeWeights() {
        for (int i = 0; i < numberOfNeurons; i++) {
            for (int j = 0; j < numberOfInputs; j++) {
                weight[i][j] = RandomGenerator.randomInRange(-0.5, 0.5);
            }
        }
    }

    public Layer getPreviousLayer() {
        return previousLayer;
    }

    public void setPreviousLayer(Layer previousLayer) {
        this.previousLayer = previousLayer;
    }

    public Layer getNextLayer() {
        return nextLayer;
    }

    public void setNextLayer(Layer nextLayer) {
        this.nextLayer = nextLayer;
    }

    public int getIndex() {
        return index;
    }

    public void setIndex(int index) {
        this.index = index;
    }

}
