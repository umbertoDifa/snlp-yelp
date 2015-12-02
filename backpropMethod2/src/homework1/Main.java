package homework1;

import com.opencsv.CSVReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.List;

/**
 *
 * @author Umberto
 */
public class Main {

    private static int N_TRAINSET;
    private static final int N_INPUTS = 3;
    private static final int N_NEURONS_FIRST = 200;
    private static final int N_NEURONS_SECOND =200;
    private static final int N_NEURONS_OUTPUT = 5;

    private static int N_TESTSET;

    private static int NUMBER_OF_EPOCHS;

    private static double[][] trainingData;
    private static double[][] testData;

    private static double[][] trainingLabels;
    private static double[][] testLabels;

    public static void main(String[] args) throws FileNotFoundException, IOException {
        //==================import test        
        CSVReader csvReaderTestData = new CSVReader(new FileReader(new File("../testForNeural/data.csv")));
        List<String[]> testDatalist = csvReaderTestData.readAll();
        // Convert to 2D array
        String[][] testDataArr = new String[testDatalist.size()][];
        testDataArr = testDatalist.toArray(testDataArr);
        testData = new double[testDataArr.length][testDataArr[0].length];

        N_TESTSET = testData.length;
        for (int i = 0; i < testDataArr.length; i++) {
            for (int j = 0; j < testDataArr[i].length; j++) {
                testData[i][j] = Float.parseFloat(testDataArr[i][j]);
            }
        }

        CSVReader csvReaderTestLabels = new CSVReader(new FileReader(new File("../testForNeural/target.csv")));
        List<String[]> testLabelslist = csvReaderTestLabels.readAll();
        // Convert to 2D array
        String[][] testLabelsArr = new String[testLabelslist.size()][];
        testLabelsArr = testLabelslist.toArray(testLabelsArr);
        testLabels = new double[testLabelsArr.length][testLabelsArr[0].length];
        for (int i = 0; i < testLabelsArr.length; i++) {
            for (int j = 0; j < testLabelsArr[i].length; j++) {
                testLabels[i][j] = Float.parseFloat(testLabelsArr[i][j]);
            }
        }

        int k = 10;
        loadTrainingData(10);
        System.out.printf("====================Percentage of training %d\n", k * 10);
        BackPropNN nn = new BackPropNN();
        //add first layer with 529 neurons and 529 inputs
        nn.addLayer(N_NEURONS_FIRST, N_INPUTS);
        //second layer
        nn.addLayer(N_NEURONS_SECOND, N_NEURONS_FIRST);
        
        //third layer
        nn.addLayer(N_NEURONS_OUTPUT, N_NEURONS_SECOND);
        // gatherStats(nn);
        float res = 0;
        //stop only if max limit of iteration or enough success
        //for (int e = 5; res < 0.95 && e <= 200; e += 5) {
        NUMBER_OF_EPOCHS = 1;
        int e = NUMBER_OF_EPOCHS;
        System.out.printf("====Epochs: %d\n", e);
        long startTime = System.currentTimeMillis();

        //each time train 5 more epochs
        nn.train(trainingData, N_TRAINSET, trainingLabels, NUMBER_OF_EPOCHS, false);
        long endTime = System.currentTimeMillis();
        long totalTime = endTime - startTime;
        System.out.printf("time %d\n", totalTime);
        res = nn.test(testData, N_TESTSET, testLabels);

       // }
    }

    private static void loadTrainingData(int k) throws IOException, NumberFormatException, FileNotFoundException {
        //import train
        CSVReader csvReaderTrainingData = new CSVReader(new FileReader(new File("../trainForNeural/data.csv")));
        List<String[]> list = csvReaderTrainingData.readAll();
        // Convert to 2D array
        String[][] dataArr = new String[list.size()][];
        dataArr = list.toArray(dataArr);

        CSVReader csvReaderTrainingLabels = new CSVReader(new FileReader(new File("../trainForNeural/target.csv")));
        List<String[]> listLabels = csvReaderTrainingLabels.readAll();
        // Convert to 2D array
        String[][] dataLabelsArr = new String[listLabels.size()][];
        dataLabelsArr = listLabels.toArray(dataLabelsArr);

        //============convert data from string to float
        trainingData = new double[dataArr.length][dataArr[0].length];
        N_TRAINSET = trainingData.length;

        for (int i = 0; i < dataArr.length; i++) {
            for (int j = 0; j < dataArr[i].length; j++) {
                trainingData[i][j] = Float.parseFloat(dataArr[i][j]);
            }
        }
        trainingLabels = new double[dataLabelsArr.length][dataLabelsArr[0].length];
        for (int i = 0; i < dataLabelsArr.length; i++) {
            for (int j = 0; j < dataLabelsArr[i].length; j++) {
                trainingLabels[i][j] = Float.parseFloat(dataLabelsArr[i][j]);
            }
        }
    }

    private static void gatherStats(BackPropNN nn) throws IOException {
        int NUMB_OF_RUNS = 10;
        long sum;

        int epochs = 20;
        loadTrainingData(8);
        sum = 0;
        for (int i = 0; i < NUMB_OF_RUNS; i++) {
            long startTime = System.currentTimeMillis();
            nn.train(trainingData, N_TRAINSET, trainingLabels, epochs, true);
            long endTime = System.currentTimeMillis();
            long totalTime = endTime - startTime;
            sum += totalTime;
        }
        System.out.printf("Execution time for %d epoch averaged on %d runs is: %d milliseconds\n", epochs, NUMB_OF_RUNS, sum / NUMB_OF_RUNS);

    }
}
