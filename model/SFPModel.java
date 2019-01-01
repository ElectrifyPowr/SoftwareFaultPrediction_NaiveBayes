/** 
  * Copyright 2018-12-31 ElectrifyPowr
  *
  * All rights reserved
  *
  * @Author: ElectriyPowr
  * 
  * 
  * SFP model - learns from dataset.txt and then makes predictions for unknown.txt
  *
  * Using Naive Bayes Classifier approach
  * 
  */
package model;

import java.io.File;
import java.io.FileReader;
import java.io.LineNumberReader;
import java.io.IOException;
import java.io.BufferedReader;

import java.util.*;

public class SFPModel {

    // ---- constants ----
    // possible categories for predicting, either faulty or non-faulty
    final int CATEGORY_NON_FAULTY = 0;
    final int CATEGORY_FAULTY = 0;
    // possible datasets -> training or testing
    final int SET_TYPE_DATASET = 0;
    final int SET_TYPE_TESTSET = 1;
    // used to split features of txt file
    final String COMMA_DELIMITER = ",";

    // will hold contents of dataset.txt
    double[][] dataset;
    // will hold contents of unknown.txt
    double[][] testSet;

    // training dataset will be separated into these 2 datasets (needed for Naive Bayes)
    double[][] faultyData;
    double[][] nonFaultyData;
    
    // dataset of NASA CM1
    //int noOfFeatures = 21;
    //int noOfModules = 496;
    
    // custom test dataset
    // number of features for classification
    int noOfFeatures = 0;
    // each training module is 1 line in dataset.txt
    int noOfTrainingModules = 0;
    // each testing module is 1 line unknown.txt
    int noOfTestingModules = 0;

    String filenameDataset = "dataset.txt";
    String filenameUnknownModules = "unknown.txt";
    // path to both text files
    String path = "./";

    // some testing data for NASA CM1 dataset
    /*double[][] testSet = new double[][]{
        {12,3,1,1,51,227.43,0.1,10.42,21.83,2369.07,0.08,131.62,0,0,1,0,10,12,26,25,5,0},
        {31,4,1,2,141,829.45,0.05,21.52,38.55,17846.19,0.28,991.46,1,19,15,0,27,32,90,51,7,1},
        {28,6,5,5,104,564.33,0.06,16.09,35.08,9078.38,0.19,504.35,2,7,0,0,20,23,67,37,11,1}
    };*/

    /**
     * Main Function - Starts prediction model
     *
     */
    public static void main(String[] args) {
        SFPModel model = new SFPModel();
        model.start();
    }

    /**
     * Loads datasets and makes predictions.
     *
     */
    public void start(){
        //load training data
        loadDataset();
        separateDataset();

        // load test data
        loadUnknownModules();

        makePredictions();
    }

    /**
     * Iterates through all testingModules and makes predictions whether they are faulty or non-faulty.
     *
     */
    public void makePredictions(){
        System.out.println("Starting prediction...");
        
        for(int set=0; set<testSet.length; set++){
            System.out.println("---------------------------------- Module: "+set+" ----------------------------------");
            naiveBayesClassifier(testSet[set]);
            System.out.println("-------------------------------------------------------------------------------");
        }
        System.out.println("End of predictions...");
    }

    /**
     * Use Naive Bayes Classifier to make prediction on a testing module.
     * 
     * Implementation of Equation 1 (which can be found in README.md file)
     *
     * @param moduleUnderTest - holds all features of a module that should be predicted
     *
     */
    private void naiveBayesClassifier(double[] moduleUnderTest){
        double[][] nonFaultyFeatureData = processNonFaultyData();
        double[][] faultyFeatureData = processFaultyData();

        //calc non-faulty probability
        // firstly, assign p(C=c) = p(C=No) = p(C=0)
        double nonFaultTmp = getCategorySpecificSetSize(CATEGORY_NON_FAULTY);
        for(int feature=0; feature<moduleUnderTest.length-1; feature++){
            double x = moduleUnderTest[feature];
            double mean = nonFaultyFeatureData[feature][0];
            double stdDev = nonFaultyFeatureData[feature][1];
            double gaussDistr = calcGaussDistribution(x, mean, stdDev);
            String s = ("NaiveBayesClassifier - nonFaultyTmp calc - x="+x+", mean="+mean+", stdDev="+stdDev+", gauss="+gaussDistr+", oldTmp="+nonFaultTmp);
            nonFaultTmp *= gaussDistr;
            System.out.println(""+s+", newTmp="+nonFaultTmp);
        }

        //calc faulty probability
        // firstly, assign p(C=c) = p(C=Yes) = p(C=1)
        double faultTmp = getCategorySpecificSetSize(CATEGORY_FAULTY);
        for(int feature=0; feature<moduleUnderTest.length-1; feature++){
            double x = moduleUnderTest[feature];
            double mean = faultyFeatureData[feature][0];
            double stdDev = faultyFeatureData[feature][1];
            double gaussDistr = calcGaussDistribution(x, mean, stdDev);
            String s = ("NaiveBayesClassifier - faultyTmp calc - x="+x+", mean="+mean+", stdDev="+stdDev+", gauss="+gaussDistr+", oldTmp="+faultTmp);
            faultTmp *= gaussDistr;
            System.out.println(""+s+", newTmp="+faultTmp);
        }

        // now find argmax to get final prediction
        int predictedCategory = argmax(new double[]{nonFaultTmp, faultTmp});
        String knownCategory = getPredictedCategoryString((int)(moduleUnderTest[moduleUnderTest.length-1]));
        String predictedString = getPredictedCategoryString(predictedCategory);

        // print out prediction
        System.out.println("Module is "+knownCategory+",\tPrediction: "+predictedString);
        // print out raw values of non-faulty prediction and faulty prediction
        System.out.println("NonFaultTmp: "+nonFaultTmp+", FaultTmp: "+faultTmp);
    }

    /**
     * Calculates all necessary means and standard deviations for each feature of non-faulty dataset.
     * 
     * @return 2D-array with following structure:
     *
     *      | f1  |   f2  |   ...     |   fn    |
     * -----|-----|-------|-----------|---------|
     * mean |     |       |           |         |
     * -----|-----|-------|-----------|---------|
     * stDe |     |       |           |         |
     * ------------------------------------------
     *      
     * Where f1, f2, ..., fn represents each feature
     *       mean represents arithmetic mean of 1 feature of all data in dataset
     *       stDe represents standard deviation of 1 feature of all data in dataset
     *
     */
    private double[][] processNonFaultyData(){
        // will contain mean & std_dev for each feature
        double[][] data = new double[noOfFeatures][2];

        for(int feature=0; feature<data.length; feature++){
            double tmpMean = arithmeticMean(nonFaultyData, feature);
            double tmpStdDev = standardDeviation(nonFaultyData, tmpMean, feature);
            data[feature] = new double[]{tmpMean, tmpStdDev};
        }
        return data;
    }
    
    /**
     * Same as 'processNonFaultyData', but here for the faulty dataset.
     * @return 2D-array
     *
     */
    private double[][] processFaultyData(){
        // will contain mean & std_dev for each feature
        double[][] data = new double[noOfFeatures][2];

        for(int feature=0; feature<data.length; feature++){
            double tmpMean = arithmeticMean(faultyData, feature);
            double tmpStdDev = standardDeviation(faultyData, tmpMean, feature);
            data[feature] = new double[]{tmpMean, tmpStdDev};
        }
        return data;
    }

    /**
     * Separates training dataset into 2 datasets:
     *      - Faulty dataset (which only contains faulty modules)
     *      - Non-faulty dataset (only contains non-faulty modules)
     */
    private void separateDataset(){
        ArrayList<Integer> faultyDataIdx = new ArrayList<>();
        ArrayList<Integer> nonFaultyDataIdx = new ArrayList<>();

        // find indeces
        for(int i=0; i<dataset.length; i++){
            if(dataset[i][noOfFeatures]==CATEGORY_NON_FAULTY){
                nonFaultyDataIdx.add(i);
            } else {
                faultyDataIdx.add(i);
            }
        }

        nonFaultyData = new double[nonFaultyDataIdx.size()][noOfFeatures+1];
        faultyData = new double[faultyDataIdx.size()][noOfFeatures+1];

        for(int i=0; i<nonFaultyDataIdx.size(); i++){
            int datasetIdx = nonFaultyDataIdx.get(i);
            nonFaultyData[i] = dataset[datasetIdx];
        }
        
        for(int i=0; i<faultyDataIdx.size(); i++){
            int datasetIdx = faultyDataIdx.get(i);
            faultyData[i] = dataset[datasetIdx];
        }
    }

    /**
     * Calculates Gauss Distribution Function.
     *
     * Implementation of Equation 3 (which can be found in README.md file)
     *
     * @param x - value of feature
     * @param mean - mean of all values of certain feature of training dataset
     * @param stDev - standard deviation of all values of certain feature of training dataset
     *
     * @return processed value on distribution function
     */
    public double calcGaussDistribution(double x, double mean, double stDev){
        double firstPart = 1 / Math.sqrt(2 * Math.PI * Math.pow(stDev, 2));
        double secPart = Math.exp(-Math.pow((x-mean), 2) / (2 * Math.pow(stDev, 2)));
        return (firstPart * secPart);
    }

    /**
     * Calculates standard deviation of one feature of dataset.
     *
     * Implementation of Equation 2 (which can be found in README.md file)
     *
     * @param data - the dataset
     * @param mean - mean of all values of certain feature of training dataset 'data'
     * @param feature - index of feature for which stDev should be calculated for
     *
     * @return standard deviation
     */
    public double standardDeviation(double[][] data, double mean, int feature){
        int n = data.length;
        double stDev = 0;
        double sum = 0;
        
        for(int i=0; i<n; i++){
            double tmp = (data[i][feature] - mean);
            sum += Math.pow(tmp, 2); //first_arg raised to power of sec_arg
        }

        stDev = Math.sqrt(sum / (double)(n-1));

        return stDev;
    }

    /**
     * Get the category specific set size.
     *
     * Implementation of Equation 5 (which can be found in README.md file)
     *
     * @param category - based on this, the set size should be returned
     *
     * @return set size
     */
    public double getCategorySpecificSetSize(int category){
        if(category==CATEGORY_NON_FAULTY)
            return ((double)(nonFaultyData.length) / (double)(dataset.length));
        else
            return ((double)(faultyData.length) / (double)(dataset.length));
    }

    /**
     * Get string representation of certain category
     *
     * @param category - 0 = non-faulty
     *                 - 1 = faulty
     *
     * @return string representation
     */
    public String getPredictedCategoryString(int category){
        if(category==CATEGORY_NON_FAULTY)
            return "Non-Faulty";
        else
            return "Faulty";
    }

    /**
     * Gets all the values of one feature in the dataset 'data'
     *
     * @param data - dataset
     * @param feature 
     *
     * @return all values for one feature
     */
    public double[] getValuesForMean(double[][] data, int feature){
        double[] vals = new double[data.length];

        for(int i=0; i<data.length; i++){
            vals[i] = data[i][feature];
        }
        return vals;
    }

    /**
     * Calculates the arithmetic mean of a certain feature in dataset 'data'.
     *
     * Implementation of Equation 4 (which can be found in README.md file)
     *
     * @param data - dataset
     * @param feature
     *
     * @return arithmetic mean
     */
    public double arithmeticMean(double[][] data, int feature){
        double[] values = getValuesForMean(data, feature);
        double n = (double)(values.length);
        double tmp = 0;

        for (int i=0; i < n; i++){
            tmp += values[i];
        }

        return (tmp/n);
    }

    /**
     * Calculates the maximum value of a given array and returns its index of the array.
     *
     * Part of Equation 1 (which can be found in README.md file)
     *
     * @param vals - data array
     *
     * @return index of max value in array
     */
    public int argmax(double[] vals){
        int idx = 0;
        for(int i=0; i<vals.length; i++){
            if(vals[i] > vals[idx])
                idx = i;
        }
        return idx;
    }

    /**
     * Loads the training data from text file into dataset array.
     */
    public void loadDataset(){
        processLinesAndFeaturesOfFile(filenameDataset, SET_TYPE_DATASET);
        BufferedReader fileReader = null;
        try {
            String line = "";
            fileReader = new BufferedReader(new FileReader(new File(path, filenameDataset)));

            dataset = new double[noOfTrainingModules][noOfFeatures+1];

            int y=0;
            while ((line = fileReader.readLine()) != null) {
                String[] tokens = line.split(COMMA_DELIMITER);
                //+1 due to defective class
                for (int x = 0; x < (noOfFeatures + 1); x++) {
                    dataset[y][x] = Double.parseDouble(tokens[x]);
                }
                y++;
            }
        } catch (Exception e) {
            System.out.println("Error in CsvFileReader !!!");
            e.printStackTrace();
        } finally {
            try {
                fileReader.close();
            } catch (IOException e) {
                System.out.println("Error while closing fileReader !!!");
                e.printStackTrace();
            }
         }
    }
    
    /**
     * Loads the test data from text file into the testSet array.
     */
    public void loadUnknownModules(){
        processLinesAndFeaturesOfFile(filenameUnknownModules, SET_TYPE_TESTSET);
        BufferedReader fileReader = null;
        try {
            String line = "";
            fileReader = new BufferedReader(new FileReader(new File(path, filenameUnknownModules)));

            testSet = new double[noOfTestingModules][noOfFeatures+1];

            int y=0;
            while ((line = fileReader.readLine()) != null) {
                String[] tokens = line.split(COMMA_DELIMITER);
                //+1 due to defective class
                for (int x = 0; x < (noOfFeatures + 1); x++) {
                    testSet[y][x] = Double.parseDouble(tokens[x]);
                }
                y++;
            }
        } catch (Exception e) {
            System.out.println("Error in CsvFileReader !!!");
            e.printStackTrace();
        } finally {
            try {
                fileReader.close();
            } catch (IOException e) {
                System.out.println("Error while closing fileReader !!!");
                e.printStackTrace();
            }
         }
    }

    /**
     * Reads the number of modules and features of a file.
     *
     * @param filename - can be either dataset.txt or unknown.txt
     * @param type - helps distinguish between both sets
     *
     */
    private void processLinesAndFeaturesOfFile(String filename, int type){
        BufferedReader fileReader = null;
        try {
            String line = "";
            fileReader = new BufferedReader(new FileReader(new File(path, filename)));

            int lineCnt=0;
            while ((line = fileReader.readLine()) != null) {
                if(lineCnt==0 && type == SET_TYPE_DATASET){
                    String[] tokens = line.split(COMMA_DELIMITER);
                    noOfFeatures = tokens.length-1;
                }
                lineCnt++;
            }

            if(type==SET_TYPE_DATASET)
                noOfTrainingModules = lineCnt;
            else if(type==SET_TYPE_TESTSET)
                noOfTestingModules = lineCnt;

        } catch (Exception e){
            System.out.println("Error getting number of lines!!!");
            e.printStackTrace();
        } finally {
            try {
                fileReader.close();
            } catch (IOException e) {
                System.out.println("Error while closing fileReader !!!");
                e.printStackTrace();
            }
         }
    }

    // ----- getters & setters -----

    public double[][] getDataset(){return dataset;}

}

