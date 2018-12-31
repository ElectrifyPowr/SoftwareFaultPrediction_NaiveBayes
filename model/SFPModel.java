/** 
  * Copyright 2018-12-31 ElectrifyPowr
  *
  * All rights reserved
  *
  * @Author: ElectriyPowr
  * 
  * 
  * SFP model
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

    final int CATEGORY_NON_FAULTY = 0;
    final int CATEGORY_FAULTY = 0;

    final int SET_TYPE_DATASET = 0;
    final int SET_TYPE_TESTSET = 1;

    double[][] dataset;
    
    // dataset of NASA CM1
    //int noOfFeatures = 21;
    //int noOfModules = 496;
    
    // custom test dataset
    int noOfFeatures = 3;
    int noOfTrainingModules = 5;
    int noOfTestingModules = 0;
    String filenameDataset = "dataset.txt";
    String filenameUnknownModules = "unknown.txt";
    String path = "./";

    double[][] faultyData;
    double[][] nonFaultyData;

    private static final String COMMA_DELIMITER = ",";

    /*double[][] testSet = new double[][]{
        {12,3,1,1,51,227.43,0.1,10.42,21.83,2369.07,0.08,131.62,0,0,1,0,10,12,26,25,5,0},
        {31,4,1,2,141,829.45,0.05,21.52,38.55,17846.19,0.28,991.46,1,19,15,0,27,32,90,51,7,1},
        {28,6,5,5,104,564.33,0.06,16.09,35.08,9078.38,0.19,504.35,2,7,0,0,20,23,67,37,11,1}
    };*/

    double[][] testSet;
    
    public static void main(String[] args) {
        SFPModel model = new SFPModel();
        model.start();
    }

    public void start(){
        //int dsCount = getNumberOfLinesInFile(nameDataset);
        //int testCount = getNumberOfLinesInFile(nameUnknownModules);
        //System.out.println("dsCount: "+dsCount+", testCount: "+testCount);

        //load training data
        loadDataset();
        separateDataset();

        // load test data
        loadUnknownModules();

        System.out.println("noOfFeatures: "+noOfFeatures+", noOfTrainingModules: "+noOfTrainingModules+", noOfTestingModules: "+noOfTestingModules);

        makePredictions();
    }

    public void makePredictions(){
        System.out.println("Starting prediction...");
        
        for(int set=0; set<testSet.length; set++){
            System.out.println("---------------------------------- Module: "+set+" ----------------------------------");
            naiveBayesClassifier(testSet[set]);
            System.out.println("-------------------------------------------------------------------------------");
        }

    }

    private void naiveBayesClassifier(double[] moduleUnderTest){
        double[][] nonFaultyFeatureData = processNonFaultyData();
        double[][] faultyFeatureData = processFaultyData();

        //calc non-faulty probability
        // lastly, multiply by p(C=c) = p(C=No) = p(C=0)
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
        // lastly, multiply by p(C=c) = p(C=Yes) = p(C=1)
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

        System.out.println("Module is "+knownCategory+",\tPrediction: "+predictedString);
        System.out.println("NonFaultTmp: "+nonFaultTmp+", FaultTmp: "+faultTmp);

    }

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

    public double calcGaussDistribution(double x, double mean, double stDev){
        double firstPart = 1 / Math.sqrt(2 * Math.PI * Math.pow(stDev, 2));
        double secPart = Math.exp(-Math.pow((x-mean), 2) / (2 * Math.pow(stDev, 2)));
        return (firstPart * secPart);
    }

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

    public double getCategorySpecificSetSize(int category){
        if(category==CATEGORY_NON_FAULTY)
            return ((double)(nonFaultyData.length) / (double)(dataset.length));
        else
            return ((double)(faultyData.length) / (double)(dataset.length));
    }

    public String getPredictedCategoryString(int category){
        if(category==CATEGORY_NON_FAULTY)
            return "Non-Faulty";
        else
            return "Faulty";
    }

    public double[] getValuesForMean(double[][] data, int feature){
        double[] vals = new double[data.length];

        for(int i=0; i<data.length; i++){
            vals[i] = data[i][feature];
        }
        return vals;
    }

    public double arithmeticMean(double[][] data, int feature){
        double[] values = getValuesForMean(data, feature);
        double n = (double)(values.length);
        double tmp = 0;

        for (int i=0; i < n; i++){
            tmp += values[i];
        }

        return (tmp/n);
    }

    public int argmax(double[] vals){
        int idx = 0;
        for(int i=0; i<vals.length; i++){
            if(vals[i] > vals[idx])
                idx = i;
        }
        return idx;
    }

    /**
     * converts information from a csv file into a 2d dataset array
     * @return
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
     * converts information from a csv file into a 2d dataset array
     * @return
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

    public double[][] getDataset(){return dataset;}

}

