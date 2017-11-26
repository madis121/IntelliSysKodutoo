package ee.tlu.intelligentsed.systeemid.test;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.trees.RandomTree;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.CSVLoader;
import weka.core.converters.CSVSaver;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;

public class Main {

	/**
	 * Kodutöö - http://lambda.ee/wiki/Ifi6057labx 
	 * Weka dokumentatsioon - http://weka.sourceforge.net/doc.stable/
	 */

	public static void main(String[] args) {
		try {
			Instances dataset = readFromCSV("IFI6057_hw2017_data.txt");
			
//			dataset = attributeSelection(dataset);
//			dataset.deleteAttributeAt(0);
//			dataset.deleteAttributeAt(1);
//			dataset.deleteAttributeAt(2);
//			dataset.deleteAttributeAt(3);
//			dataset.deleteAttributeAt(4);
//			dataset.deleteAttributeAt(5);
//			dataset.deleteAttributeAt(6);
//			dataset.deleteAttributeAt(7);
//			dataset.deleteAttributeAt(8);

			dataset.randomize(new Random(0));
			double percentage = 0.8;
			int trainingSetSize = (int) Math.round(dataset.numInstances() * percentage);
			int testSetSize = (int) Math.round(dataset.numInstances() - trainingSetSize);
			Instances trainingSet = new Instances(dataset, 0, trainingSetSize);
			Instances testSet = new Instances(dataset, trainingSetSize, testSetSize);
			trainingSet.setClassIndex(trainingSet.numAttributes() - 1);
			testSet.setClassIndex(testSet.numAttributes() - 1);
			
			for (int i = 1; i < 100; i += 9) {
				double p = (double) i / 100;
				int currentTrainingSetSize = (int) Math.round(trainingSet.numInstances() * p);
				Instances currentTrainingSet = new Instances(trainingSet, 0, currentTrainingSetSize);
				currentTrainingSet.setClassIndex(currentTrainingSet.numAttributes() - 1);
				System.out.println("\nCurrent training set size: " + currentTrainingSetSize);
				
				//decisionTree(currentTrainingSet, testSet);
				neuralNetwork(currentTrainingSet, testSet);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public static void neuralNetwork(Instances trainingSet, Instances testSet) {
		try {
			MultilayerPerceptron perceptron = new MultilayerPerceptron();
//			perceptron.setLearningRate(0.1);
//			perceptron.setMomentum(0.2);
//			perceptron.setHiddenLayers("3");
//			perceptron.setTrainingTime(2000);
			perceptron.buildClassifier(trainingSet);

			Evaluation evaluation = new Evaluation(trainingSet);
			evaluation.evaluateModel(perceptron, testSet);
			ArrayList<Prediction> predictions = evaluation.predictions();
			
			int correct = 0;
			
			for (int i = 0; i < predictions.size(); i++) {
				Instance instance = testSet.instance(i);
				Prediction prediction = predictions.get(i);
				
				if (prediction.actual() == prediction.predicted()) {
					//System.out.println("Correctly predicted: " + instance);
					correct++;
				} else {
					//System.out.println("Incorrectly predicted: " + instance + ", predicted value: " + prediction.predicted());
				}
			}
			
			int correctPercentage = correct * 100 / testSet.size();
			System.out.println("(Neural network) Correctly predicted " + correct + " out of " + testSet.size() + " (" + correctPercentage + "%)");

			// saveToCSV(testSet, "nnOutput.csv");
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public static void decisionTree(Instances trainingSet, Instances testSet) {
		try {
			// Classifier tree = new M5P();
			RandomTree tree = new RandomTree();
			String optionsString = 
					"-K " + 9 + // Number of attributes to randomly investigate
					" -M " + 0 + // Set minimum number of instances per leaf
					" -S " + 0 + // Seed for random number generator
					" -N " + 2 + // Number of folds for backfitting
					" -depth " + 0 + // The maximum depth of the tree, 0 for unlimited.
					" -V " + 0.001;
			String[] options = Utils.splitOptions(optionsString);
			tree.setOptions(options);
			tree.buildClassifier(trainingSet);

			Evaluation evaluation = new Evaluation(trainingSet);
			evaluation.evaluateModel(tree, testSet);
			ArrayList<Prediction> predictions = evaluation.predictions();
			
			int correct = 0;
			
			for (int i = 0; i < predictions.size(); i++) {
				Instance instance = testSet.instance(i);
				Prediction prediction = predictions.get(i);
				
				if (prediction.actual() == prediction.predicted()) {
//					System.out.println("Correctly predicted: " + instance);
					correct++;
				} else {
//					System.out.println("Incorrectly predicted: " + instance + ", predicted value: " + prediction.predicted());
				}
			}
			
			int correctPercentage = correct * 100 / testSet.size();
			System.out.println("(Decision tree) Correctly predicted " + correct + " out of " + testSet.size() + " (" + correctPercentage + "%)");

			// saveToCSV(testSet, "dtOutput.csv");
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * Filters out data which the algorithms believe to be irrelevant using CfsSubsetEval and GreedySetwise
	 */
	public static Instances attributeSelection(Instances dataset) {
		try {
			AttributeSelection filter = new AttributeSelection();
			CfsSubsetEval evaluator = new CfsSubsetEval();
			GreedyStepwise search = new GreedyStepwise();
			// BestFirst search = new BestFirst();
			search.setSearchBackwards(true);
			filter.setEvaluator(evaluator);
			filter.setSearch(search);
			filter.setInputFormat(dataset);

			Instances newData = Filter.useFilter(dataset, filter);
			return newData;
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}

	private static Instances readFromCSV(String fileName) throws IOException {
		CSVLoader csvLoader = new CSVLoader();
		csvLoader.setSource(new File(fileName));
		Instances dataset = csvLoader.getDataSet();
		return dataset;
	}

	@SuppressWarnings("unused")
	private static void saveToCSV(Instances dataset, String fileName) throws IOException {
		CSVSaver csvSaver = new CSVSaver();
		csvSaver.setInstances(dataset);
		csvSaver.setFile(new File(fileName));
		csvSaver.writeBatch();
	}

}
