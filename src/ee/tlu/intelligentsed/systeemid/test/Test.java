package ee.tlu.intelligentsed.systeemid.test;

import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.SMOreg;
import weka.core.Instances;

public class Test {
	
	/**
	 * Crap for testing, ignore this
	 */

	public static Instances crossValidation(Instances dataset) {
		try {
			/** 
			 * Määra viimane atribuut indeksiks
			 */
			dataset.setClassIndex(dataset.numAttributes() - 1);

			/**
			 * Koosta klassifikaator
			 */
			//NaiveBayes classifier = new NaiveBayes();
			SMOreg classifier = new SMOreg();

			int seed = 1;
			int folds = 15;
			/** 
			 * Koosta suvaline andmekogu
			 */
			Random random = new Random(seed);
			Instances randomData = new Instances(dataset);
			randomData.randomize(random);

			/** 
			 * Ristvalideerimine
			 */
			for (int i = 0; i < folds; i++) {
				Evaluation evaluation = new Evaluation(randomData);

				/** 
				 * Jaga andmed osadeks
				 */
				Instances trainingSet = randomData.trainCV(folds, i);
				Instances testSet = randomData.testCV(folds, i);
				/**
				 * Ehita ja evalueeri klassifikaator
				 */
				classifier.buildClassifier(trainingSet);
				evaluation.evaluateModel(classifier, testSet);

				System.out.println("Fold number: " + i);
				System.out.println(evaluation.toSummaryString());
			}
		} catch (Exception e) {
			e.printStackTrace();
		}

		return null;
	}

	public static Instances regression(Instances dataset) {
		try {
			System.out.println("\nNew data after regression():");
			Instances newData = new Instances(dataset);
			newData.setClassIndex(newData.numAttributes() - 1);
			LinearRegression lr = new LinearRegression();
			lr.buildClassifier(newData);
			System.out.println(lr);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}
	
}
