package put.prediction.recommender;

import java.io.File;

import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.AverageAbsoluteDifferenceRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.RMSRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;
import org.apache.mahout.common.RandomUtils;

class EvaluatorIntro { 
	private EvaluatorIntro() { } 
	public static void main(String[] args) throws Exception { 
		
		// set seed - running tests several times in a fair way
		RandomUtils.useTestSeed(); 
		
		DataModel model = new FileDataModel(new File("data/movies.csv")); 
		
		// evaluator - average absolute difference
		RecommenderEvaluator evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator(); 
		
		// build the same recommender for testing as in RecommenderIntro 
		RecommenderBuilder recommenderBuilder = new RecommenderBuilder() { 
			@Override 
			public Recommender buildRecommender(DataModel model) throws TasteException { 
				UserSimilarity similarity = new PearsonCorrelationSimilarity(model); 
				UserNeighborhood neighborhood = new NearestNUserNeighborhood(50, similarity, model); 
				return new GenericUserBasedRecommender(model, neighborhood, similarity); 
			} 
		}; 
			
		// 0.7 - 70% learning data, 1.0 - 100% test data
		double score = evaluator.evaluate(recommenderBuilder, null, model, 0.7, 1.0);  
		
		System.out.println(score); 
	} 
} 
