package put.prediction.recommender;

import java.io.File;

import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.similarity.EuclideanDistanceSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

class ExampleSimilarity { 
	public static void main(String[] args) throws Exception { 

		// feed model with data read from file
		DataModel model = new FileDataModel(new File("data/movies.csv")); 
		
		//similarity definition
		UserSimilarity pearson = new PearsonCorrelationSimilarity(model); 
		
		//similarity measures for selected pair of users
		System.out.println("Pearson: " + pearson.userSimilarity(1, 2)); 
		System.out.println("Pearson: " + pearson.userSimilarity(1, 3)); 

		//different similarity definition
		UserSimilarity euclidean = new EuclideanDistanceSimilarity(model); 		
		System.out.println("Euclidean: " + euclidean.userSimilarity(1, 2)); 
		System.out.println("Euclidean: " + euclidean.userSimilarity(1, 3)); 
		
	} 
} 
