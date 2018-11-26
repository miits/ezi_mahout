package put.prediction.recommender;

import java.io.File;
import java.util.List;

import org.apache.mahout.cf.taste.impl.model.file.*; 
import org.apache.mahout.cf.taste.impl.neighborhood.*; 
import org.apache.mahout.cf.taste.impl.recommender.*; 
import org.apache.mahout.cf.taste.impl.similarity.*; 
import org.apache.mahout.cf.taste.model.*; 
import org.apache.mahout.cf.taste.neighborhood.*; 
import org.apache.mahout.cf.taste.recommender.*; 
import org.apache.mahout.cf.taste.similarity.*; 

class RecommenderIntro { 
	
	private RecommenderIntro() { } 
	
	public static void main(String[] args) throws Exception { 
		
		//feed model with data read from file
		DataModel model = new FileDataModel(new File("data/movies.csv")); 
		
		//define user similarity as Pearson correlation coefficient
		UserSimilarity similarity = new PearsonCorrelationSimilarity(model); 
		
		//neighborhood = 50 nearest neighbors
		UserNeighborhood neighborhood = new NearestNUserNeighborhood(50, similarity, model); 
		
		// recommender instance - User-based CF
		GenericUserBasedRecommender recommender = new GenericUserBasedRecommender(model, neighborhood, similarity); 
		
		//top 10 recommendation for user 50
		List<RecommendedItem> recommendations = recommender.recommend(50, 10); 
		
		//print recommendation
		for (RecommendedItem recommendation : recommendations) { 
			
			System.out.println(recommendation); 
		}
		
	} 
} 
