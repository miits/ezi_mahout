package put.prediction.recommender;

import java.io.File;
import java.io.IOException;
import java.util.List;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveArrayIterator;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.recommender.GenericItemBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.LogLikelihoodSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.TanimotoCoefficientSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;

public class RecommenderIntroAdvanced {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		try {

			// feed model with data read from file
			DataModel dataModel = new FileDataModel(new File("data/movies.csv"));
			
			// define item similarity as Tanimoto coefficient
			ItemSimilarity similarity = new TanimotoCoefficientSimilarity(dataModel);
			
			// recommender instance - Item-based CF
			GenericItemBasedRecommender recommender = new GenericItemBasedRecommender(dataModel, similarity);
						
			for(LongPrimitiveIterator items = dataModel.getItemIDs(); items.hasNext();){
				
				long itemId = items.nextLong();

				//get 5 most similar items for each item
				List<RecommendedItem> recommendations = recommender.mostSimilarItems(itemId, 5);
				
				//write 5 most similar items
				for (RecommendedItem recommendation : recommendations){
					
					System.out.println(itemId + "," + recommendation.getItemID() + "," + recommendation.getValue());
				}								
			}
			
		} catch (IOException e) {
			System.out.println("There was an error");
		} catch (TasteException e) {
			System.out.println("There was a Taste exception");
		}
				
	}
}
