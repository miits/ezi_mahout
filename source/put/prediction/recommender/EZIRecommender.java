package put.prediction.recommender;

import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.ThresholdUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.EuclideanDistanceSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

import java.io.File;
import java.util.List;

public class EZIRecommender {

    public static void main(String[] args) throws Exception{
        DataModel model = new FileDataModel(new File("data/movies.csv"));
        UserSimilarity similarity = new EuclideanDistanceSimilarity(model);
        UserNeighborhood neighborhood = new ThresholdUserNeighborhood(0.7, similarity, model);
        GenericUserBasedRecommender recommender = new GenericUserBasedRecommender(model, neighborhood, similarity);
        for (LongPrimitiveIterator it = model.getUserIDs(); it.hasNext(); ) {
            Long userId = it.next();
            List<RecommendedItem> recommendations = recommender.recommend(userId, 3);
            System.out.print(String.format("User: %d Recommendations: ", userId));
            for(RecommendedItem item: recommendations) {
                System.out.print(String.format("%d (%f) ", item.getItemID(), item.getValue()));
            }
            System.out.println();
//            User: 943 Recommendations: 258 (5,000000) 300 (4,333333) 751 (4,000000)
        }
    }
}
