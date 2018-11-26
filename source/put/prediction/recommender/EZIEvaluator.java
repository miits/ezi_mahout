package put.prediction.recommender;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.RMSRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.neighborhood.ThresholdUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.EuclideanDistanceSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.TanimotoCoefficientSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;
import org.apache.mahout.common.RandomUtils;

import java.io.File;
import java.nio.file.attribute.UserPrincipalNotFoundException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class EZIEvaluator {

    public static void main(String[] args) throws Exception {
        RandomUtils.useTestSeed();
        DataModel model = new FileDataModel(new File("data/movies.csv"));
        RecommenderEvaluator evaluator = new RMSRecommenderEvaluator();

        HashMap<String, UserSimilarity> similaritiesByName = new HashMap<>();
        similaritiesByName.put("Pearson Correlation", new PearsonCorrelationSimilarity(model));
        similaritiesByName.put("Euclidean Distance", new EuclideanDistanceSimilarity(model));
        similaritiesByName.put("Tanimoto Coefficient", new TanimotoCoefficientSimilarity(model));
        for(Map.Entry<String, UserSimilarity> similarity: similaritiesByName.entrySet()) {
            HashMap<String, UserNeighborhood> neighborhoodsByName = new HashMap<>();
            neighborhoodsByName.put("Threshold 0.5", new ThresholdUserNeighborhood(0.5, similarity.getValue(), model));
            neighborhoodsByName.put("Threshold 0.7", new ThresholdUserNeighborhood(0.7, similarity.getValue(), model));
            neighborhoodsByName.put("NearestN 5", new NearestNUserNeighborhood(5, similarity.getValue(), model));
            neighborhoodsByName.put("NearestN 9", new NearestNUserNeighborhood(9, similarity.getValue(), model));
            for(Map.Entry<String, UserNeighborhood> neighborhood: neighborhoodsByName.entrySet()) {
                RecommenderBuilder recommenderBuilder = new RecommenderBuilder() {
                    @Override
                    public Recommender buildRecommender(DataModel model) throws TasteException {
                        return new GenericUserBasedRecommender(model, neighborhood.getValue(), similarity.getValue());
                    }
                };
                double score = evaluator.evaluate(recommenderBuilder, null, model, 0.7, 1.0);
                System.out.println(String.format("%s - %s - score: %f", similarity.getKey(), neighborhood.getKey(), score));
//                Tanimoto Coefficient - Threshold 0.5 - score: 0,882644
//                Tanimoto Coefficient - NearestN 9 - score: 1,067234
//                Tanimoto Coefficient - Threshold 0.7 - score: NaN
//                Tanimoto Coefficient - NearestN 5 - score: 1,100279
//                Pearson Correlation - Threshold 0.5 - score: 0,789053
//                Pearson Correlation - NearestN 9 - score: 0,810898
//                Pearson Correlation - Threshold 0.7 - score: 0,742296
//                Pearson Correlation - NearestN 5 - score: 0,868883
//                Euclidean Distance - Threshold 0.5 - score: 0,656538
//                Euclidean Distance - NearestN 9 - score: 0,373923
//                Euclidean Distance - Threshold 0.7 - score: 0,081290
//                Euclidean Distance - NearestN 5 - score: 0,330527
            }
        }
    }
}
