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
import java.util.List;

public class EZIEvaluator {

    public static void main(String[] args) throws Exception {
        RandomUtils.useTestSeed();
        DataModel model = new FileDataModel(new File("data/movies.csv"));
        RecommenderEvaluator evaluator = new RMSRecommenderEvaluator();

        List<UserSimilarity> similarities = new ArrayList<UserSimilarity>();
        similarities.add(new PearsonCorrelationSimilarity(model));
        similarities.add(new EuclideanDistanceSimilarity(model));
        similarities.add(new TanimotoCoefficientSimilarity(model));
        for(UserSimilarity similarity: similarities) {
            List<UserNeighborhood> neighbourhoods = new ArrayList<UserNeighborhood>();
            neighbourhoods.add(new ThresholdUserNeighborhood(0.5, similarity, model));
            neighbourhoods.add(new ThresholdUserNeighborhood(0.7, similarity, model));
            neighbourhoods.add(new NearestNUserNeighborhood(5, similarity, model));
            neighbourhoods.add(new NearestNUserNeighborhood(9, similarity, model));
            for(UserNeighborhood neighborhood: neighbourhoods) {
                RecommenderBuilder recommenderBuilder = new RecommenderBuilder() {
                    @Override
                    public Recommender buildRecommender(DataModel model) throws TasteException {
                        return new GenericUserBasedRecommender(model, neighborhood, similarity);
                    }
                };
                double score = evaluator.evaluate(recommenderBuilder, null, model, 0.7, 1.0);
                System.out.println(String.format("%s - %s - score: %f", neighborhood.toString(), similarity.toString(), score));
//                Threshold  Neighbourhood (0.7) and Euclidean Distance: RMSE = 0.080520
            }
        }
    }
}
