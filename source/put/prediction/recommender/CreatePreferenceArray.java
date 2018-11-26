package put.prediction.recommender;

import org.apache.mahout.cf.taste.impl.model.GenericUserPreferenceArray;
import org.apache.mahout.cf.taste.model.Preference; 
import org.apache.mahout.cf.taste.model.PreferenceArray; 

class CreatePreferenceArray { 
	private CreatePreferenceArray() { } 

	/**
	 * Create a Preference object 
	 * Set preferences through some simple Java call 
	 * @param args
	 */
	public static void main(String[] args) { 
		
		// PreferenceArray stores preference of a single user
		PreferenceArray user1Prefs = new GenericUserPreferenceArray(2); 
		
		//data has form of triples (userID, itemID, value)
		
		user1Prefs.setUserID(0, 1L); 
		
		// score 2 for item 101
		user1Prefs.setItemID(0, 101L); 		
		user1Prefs.setValue(0, 2.0f); 
		
		// score 3 for item 102
		user1Prefs.setItemID(1, 102L); 		
		user1Prefs.setValue(1, 3.0f); 
		
		Preference pref = user1Prefs.get(1); 
		
		System.out.println(pref); 
	} 
} 