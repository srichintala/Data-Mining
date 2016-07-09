# Movie-Recommendation-System

Algorithm:
1. Load the train data set.
2. Create a nested dictionary which is of type
{uid1 : {mid1 : rating1; mid2 : rating}...}
3. Create a dictionary: for each movie, get users who have watched it.
{mid1 : [uid1; uid2; ...]; mid2 : [uid1; ...]...}
4. For each user calculate the distance between every other user by considering
the movies they have watched.
5. Users and the distances with every other user is stored in a nested dictionary
of the following format:
{uid1 : {uid2 : distance; uid3 : distance; ...}...}
6. Sort the dictionary based on the distance values so that similar users will
be first in the dictionary.
7. Read test data and for each user and movie in the file, get the users from the
users similarity dictionary and check for 30 of those users who have watched
the movie in test file.
i. If yes, get the ratings of 30 users and take average to predict rating.
ii. If 30 users are not found, get the ratings of all users from dictionary in step
3 and their ratings from dictionary in step 2. Takes average of all users rating
and predict.
iii. If the movie id is not present in base file, then assign predicted rating as
3 (chose the median value)
8. Finally, Calculate the MAD error to evaluate performance of recommendation
system.
I have tried multiple k values to get better prediction.
k=30 gave better results when compared to k=20, k=25, k=35, k=50.
Evaluated the performance of the recommendation system for 3 distance metrics.
Lmax and Euclidean gave better results compared to Manhattan distance
