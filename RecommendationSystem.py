import urllib2
from itertools import combinations
import math
import pandas as pd
import csv
import numpy as np
from collections import namedtuple, defaultdict

class RecommendationSystem:

    def __init__(self):
        self.users = defaultdict(dict)
        self.userSeenMovies = defaultdict(list)
        self.user_sim = defaultdict(dict)
        self.sorted_dict = defaultdict(dict)
        self.diff_total = []
        self.diff_list = []
        self.r_total = []
        self.error = 0.0
        self.MAD_values = []
        self.base_file_path = ''
        self.test_file_path = ''
        self.average_mad_error = 0.0
#load train data set
#ml-100k/u1.base
    def load_data(self, base_file_path):
        UserMovieRating = namedtuple('UserMovieInfo', 'User_ID, Movie_ID, Rating, Timestamp')
        for row in map(UserMovieRating._make, csv.reader(open(base_file_path, 'rb'), delimiter='\t')):
            yield row
            #print "row", row

    #create a nested dictionary of {users: {movie : ratings}}
    def userWiseRatings(self,base_file_path,distance_type):
        for line in self.load_data(base_file_path):
            #upper = (int(line.Rating) - 1)
            #norm_rating = (float(upper)/4)
            self.users[int(line.User_ID)][int(line.Movie_ID)] = (int(line.Rating))
            if(self.userSeenMovies.has_key(int(line.Movie_ID))):
                self.userSeenMovies[int(line.Movie_ID)].append(int(line.User_ID))
            else:
                self.userSeenMovies[int(line.Movie_ID)] = [int(line.User_ID)]
        print "User wise movies and ratings nested dictionary is done!!!"
        print "Movie wise user dictionary is done!!!"
        for user_id in self.users.keys():
            self.finding_similar_users(self.users,user_id,distance_type)
        print "Similar users for each user is calculated using the distance metric you chose!!!"
        return self.users

    #compare each users with all other users
    def finding_similar_users(self,users,user_id,distance_type):
        self.diff_list = []
        for keys in users:
            if user_id != keys:
                self.calculate_distance(keys,user_id,distance_type)

    #calculate euclidean/manhattan/lmax distance (based on user input) distance between users and store the scores in a dictionary
    #{u1:{u2: score, u3: score..}, u2:{u1:score, u3:score..}..)
    def calculate_distance(self,keys,user_id,distance_type):
        diff = 0
        count = 0
        diff2 = 0
        max_value = 0
        for movie_ids in self.users[user_id]:
            movieid1 = movie_ids
            rating1 = self.users[user_id][movie_ids]
            #print "id", movieid,"rating", rating
            if(self.users[keys].has_key(movieid1)):
                rating2= self.users[keys][movieid1]
                if(distance_type == 1 | distance_type == 2):
                    diff = diff+(abs(rating1 - rating2)**distance_type)
                else:
                    diff2 = abs(rating1-rating2)
                    self.diff_list.append(diff2)
                count= count +1
                max_value = max(self.diff_list)
        #print "count", count
        if(distance_type == 1):
            user_dist = diff
        elif(distance_type ==2):
            user_dist = math.sqrt(diff)
        elif(distance_type == 3):
            user_dist = max_value
        self.store_in_dict(user_id, keys, user_dist)
        if(self.user_sim[keys].has_key(user_id)):
            self.store_in_dict(user_id, keys, user_dist)
        else:
            self.user_sim[user_id][keys] = (user_dist)

    def store_in_dict(self, user_id, keys, user_dist):
        self.user_sim[user_id][keys] = user_dist

    #sort the dictionary of users and scores in ascending order of their scores
    def sort_dict(self, user_sim):
        for userid1 in user_sim:
            user_distances =  user_sim[userid1]
            s= sorted(user_distances.iteritems(), key=lambda (k,v): (v,k))
            self.sorted_dict[userid1] = s
        print "User and the distances between every user is sorted in ascending order i.e, most similar user will be first."

    #read the test file to find movies which a user did not watch and
    #predict rating based on similar users rating for that movie
    #ml-100k/u1.test
    def load_test_data(self,test_file_path,distance_type):
        print "Test file is loaded to predict ratings for movies in test file"
        self.r_total = []
        self.diff_total = []
        for line in open(test_file_path, 'rb'):
            row = line.strip().split('\t')
            self.predict_rating(row)
        var1 = sum(self.r_total)
        var2 = sum(self.diff_total)
        self.error = float(float(var2)/float(var1))
        self.calculate_final_MAD(self.error)

    #For each user id and movieid in test file, this function is called
    #Here the dictionary of distances between users is used to get top k users who
    #has seen that movie and their ratings are added and averaged to get a predict rating
    #for that movie.
    #If k users are not available, ratings of all the users who watched that movie is
    #taken from a dictionary called Userseenmovies {movieid1: {userid1, userid2,..}..}
    #and is the average value of all those ratings is given as predicted rating.
    #For the case where a particular movie id is not present in the Train data,
    #predicted rating is given as 3
    def predict_rating(self,row):
        id_list= []
        similar_user_ratings = 0
        movie_seen = 0
        all_user_ratings = 0
        all_count = 0
        u_ids = int(row[0])
        m_ids = int(row[1])
        true_rating = int(row[2])
        r_ij = False
        rating_indicator = 0
        diff_in_prediction = 0
        for sim_ids in self.sorted_dict[u_ids]:
            id_list.append(sim_ids[0])

        for p in range (len(id_list)):
            if(self.users[id_list[p]].has_key(m_ids)):
                r = self.users[id_list[p]][m_ids]
                similar_user_ratings = similar_user_ratings + r
                movie_seen= movie_seen+1
                if(movie_seen >=30):
                    break
        if(movie_seen >= 30):
            r_ij = True
            predicted_rating = int(round((similar_user_ratings*1.0)/movie_seen))
        elif(movie_seen > 0 & movie_seen < 30):
            r_ij = True
            for k in range(len(self.userSeenMovies[m_ids])):
                all_r = self.users[(self.userSeenMovies[m_ids])[k]][m_ids]
                all_user_ratings = all_user_ratings+all_r
                all_count = all_count+1
            predicted_rating = int(round((all_user_ratings*1.0)/all_count))
        elif(movie_seen == 0):
            predicted_rating = 3
        if(r_ij == True):
            rating_indicator = 1
        else:
            rating_indicator = 0
        self.r_total.append(rating_indicator)
        diff_in_prediction = (rating_indicator * abs(predicted_rating-true_rating))
        self.diff_total.append(diff_in_prediction)

    def calculate_final_MAD(self,error):
        self.MAD_values.append(self.error)
        add_mad_values = sum(self.MAD_values)
        self.avg_mad_error = float(float(add_mad_values)/float(5.0))

def main():
    rs = RecommendationSystem()
    distance_type = int(raw_input('Choose the distance type: Enter 1 for Manhattan distance, Enter 2 for Euclidean distance, Enter 3 for Lmax distance: '))
    for n_fold in range(0,5):
        base_file_path = raw_input('Enter train file complete path: ')
        test_file_path = raw_input('Enter test file complete path: ')
        rs.load_data(base_file_path)
        rs.userWiseRatings(base_file_path, distance_type)
        rs.sort_dict(rs.user_sim)
        rs.load_test_data(test_file_path,distance_type)
        #print "MAD error: ", rs.error
    if(distance_type == 1):
        print "MAD Error after 5 fold cross-validation using Manhattan distance is: ", rs.avg_mad_error
    elif(distance_type == 2):
        print "MAD Error after 5 fold cross-validation using Euclidean distance is: ", rs.avg_mad_error
    elif(distance_type == 3):
       print "MAD Error after 5 fold cross-validation using Lmax distance is: ", rs.avg_mad_error

main()

