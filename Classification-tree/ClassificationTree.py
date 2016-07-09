import numpy as np
import operator
from numpy import *

#to hold the objects of tree
class Node(object):
    def __init__(self,value_of_node,threshold_value,right_child,left_child,leaf):
        self.root_value = value_of_node
        self.root_left = left_child
        self.root_right = right_child
        self.threshold = threshold_value
        self.leaf = leaf

#finally calculate accuracy of the predictions made on test sets
#total accuracy is the average of all fold accuracy
def calculate_accuracy(test_class_label_list,predicted_list):
    correct_predictions = []
    wrong_predictions = []
    for labels in range(len(test_class_label_list)):
        if(test_class_label_list[labels] == predicted_list[labels]):
            correct_predictions.append(labels)
        else:
            wrong_predictions.append(labels)
    true_predictions = len(correct_predictions)
    false_predictions = len(wrong_predictions)
    fold_accuracy = float(true_predictions) / len(predicted_list)
    return fold_accuracy

#classify the tree by its threshold values
def predict_on_test_set(test_dictionary,created_tree):
    current_node = created_tree
    while not current_node.leaf:
        if test_dictionary[current_node.root_value] >= current_node.threshold:
            current_node = current_node.root_right
        else:
            current_node = current_node.root_left
    return current_node.root_value

#make predictions by calculating the correct and incorrect predictions
def make_predictions(test_feature_list,test_class_label_list,created_tree):
    accuracy_count= 0.0
    predicted_list = []
    for line in test_feature_list:
        test_dictionary ={}
        for index,value in enumerate(line):
            test_dictionary[index] = value
        prediction = predict_on_test_set(test_dictionary,created_tree)
        predicted_list.append(prediction)
    fold_accuracy = calculate_accuracy(test_class_label_list,predicted_list)
    accuracy_count += fold_accuracy
    return accuracy_count

#to get the reduced data dictionary for every recursive call and continue tp split on this new reduced dictionary
def get_sub_featureclass_list(data_dictionary,split_indices):
    new_sub_dictionary = {}
    for feature in data_dictionary.keys():
        class_labels_list = []
        new_sub_class_value_list = []
        value_n_labels_list = data_dictionary[feature]
        for index,value in enumerate(value_n_labels_list):
            if index not in split_indices:
                new_sub_class_value_list.append(value)
                class_labels_list.append(value[1])
        new_sub_dictionary[feature] = new_sub_class_value_list
    return new_sub_dictionary, class_labels_list

#calculates the entropy
def get_entropy(class_labels):
    class_label_frequency = {}
    attribute_entropy = 0.0
    for labels in class_labels:
        if class_label_frequency.has_key(labels):
            class_label_frequency[labels] += 1
        else:
            class_label_frequency[labels] = 1
    for freq in class_label_frequency.values():
        probability = (-freq/float(len(class_labels)))
        attribute_entropy += probability * math.log(freq/float(len(class_labels)),2)
    return attribute_entropy

#to calculate gini value
def get_gini(class_labels):
    class_label_frequency = {}
    gini_value = 0.0
    for labels in class_labels:
        if class_label_frequency.has_key(labels):
            class_label_frequency[labels] += 1
        else:
            class_label_frequency[labels] = 1
    for freq in class_label_frequency.values():
        gini_value += (freq/float(len(class_labels))) **2
    return (1-gini_value)

#to get the best split value for each feature
def get_best_split_value(feature_list, split_measure):
    feature_values = []
    class_labels = []
    highest_information_gain = 0
    lowest_gini = float('inf')
    split_value = 0
    left_list_indices = []
    right_list_indices = []
    for values in feature_list:
        feature_values.append(values[0])
        class_labels.append(values[1])
    attribute_entropy = get_entropy(class_labels)
    feature_values.sort()
    for fea_val in range(len(feature_list) - 1):
            lesser_threshold_indices = []
            greater_threshold_indices = []
            lesser_threshold_values = []
            greater_threshold_values = []
            current_threshold = float(feature_values[fea_val]+feature_values[fea_val+1])/ 2
            for counter,value_n_label in enumerate(feature_list):
                if value_n_label[0] >= current_threshold:
                    greater_threshold_indices.append(counter)
                    greater_threshold_values.append(value_n_label[1])
                else:
                    lesser_threshold_indices.append(counter)
                    lesser_threshold_values.append(value_n_label[1])
            lesser_attributes_entropy = get_entropy(lesser_threshold_values)
            greater_attributes_entropy = get_entropy(greater_threshold_values)
            if(split_measure == 1):
                current_information_gain = attribute_entropy - (lesser_attributes_entropy*(len(lesser_threshold_indices)/float(len(feature_list)))) \
                                - (greater_attributes_entropy*(len(greater_threshold_indices)/float(len(feature_list))))
                if current_information_gain > highest_information_gain:
                    highest_information_gain = current_information_gain
                    split_value = current_threshold
                    left_list_indices = lesser_threshold_indices
                    right_list_indices = greater_threshold_indices
            else:
                gini_of_less_attribute = get_gini(lesser_threshold_values)
                gini_of_greater_attribute = get_gini(greater_threshold_values)
                current_gini = (gini_of_less_attribute*(len(lesser_threshold_indices)/float(len(feature_list)))) + (gini_of_greater_attribute*(len(greater_threshold_indices)/float(len(feature_list))))
                if(current_gini < lowest_gini):
                    lowest_gini = current_gini
                    split_value = current_threshold
                    left_list_indices = lesser_threshold_indices
                    right_list_indices = greater_threshold_indices
    if(split_measure == 1):
        return highest_information_gain, split_value, left_list_indices, right_list_indices
    else:
        return lowest_gini, split_value, left_list_indices, right_list_indices

#to get the best feature and the best split value for each one on which the data can be split
def get_best_feature(data_dictionary, split_measure):
    best_left_indices = []
    best_right_indices = []
    highest_information_gain = -1
    lowest_gini = float('inf')
    best_split_value = 0
    value_at_node = None
    #loops for the number of feature vectors and gets the best feature and the split value and returns
    for key_val in data_dictionary.keys():
        calculated_measure_value, threshold_to_split, index_left_list, index_right_list = get_best_split_value(data_dictionary[key_val], split_measure)
        if(split_measure == 1):
            if calculated_measure_value > highest_information_gain:
                highest_information_gain = calculated_measure_value
                best_split_value = threshold_to_split
                value_at_node = key_val
                best_left_indices = index_left_list
                best_right_indices = index_right_list
        else:
            if calculated_measure_value < lowest_gini:
                lowest_gini = calculated_measure_value
                best_split_value = threshold_to_split
                value_at_node = key_val
                best_left_indices = index_left_list
                best_right_indices = index_right_list
    return value_at_node,best_split_value,best_left_indices,best_right_indices

#to find the class label with highest frequency
def get_majority_value(training_class_labels_list):
    class_frequency={}
    for label in training_class_labels_list:
        if label in class_frequency.keys():
            class_frequency[label]+=1
        else:
            class_frequency[label]=0
    sorted_class_frequency=sorted(class_frequency.items(),key=operator.itemgetter(1),reverse=True)
    majority_value = sorted_class_frequency[0][0]
    return majority_value

#this method is called for each attribute and it recursively gets best feature, best value to split the data
#by calculating entropy, information gain, gini values and then adds to left and right nodes
#finally returns the generated tree
def create_classficiation_tree(data_dictionary,class_labels,split_measure):
    #check for tree terminating conditions
    #1. If all the class labels are same then return the node
    #2. If the dataset is empty or the attributes list is empty, return the majority class
    #otheriwse proceed to get best features and split the tree
    if len(set(class_labels)) ==1:
        root_node = Node(class_labels[0],0,None,None,True)
        return root_node
    elif not data_dictionary:
        majority_class_label = get_majority_value(class_labels)
        root_node = Node(majority_class_label,0,None,None,True)
        return root_node
    else:
        value_at_node,best_split_value,best_left_indices,best_right_indices = get_best_feature(data_dictionary, split_measure)
        dictionary_of_left, left_class_labels = get_sub_featureclass_list(data_dictionary,best_left_indices)
        dictionary_of_right, right_class_labels = get_sub_featureclass_list(data_dictionary,best_right_indices)
        left_child_node = create_classficiation_tree(dictionary_of_left,left_class_labels,split_measure)
        right_child_node = create_classficiation_tree(dictionary_of_right,right_class_labels,split_measure)
        root_node = Node(value_at_node,best_split_value,left_child_node,right_child_node,False)
        return root_node

#create a dictionary with feature ID as key and all the corresponding values and their class labels as value
def make_dictionary(train_features_list, train_class_labels_list):
    data_dictionary = {}
    feature_range = train_features_list.shape[1]
    for fea_id in range(feature_range):
        feature_set = train_features_list[:,fea_id]
        feature_list =[]
        for counter, val in enumerate(feature_set):
            feature_value = []
            feature_value.append(val)
            feature_value.append(train_class_labels_list[counter])
            feature_list.append(feature_value)
        data_dictionary[fea_id]= feature_list
    return data_dictionary

#creates test and train sets and calls the function that creates dictionary and
#then calls the create decision tree function with each of the created training feature and class labels list
def create_train_and_test_set(normalized_data, split_measure):
    split_row = round(0.1*(normalized_data.shape[0]))
    test_feature_list =[]
    test_class_label_list =[]
    training_features_list = []
    training_class_label_list = []
    accuracy_count = []
    first = 0
    last = split_row
    for kfold in range(10):
        train_temp1 = normalized_data[:first, :]
        train_temp2 = normalized_data[last:, :]
        split_training = np.concatenate((train_temp1, train_temp2), axis=0) #to get rows that will not be test set
        training_class = split_training[:,-1] #gets only the class label
        training_class = training_class.flatten() #flattens into list
        split_training = split_training[:,:-1] #gets only the features from the previously concatenated training set removing class label
        split_training = split_training.astype(np.float)
        split_test = normalized_data[first:last , :] #gets the test data
        test_class = split_test[:, -1] #gets only class label of test dat
        test_class = test_class.flatten()
        split_test = split_test[:,:-1] #gets the features lists of test data
        split_test = split_test.astype(np.float)
        #append the respective lists so that each one of them sent is sent for training and then testing
        test_feature_list.append(split_test)
        test_class_label_list.append(test_class)
        training_features_list.append(split_training)
        training_class_label_list.append(training_class)
        first = last
        last = last+split_row
        data_dictionary = make_dictionary(training_features_list[kfold],training_class_label_list[kfold]) #here the lists of k-fold is sent (for 10 times)
        created_tree = create_classficiation_tree(data_dictionary,training_class_label_list[kfold],split_measure)
        fold_accuracy = make_predictions(test_feature_list[kfold],test_class_label_list[kfold],created_tree)
        accuracy_count.append(fold_accuracy)
        print "Accuracy for fold", kfold, "is: " ,fold_accuracy
    total_accuracy = float(sum(accuracy_count))
    print "Calculated accuracy after 10 fold cross validation is",float(total_accuracy)/10

#to randomize and normalize the dataset for running 10-fold cross validation
def preprocess_data(data_set):
    np.random.shuffle(data_set)
    randomized_data = data_set
    class_label = randomized_data[:,-1]
    class_label_array = class_label.reshape(len(class_label),1)
    attributes_set = randomized_data[:,:-1]
    attribute_array= np.array(attributes_set)
    attribute_array = attribute_array.astype(np.float)
    normalized_data = np.apply_along_axis(lambda value: (value-np.min(value))/float(np.max(value)-np.min(value)), 0 , attribute_array)
    normalized_final_data = np.concatenate((normalized_data,class_label_array),axis=1)
    return normalized_final_data

#load the training data into list
def load_data(filepath):
    dataset = genfromtxt(filepath, delimiter=",",dtype=str)
    return dataset

def main():
    #K:\IUB Sem-4\Data mining- B565\Assignment-2\ques3\iris.csv
    newfile = raw_input("Enter the complete path of the CSV file:")
    split_measure = int(raw_input("Enter 1 for splitting based on Information gain or Enter 2 for splitting based on GINI:"))
    data_set = load_data(newfile)
    normalized_data = preprocess_data(data_set)
    create_train_and_test_set(normalized_data, split_measure)
main()
