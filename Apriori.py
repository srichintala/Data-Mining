import csv
import itertools
import operator
import numpy as np
import math
from collections import defaultdict

#This function uses Fk-1 * F1 method to generate candidate itemsets at each level
def apriori_freq_item_gen1(freq_itemsets_k_1, k):
    final_freq_k_itemset = []
    for item1 in range(len(freq_itemsets_k_1)):
        for item2 in range(len(freq_itemsets_level1)):
            item_list1 = list(freq_itemsets_k_1[item1])
            item_list2 = list(freq_itemsets_level1[item2])
            item_list1.sort()
            item_list2.sort()
            if not (freq_itemsets_level1[item2].issubset(freq_itemsets_k_1[item1])):
                if ((freq_itemsets_level1[item2]|freq_itemsets_k_1[item1]) not in final_freq_k_itemset):
                        #print freq_itemsets_level1[item2]|freq_itemsets_k_1[item1]
                        final_freq_k_itemset.append(freq_itemsets_level1[item2]|freq_itemsets_k_1[item1])
    return final_freq_k_itemset

#This function uses Fk-1 * Fk-1 method to generate the candidate itemsets at each level
def apriori_freq_item_gen2(freq_itemsets_k_1, k):
    final_freq_k_itemset = []
    for item1 in range(len(freq_itemsets_k_1)):
        for item2 in range(item1+1, len(freq_itemsets_k_1)):
            item_list1 = list(freq_itemsets_k_1[item1])[:k-2]
            item_list2 = list(freq_itemsets_k_1[item2])[:k-2]
            item_list1.sort()
            item_list2.sort()
            if item_list1 == item_list2:
                final_freq_k_itemset.append(freq_itemsets_k_1[item1]|freq_itemsets_k_1[item2])
    return final_freq_k_itemset

#For each candidate itemset generated at each level, this method gets the counts of each item and returns
#a list of frequent itemsets
def get_freq_itemsets_each_level(total_transactions, candidates_size_1, minSupport, num_of_transactions):
    cand_item1_count = {}
    freq_item1_with_support = {}
    freq_items_list = []
    for each_transaction in total_transactions:
        for each_cand in candidates_size_1:
            if each_cand.issubset(each_transaction):
                if not cand_item1_count.has_key(each_cand):
                    cand_item1_count[each_cand] = 1
                else:
                    cand_item1_count[each_cand] += 1
    for key in cand_item1_count:
        support=cand_item1_count[key]/num_of_transactions
        if support >= minSupport:
            freq_items_list.append(key)
            freq_item1_with_support[key]=support
    return freq_items_list, freq_item1_with_support

#After getting the 1-itemset, this algorithm runs by generating frequent itemsets at every level
#and proceeding to the next level until no more frequent itemsets are generated. It returns the
#frequent itemsets in a list of list and also a dictionary with support of each frequent itemset.
#It prints the number of candidate itemsets and frequent itemsets generated for each method.
def apriori_algorithm(item_sets, freq_itemsets_level1, freq_itemsets_with_count, k, min_support, num_of_transactions, method_type):
    final_freq_itemsets = [freq_itemsets_level1]
    gen_freq_itemsets = 0
    gen_candidate_sets = 0
    print "Level 1 itemsets: "
    for i in range(len(freq_itemsets_level1)):
            print list(freq_itemsets_level1[i]),
    while len(final_freq_itemsets[k-2])>0:
        if(method_type == 1):
            candidates_level_k = apriori_freq_item_gen1(final_freq_itemsets[k-2], k)
        else:
            candidates_level_k = apriori_freq_item_gen2(final_freq_itemsets[k-2], k)
        gen_candidate_sets = gen_candidate_sets + len(candidates_level_k)
        freq_itemsets_level_k, freq_itemsets_level_k_with_support = get_freq_itemsets_each_level(item_sets, candidates_level_k, min_support, num_of_transactions)
        print "\n", "Level", k, "itemsets: "
        final_freq_itemsets.append(freq_itemsets_level_k)
        for j in (final_freq_itemsets[k-1]):
            print list(j),
        freq_itemsets_with_count.update(freq_itemsets_level_k_with_support)
        k = k+1
    for i in range(len(final_freq_itemsets)):
        for j in final_freq_itemsets[i]:
            gen_freq_itemsets = gen_freq_itemsets + 1
    print "Number of Frequent itemsets generated: ", gen_freq_itemsets
    print "Candidate Itemsets generated: ", gen_candidate_sets
    return final_freq_itemsets, freq_itemsets_with_count

#After generating the frequent itemsets, this method enumerates over each itemset
#to find the maximal itemsets by checking if any of the items super set is frequent.
#If the super set of an item is not in frequent itemlist then it appends that item
#to list of maximal itemsets and prints the number of maximal elements.
def find_maximal_itemsets(final_freq_itemsets):
    maximal_sets = []
    for i in range(len(final_freq_itemsets)-1):
        current_set = final_freq_itemsets[i]
        super_set = final_freq_itemsets[i+1]
        for each_item in range(len(current_set)):
            maximal = True
            for each_item2 in range(len(super_set)):
                if current_set[each_item].issubset(super_set[each_item2]):
                    maximal = False
            if(maximal == True):
                maximal_sets.append(current_set[each_item])
    print "Total Number of Maximal Itemsets: ", len(maximal_sets)
    return maximal_sets

#After generating the frequent itemsets, this method enumerates over each itemset
#to find the closed itemsets by checking if any of the items super set has same support as its.
#If the support of item does not match with any of its super set in frequent itemlist
#then it appends that item to list of closed itemsets and prints the number of closed itemsets.
def find_closed_itemsets(final_freq_itemsets, final_freq_itemsets_with_counts):
    closed_itemsets = []
    for i in range(len(final_freq_itemsets)-1):
        current_set = final_freq_itemsets[i]
        super_set = final_freq_itemsets[i+1]
        for each_item in range(len(current_set)):
            maximal = True
            for each_item2 in range(len(super_set)):
                if (current_set[each_item].issubset(super_set[each_item2])) & (final_freq_itemsets_with_counts[current_set[each_item]] == final_freq_itemsets_with_counts[super_set[each_item2]]):
                    maximal = False
            if(maximal == True):
                closed_itemsets.append(current_set[each_item])
    print "Total Number of Closed Itemsets: ", len(closed_itemsets)
    return closed_itemsets

#This method generates rules based on confidence or lift as given by the user.
def generate_rules(final_freq_itemsets, final_freq_itemsets_with_counts, threshold, string, unique_items):
    list_of_rules = []
    total_rules = []
    cand_rules = 0
    brute_force_rules = 0
    print "\n", "-----------------------rules--------------------------", "\n"
    for level in range(1, len(final_freq_itemsets)):
        for freq_itemset in final_freq_itemsets[level]:
            item_1_consequents = [frozenset([item]) for item in freq_itemset]
            total_rules, cand_rules = apriori_rules_gen(freq_itemset, final_freq_itemsets_with_counts, item_1_consequents, threshold, list_of_rules, level, string, cand_rules)
    sorted_rules = sorted(total_rules,key=operator.itemgetter(2),reverse=True)
    if len(sorted_rules)>= 10:
        for rule in range(10):
            r = sorted_rules[rule]
            if(string == "confidence"):
                print r[0], "-->", r[1], "Confidence: ", r[2]
            else:
                print r[0], "-->", r[1], "Lift: ", r[2]
    else:
        for rule in range(len(sorted_rules)):
            r = sorted_rules[rule]
            if(string == "confidence"):
                print r[0], "-->", r[1], "Confidence: ", r[2]
            else:
                print r[0], "-->", r[1], "Lift: ", r[2]
    for i in range(len(final_freq_itemsets)):
        for j in range(len(final_freq_itemsets[i])):
            size = len(final_freq_itemsets[i][j])
            r = int(math.pow(2,size)) - 2
            brute_force_rules = brute_force_rules +r
    print "Total rules generated after confidence pruning: ", len(total_rules)
    print "Total candidate rules generated: ", cand_rules
    print "Total rules generated by brute force method: ", brute_force_rules
    return total_rules

#This function prunes the candidate rules to find confident rules and returns the pruned rules
def get_pruned_rules(freq_K_itemset, final_freq_itemsets_with_counts, item_1_consequents, threshold, list_of_rules, level, string, cand_rules):
    pruned_rules = []
    cand_rules = cand_rules + len(item_1_consequents)
    for each_h_m_plus_1 in item_1_consequents:
        if(string == "confidence"):
            rule_confidence = final_freq_itemsets_with_counts[freq_K_itemset]/final_freq_itemsets_with_counts[freq_K_itemset-each_h_m_plus_1]
            if rule_confidence >= threshold:
                #print "Rule: ", list(freq_K_itemset-each_h_m_plus_1),'->',list(each_h_m_plus_1),'Confidence:',rule_confidence
                list_of_rules.append((freq_K_itemset-each_h_m_plus_1, each_h_m_plus_1, rule_confidence))
                pruned_rules.append(each_h_m_plus_1)
        else:
            antecedent = freq_K_itemset-each_h_m_plus_1
            rule_confidence = final_freq_itemsets_with_counts[freq_K_itemset]/final_freq_itemsets_with_counts[antecedent]
            consequent_support = final_freq_itemsets_with_counts[each_h_m_plus_1]
            lift_value = rule_confidence/consequent_support
            if lift_value >= threshold:
                #print "Rule: ", list(freq_K_itemset-each_h_m_plus_1),"->",list(each_h_m_plus_1),"Lift: ", lift_value
                list_of_rules.append((freq_K_itemset-each_h_m_plus_1, each_h_m_plus_1, lift_value))
                pruned_rules.append(each_h_m_plus_1)
    return list_of_rules, pruned_rules, cand_rules

#This method gets the item-1 cnsequent rules and prunes them and then generates combinations of rules
#if the length of pruned rules >= 1 and finally returns the rules if their length is equal to length of itemset
#i.e., no more combinations can be generated.
#If lift is given as input, it generates all the combinations returns the list of rules
def apriori_rules_gen(freq_K_itemset, final_freq_itemsets_with_counts, item_consequents, threshold, list_of_rules, level, string, cand_rules):
    k = len(freq_K_itemset)
    if(item_consequents != []):
        m = len(item_consequents[0])
    else:
        return list_of_rules, cand_rules
    if k > m:
        list_of_rules, pruned_rules, cand_rules = get_pruned_rules(freq_K_itemset, final_freq_itemsets_with_counts, item_consequents, threshold, list_of_rules, level, string, cand_rules)
        if k == m+1:
            return list_of_rules, cand_rules
        if(string == "confidence"):
            if(len(pruned_rules)>1):
                H_m_plus_1 = apriori_freq_item_gen2(pruned_rules, m+1)
                apriori_rules_gen(freq_K_itemset, final_freq_itemsets_with_counts, H_m_plus_1, threshold, list_of_rules, level, string, cand_rules)
        else:
            if(m+1 >= 2):
                H_m_plus_1 = apriori_freq_item_gen2(item_consequents, m+1)
                apriori_rules_gen(freq_K_itemset, final_freq_itemsets_with_counts, H_m_plus_1, threshold, list_of_rules, level, string, cand_rules)
        return list_of_rules,cand_rules

#This function creates the itemsets from the binary list where there is 1
# eg: [[1, 2, 3, 5, 6..], [2, 4, 5, 6, 8..]...} and itemset1
def get_transactions(binary_item_list):
    total_transactions = []
    for row in range(len(binary_item_list)):
        each_itemlist = binary_item_list[row]
        items_set = []
        for index in range(len(each_itemlist)):
            if each_itemlist[index] == 1:
                items_set.append(index+1)
        total_transactions.append(items_set)
    itemset_1 = []
    for transaction in total_transactions:
        for each_item in transaction:
            if [each_item] not in itemset_1:
                itemset_1.append([each_item])
    itemset_1.sort()
    num_of_transactions = float(len(total_transactions))
    return list(map(frozenset,itemset_1)), list(map(set,total_transactions)), num_of_transactions

#to create binary transaction matrix this function creates a dictionary
#of unique items in each column. The items are mapped to integer
# Eg: {high:1, vhigh:2, med:3, low:4, ...}
#a list of lists is used to store the binary format of transactions
#[[1 0 0 1 0 ...][0 1 0 0 0 1...]...]
def make_binary_matrix(dataset):
    np_array = np.array(dataset)
    unique = []
    count = 0
    final_dict = []
    for i in range(len(np_array[0])):
        item = set(np_array[:,i])
        data_dict = {}
        for j in item:
            unique.append(j)
            data_dict[j] = count
            count = count + 1
        final_dict.append(data_dict)
    binary_item_list = [[0 for x in range(len(unique))] for x in range(len(dataset))]
    for i, item in enumerate(np_array):
        for i2 in range(len(item)):
            if final_dict[i2].has_key(item[i2]):
                binary_item_list[i][final_dict[i2][item[i2]]] = 1
    return len(unique), binary_item_list

#loads the dataset
def loadDataSet(file_name):
    with open(file_name, 'rb') as f:
        reader = csv.reader(f)
        dataset = list(reader)
    return dataset

if __name__ == '__main__':
    k = 2
    min_support = 0.1
    confidence = 0.6
    lift = 0.7
    conf_string = "confidence"
    lift_string = "lift"
    measure = 1
    generated_rules = []
    while True:
        try:
            method_type = int(raw_input("Please enter 1 to choose Fk-1 * F1 method or 2 to choose Fk-1 * Fk-1 method: "))
            measure = int(raw_input("Please enter 1 for using Confidence as measure or 2 for using Lift as measure: "))
            if(measure == 1):
                min_support = float(raw_input("Please enter a minimum support Eg: 0.6: "))
                confidence = float(raw_input("Please enter minimum confidence Eg: 0.6: "))
            elif(measure == 2):
                min_support = float(raw_input("Please enter a minimum support Eg: 0.6: "))
                lift = float(raw_input("Please enter a minimum lift Eg: 0.6: "))
            break

        except ValueError:
            print "Invalid input: Enter correct input values"

    dataset = loadDataSet("K://IUB Sem-4//Data mining- B565//Assignment-4//question-3//nursery_data.txt")
    unique_items, binary_item_list = make_binary_matrix(dataset)
    candidates_size_1, item_sets, num_of_transactions = get_transactions(binary_item_list)
    freq_itemsets_level1, freq_itemset1_with_count = get_freq_itemsets_each_level(item_sets, candidates_size_1, min_support, num_of_transactions)
    final_freq_itemsets, final_freq_itemsets_with_counts = apriori_algorithm(item_sets, freq_itemsets_level1, freq_itemset1_with_count, k, min_support, num_of_transactions, method_type)
    if(measure == 1):
        generated_rules = generate_rules(final_freq_itemsets, final_freq_itemsets_with_counts, confidence, conf_string, unique_items)
    elif(measure == 2):
        generated_rules = generate_rules(final_freq_itemsets, final_freq_itemsets_with_counts, lift, lift_string, unique_items)
    maximal_itemsets = find_maximal_itemsets(final_freq_itemsets)
    closed_itemsets = find_closed_itemsets(final_freq_itemsets, final_freq_itemsets_with_counts)

def main():
    main()