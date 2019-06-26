
import csv
import numpy as np
import random
NUMBER_OF_INSTANCES_TO_PREDICT = 286
NUMBER_OF_EVALUATIONS = 1
NUM_OF_UNSUPERVISED_UPDATE_PR = 10

K = 1

# lists of attributes and classes which are relevant for breast-cancer data set
attributes = ['age', 'menopause', 'tumor_size', 'inv_nodes', 'node_caps', 'deg_malig', 'breast', 'breast_quad',
              'irradiat']
classes = ['recurrence-events', 'no-recurrence-events']
first_class_pr = classes[0] + '-pr'
second_class_pr = classes[1] + '-pr'


breast_cancer = []

# we will use it when we will calculate the laplace smoothing
number_of_instances_classes = {}
number_of_attribute_values = {}

priors_pr = {}
# this dict holds the posteriors probabilities
all_attributes_dict = {}

diagonal = {}
diagonal['main_diagonal'] = True
diagonal['second_diagonal'] = False

def main():
    print("uploading the breast-cancer data set ...")
    preprocess()
    print()
    print("supervised model results:")
    print()
    # will build the prior and posteriors probability for the supervised model
    train_supervised()
    # will call the prediction on different instances (several times) and evaluate the results
    evaluate_supervised()

    print()
    print("unsupervised model results:")
    print()
    # will build the prior and posteriors probability for the unsupervised model
    train_unsupervised()


def preprocess():
    """
    This method should open a data file in csv, and transform it into a usable format.
    :return:
    """
    csv_reader = csv.reader(open('breast-cancer.csv', newline=''), delimiter=' ', quotechar='|')
    #count = 0
    for instance in csv_reader:
        # here I can play with distribution of the data to classes by 'class_name': c
        # count += 1
        # if count  % 2 == 0:
        #     c  = 'no-recurrence-events'
        # else:
        #     c = 'recurrence-events'
        instance = ''.join(instance)
        age, menopause, tumor_size, inv_nodes, node_caps, deg_malig, breast, breast_quad, irradiat, class_name = \
            instance.split(",")
        breast_cancer.append({'age': age, 'menopause': menopause, 'tumor_size': tumor_size, 'inv_nodes': inv_nodes,
                              'node_caps': node_caps, 'deg_malig': deg_malig, 'breast': breast,
                              'breast_quad': breast_quad, 'irradiat': irradiat, 'class_name': class_name})

def init_data_structures():

    number_of_attribute_values.clear()
    for attr_name in attributes:
        number_of_attribute_values[attr_name] = {}
        for c in classes:
            number_of_attribute_values[attr_name][c] = 0

    for instance in breast_cancer:
        for attribute_name, attribute_value in instance.items():
            if attribute_value != '?' and attribute_name not in('class_name', first_class_pr, second_class_pr):
                if attribute_name not in all_attributes_dict:
                    all_attributes_dict[attribute_name] = {}

                if classes[0] not in all_attributes_dict[attribute_name]:
                    all_attributes_dict[attribute_name][classes[0]] = {}
                if attribute_value not in all_attributes_dict[attribute_name][classes[0]]:
                    all_attributes_dict[attribute_name][classes[0]][attribute_value] = 0
                    number_of_attribute_values[attribute_name][classes[0]] += 1

                if classes[1] not in all_attributes_dict[attribute_name]:
                    all_attributes_dict[attribute_name][classes[1]] = {}
                if attribute_value not in all_attributes_dict[attribute_name][classes[1]]:
                    all_attributes_dict[attribute_name][classes[1]][attribute_value] = 0
                    number_of_attribute_values[attribute_name][classes[1]] += 1


def deterministically_init():
    for instance in breast_cancer:
        if random.random() > 0.5:
            instance['guessed_class'] = classes[0]
        else:
            instance['guessed_class'] = classes[1]

    count_correct = 0
    count_class1 = 0
    for instance in breast_cancer:
        if instance['guessed_class'] == classes[0]:
            count_class1 += 1
        if instance['guessed_class'] == instance['class_name']:
            count_correct += 1

    print("random give us : " + str(count_class1) + " of " + "'" + classes[0] + "'" + 'and ' +
          str(len(breast_cancer) - count_class1) + ' of class ' + "'" + classes[1] + "'")
    print("Accuracy deterministically labelling % : " + str(count_correct / len(breast_cancer) * 100))


def train_unsupervised():

    det_init = False

    # deterministically labelling the instances
    if det_init:
        deterministically_init()
    else:

        # calculate the prior probabilities that will use us for calculate the new instances's probabilities
        # and then for the prediction
        prior_pr1, prior_pr2 = calc_priors_unsupervised(True)
        priors_pr[first_class_pr] = prior_pr1
        priors_pr[second_class_pr] = prior_pr2

        # iteration phase of calculating the posteriors probabilities and then update the instances's probabilities
        # and calculate again ...
        for i in range(NUM_OF_UNSUPERVISED_UPDATE_PR):
            calc_posteriors_unsupervised()
            update_classes_pr()
            prior_pr1, prior_pr2 = calc_priors_unsupervised(False)
            priors_pr[first_class_pr] = prior_pr1
            priors_pr[second_class_pr] = prior_pr2

            # will call the prediction on different instances (several times) and evaluate the results
            if i == 0:
                evaluate_unsupervised(True)
            else:
                evaluate_unsupervised(False)


def train_supervised():
    prior_pr1, prior_pr2 = calc_priors_supervised()
    priors_pr[first_class_pr] = prior_pr1
    priors_pr[second_class_pr] = prior_pr2
    calc_posteriors_supervised()


def calc_priors_unsupervised(is_first_time):
    """
    calculate prior probabilities for unsupervised.
    first, for each instance, for each class we will give a random probability between 0 and 1, s.t the sum
    of classes probability is 1.
    then, we will sum up the probabilities for each class.
    finally, we will divide the sum by number of instances to get the prior probability for the class.
    :return: prior probabilities
    """

    if is_first_time:
        s = np.random.dirichlet(np.ones(2), len(breast_cancer))
        for i in range(len(s)):
            breast_cancer[i][first_class_pr] = s[i][0]
            breast_cancer[i][second_class_pr] = s[i][1]

    count_class1 = 0
    count_class2 = 0

    # sum up the probabilities for each class
    for instance in breast_cancer:
        count_class1 += instance[first_class_pr]
        count_class2 += instance[second_class_pr]

    sum_entries = len(breast_cancer)
    number_of_instances_classes[classes[0]] = int(count_class1)
    number_of_instances_classes[classes[1]] = int(count_class2)

    prior_pr_class1 = count_class1 / sum_entries
    prior_pr_class2 = count_class2 / sum_entries

    return prior_pr_class1, prior_pr_class2


def calc_priors_supervised():
    count_class1 = 0
    count_class2 = 0

    for instance in breast_cancer:
        if instance['class_name'] == classes[1]:
            count_class2 += 1
        elif instance['class_name'] == classes[0]:
            count_class1 += 1
        else:
            pass

    sum_entries = len(breast_cancer)

    number_of_instances_classes[classes[0]] = count_class1
    number_of_instances_classes[classes[1]] = count_class2

    prior_pr_class1 = count_class1 / sum_entries
    prior_pr_class2 = count_class2 / sum_entries

    return prior_pr_class1, prior_pr_class2


def calc_posteriors_unsupervised():
    """
    save in all_attributes_dict the counters of the different attribute value of the different attributes
    of the different classes. if it's first time we encountered the arrtibute value, we will set the dictionary
    in the suitable place to be 0. otherwise, we will add the probability of the class for the correct attribute
    name and attribute value in dictionary.
    :return:
    """

    init_data_structures()

    for instance in breast_cancer:
        for attribute_name, attribute_value in instance.items():
            if attribute_value != '?' and attribute_name not in (first_class_pr, second_class_pr, 'class_name'):
                all_attributes_dict[attribute_name][classes[0]][attribute_value] += instance[first_class_pr]
                all_attributes_dict[attribute_name][classes[1]][attribute_value] += instance[second_class_pr]

    for attr_name, attr_dict in all_attributes_dict.items():
        for class_name, class_dict in attr_dict.items():
            for attr_value, count in class_dict.items():
                if attr_value != '?':
                    all_attributes_dict[attr_name][class_name][attr_value] = \
                        count / number_of_instances_classes[class_name]

    # calc_add_k_smoothing()


def calc_posteriors_supervised():
    """
    Fill the all_attributes_dict with the counters of posteriors on the fly
    :return:
    """
    # an important point is that the number of attributes values for an attribute is different between the classes
    # so in any case we want to add all of the attribute values also to the other class dictionary.
    # but update number_of_attribute_values and all_attributes_dict[attribute_name][c][attribute_value] according
    # to the correct class of the instance

    init_data_structures()

    for instance in breast_cancer:
        class_name = instance['class_name']
        for attribute_name, attribute_value in instance.items():
            if attribute_value != '?' and attribute_name != 'class_name':
                all_attributes_dict[attribute_name][class_name][attribute_value] += 1

    # for attr_name, attr_dict in all_attributes_dict.items():
    #     for class_name, class_dict in attr_dict.items():
    #         for attr_value, count in class_dict.items():
    #             if attr_value != '?':
    #                 all_attributes_dict[attr_name][class_name][attr_value] = \
    #                     count / number_of_instances_classes[class_name]

    calc_add_k_smoothing()


def calc_add_k_smoothing():
    for attr_name, attr_dict in all_attributes_dict.items():
        for class_name, class_dict in attr_dict.items():
            for attr_value, count in class_dict.items():
                if attr_value != '?':
                    all_attributes_dict[attr_name][class_name][attr_value] = \
                    (count + K) / ((K * number_of_attribute_values[attr_name][class_name]) +
                                   number_of_instances_classes[class_name])

def evaluate_supervised():
    """

    :return:
    """

    count_correct = 0
    count_all = 0
    sum = len(breast_cancer)
    arr = np.arange(sum)

    for i in range(NUMBER_OF_EVALUATIONS):

        np.random.shuffle(arr)
        # shuffle the arr of numbers of instances
        instances_to_predict = []
        # we will take the first NUMBER_OF_INSTANCES_TO_PREDICT places from arr
        # we will append to instances_to_predict these instances
        for idx in arr[0:NUMBER_OF_INSTANCES_TO_PREDICT]:
            instances_to_predict.append(breast_cancer[idx])

        count_correct += predict_supervised(instances_to_predict)
        count_all += len(instances_to_predict)

    print("Number of evaluations(=set of predictions): " + str(NUMBER_OF_EVALUATIONS))
    print("Number of instances predicted in one prediction: " + str(NUMBER_OF_INSTANCES_TO_PREDICT))
    print("Total instances predicted from all of the evaluations: " +
          str(NUMBER_OF_INSTANCES_TO_PREDICT * NUMBER_OF_EVALUATIONS))
    print("Count correct : " + str(count_correct))
    percent_correct = count_correct / count_all
    print("Accuracy in percents %: " + str(int(percent_correct * 100)))


def predict_supervised(instance_list):
    """
    :param instance_list: list of instances we want to predict the class
    :return: count correct predictions of instances
    """
    count_correct = 0

    for instance in instance_list:
        pr_class1 = priors_pr[first_class_pr]
        pr_class2 = priors_pr[second_class_pr]

        for attr_name, attr_value in instance.items():
            if attr_value != '?' and attr_name != 'class_name':
                pr_class1 *= all_attributes_dict[attr_name][classes[0]][attr_value]
                pr_class2 *= all_attributes_dict[attr_name][classes[1]][attr_value]

        if pr_class1 >= pr_class2:
            predicted_class = classes[0]
        else:
            predicted_class = classes[1]

        if instance['class_name'] == predicted_class:
            count_correct += 1

    return count_correct


def evaluate_unsupervised(is_first_time):
    """
    Calls prediction on random instances by using shuffle.
    and returns confusion matrix of the result (true positive, true negative, false negative, false positive)
    :return:
    """

    confusion_matrix = []
    confusion_matrix.append([0, 0])
    confusion_matrix.append([0, 0])
    count_all = 0
    sum = len(breast_cancer)
    arr = np.arange(sum)

    for i in range(NUMBER_OF_EVALUATIONS):
        # shuffle the arr of numbers of instances
        np.random.shuffle(arr)
        instances_to_predict = []
        # we will take the first NUMBER_OF_INSTANCES_TO_PREDICT places from arr
        # we will append to instances_to_predict these instances
        for idx in arr[0:NUMBER_OF_INSTANCES_TO_PREDICT]:
            instances_to_predict.append(breast_cancer[idx])

        true_pos, true_neg, false_pos, false_neg = predict_unsupervised(instances_to_predict)
        confusion_matrix[1][1] += true_pos
        confusion_matrix[0][0] += true_neg
        confusion_matrix[1][0] += false_neg
        confusion_matrix[0][1] += false_pos

        count_all += len(instances_to_predict)

    # print the confusion matrix and the accuracy
    print("Number of evaluations(=set of predictions): " + str(NUMBER_OF_EVALUATIONS))
    print("Number of instances predicted in one prediction: " + str(NUMBER_OF_INSTANCES_TO_PREDICT))
    print("Total instances predicted from all of the evaluations: " +
          str(NUMBER_OF_INSTANCES_TO_PREDICT * NUMBER_OF_EVALUATIONS))
    print("true negative, false positive: " + str(confusion_matrix[0]))
    print(" false negative, true positive: " + str(confusion_matrix[1]))

    # take the best result between the two classes (swapping if necessary)

    if is_first_time:
        if confusion_matrix[1][1] + confusion_matrix[0][0] >= confusion_matrix[0][1] + confusion_matrix[1][0]:
            diagonal['main_diagonal'] = True
            diagonal['second_diagonal'] = False
            diagonal_sum = confusion_matrix[1][1] + confusion_matrix[0][0]
        else:
            diagonal['main_diagonal'] = False
            diagonal['second_diagonal'] = True
            diagonal_sum = confusion_matrix[0][1] + confusion_matrix[1][0]

    else:
        if diagonal['main_diagonal']:
            diagonal_sum = confusion_matrix[1][1] + confusion_matrix[0][0]
        else:
            diagonal_sum = confusion_matrix[0][1] + confusion_matrix[1][0]

    accuracy = int(diagonal_sum / count_all * 100)
    print("accuracy in percents % : " + str(accuracy))


def update_classes_pr():
    """
    for each instance we will find class distribution (probaility for first class and for the second)
    :return:
    """

    for instance in breast_cancer:
        pr_class1 = priors_pr[first_class_pr]
        pr_class2 = priors_pr[second_class_pr]
        for attr_name, attr_value in instance.items():
            if attr_value != '?' and attr_name not in (first_class_pr, second_class_pr, 'class_name'):
                pr_class1 *= all_attributes_dict[attr_name][classes[0]][attr_value]
                pr_class2 *= all_attributes_dict[attr_name][classes[1]][attr_value]

        # normalize, to find new class distribution for the instance
        sum_pr = pr_class1 + pr_class2
        pr_class1 /= sum_pr
        pr_class2 /= sum_pr

        # update the instance with the new probabilities
        instance[first_class_pr] = pr_class1
        instance[second_class_pr] = pr_class2


def predict_unsupervised(instance_list):
    """
    :param instance_list: list of instances we want to predict the class
    :return: confusion matrix for these instances (#true_positive, #true_negative, #false_negative, #false_positive)
    """
    true_positive = 0
    true_negative = 0
    false_negative = 0
    false_positive = 0

    for instance in instance_list:
        pr_class1 = priors_pr[first_class_pr]
        pr_class2 = priors_pr[second_class_pr]

        for attr_name, attr_value in instance.items():
            if attr_value != '?' and attr_name not in (first_class_pr, second_class_pr, 'class_name'):
                pr_class1 *= all_attributes_dict[attr_name][classes[0]][attr_value]
                pr_class2 *= all_attributes_dict[attr_name][classes[1]][attr_value]

        if pr_class1 >= pr_class2:
            predicted_class = classes[0]
        else:
            predicted_class = classes[1]

        if instance['class_name'] == classes[0] and predicted_class == classes[0]:
            true_positive += 1

        if instance['class_name'] == classes[1] and predicted_class == classes[1]:
            true_negative += 1

        if instance['class_name'] == classes[0] and predicted_class == classes[1]:
            false_negative += 1

        if instance['class_name'] == classes[1] and predicted_class == classes[0]:
            false_positive += 1

    return true_positive, true_negative, false_positive, false_negative

main()
"""

I did this project by myself: Oren Sultan.
I used python version 3.5
Questions:
1. 

What I did in order to answer this question, I changed the data set of breast_cancer in different ways:
1. source data 'no-recurrence-events' is 70% of the classes and 'recurrence-events' 30%
2. extreme dominant class for example all of instances are 'no-recurrence-events'
3. normal distribution 50% of instances 'no-recurrence-events' and 50% 'recurrence-events'

-We expect the prior probabilities to be close to 50% for each class (because each class is a sum of 
probabilities that chose randomely).
Indeed I confirmed it in we most of the time will get priors in range 45%-55% for each class.

now,
- if data in normal distribution (actual 50%-50%) we will always get accuracy in the initial phase of close to 
50% and after iterations also close to 50%(because the priors and the actual are really very close)

- if it's exterme dominant class for example all of classes 'no-recurrence-events'.
as the prior of 'no-recurrence-events' is higher than 50%, the accuracy of initial phase will be better. e.g
true negative, false positive: [204, 82]
 false negative, true positive: [0, 0]
accuracy in percents % : 71

What I noticed is that when you have exterme dominant class you are more sensible to priors which are far from the 
truth. for example 'recurrence-events' will get prior 0.9 and the 
'no-recurrence-events' will get 0.1.
true negative, false positive: [0, 286]
 false negative, true positive: [0, 0]
-> 0/286 = 0% (of course with swapping we will get 100% accuracy)

On the other hand if your data is distributed 50%-50% between classes so if the priors you get are very far from 50%. 
e.g 0.95 and 0.05 (doesnt matter which class get which pr) it will be close to 50% accuracy. 

I have no math proofs for all of it by I thought about it alone and confirmed it.

Why does it work?
First we have the iterations of update of probabilities which is implemented and thanks to it we can get better 
probabilities we are more accurate than random, and converge to a better accuracy in the end.
In addition I implemented the swapping (just take the higest sum of diagonal in my 2*2 confusion matrix, and then we 
stick to the class with higher probability after iteration 0 after swapping if needed)

In addition, we have dominant class which is 'no-recurrence-events' (it's not so dominant but still 70% is relatively 
higher than normal of 50% dist) that's what give us the oppertunity to get accuracy which is much higher than 50% in 
the end, in contrast to distribution of 50%-50% that as I demonstrated will give us very close to 50% accuracy.


When it will fail?
We choose the initial prior randomely, if prior probabilities that will be choose are very far from the actual, 
for example we have class A and class B and the actual is P(A) = 0.99 and P(B) = 0.01 if we will change it to be 
P(A) = 0.01 and P(B) = 0.99 (we took the complement prob.)
so without swapping the initial phase will give for first iteration very bad.
for example:
for 
priors_pr[first_class_pr] = 0.99
priors_pr[second_class_pr] = 0.01
true negative, false positive: [0, 201]
false negative, true positive: [0, 85]
so we have (0+85)/286 = 85/286 = 30%

and as we have more dominant class it will be more sensible - more severe.

so for dataset with low number of instances (we have only 286) it will be hard to converge for better accuracy also 
if we have iterations.




4. I implemented add-k smoothing, the method in the code is called "calc_add_k_smoothing"
and I run it on supervised model on breast-cancer and I notice differences such as
K is more larger it may damage the accuracy and make it lower accuracy.
for example:
k = 1 , (prediction on all instances, one evaluation) gives accuracy = 75%
and as k is higher, the accuracy is lower.
( k = 10, accuracy = 74%
  k = 100, accuracy = 72%
  ...
)


I also noticed that there is some limit how it can affect the accuracy to be good or bad.
for example no matter what is k, I can't get better accuracy than 75%.
So yes I see a change from the original implementation which is without smoothing.
the change is depends on k, if it's a very large k I will see much more dramatic difference.

Explaination:
when K is large it assigns too much probability to unseen events, that's why the result we get may be a 
distortion of the reality. as k is larger the calculation will be less accuracte.
On the other hand if k is small enough (like epsilon) the effect will be neglegible, so it will give 0.00001 
probability for a 0 probability for example, that's why the result will be very close
to reality.
In addition, in this data set zero-probability is rare, there is no much different between add-k with small 
k (1 for example) to implementation without smoothing.
also the accuracy without smoothing is 75% as I got with add-1 smoothing.

To conclude:
if in your data set you may get a lot of zero - probabilities, use smoothing with low K it will 
improve your accuracy, if no zero-probabilities avoid use smoothing because you will not get from his advantage and 
you are exposed to his disadvantages.

5. 

I implemented a method which is called deterministically_init() to check it
you should assign det_init = True.

confirmation:

As we can see the accuracy close to 50% in each run (in range 45%-55% most of the time)
And we have seen that the unsupervised NB will give us 65-70% most of the time (which is at most 
10% less than the supervised). So, we see that deterministically_init is most of the time
worse than the guessing method of probability of the instances which we implemented.
So confirmation OK.

demonstration:

Since we choose randomely the class for each instance we expect to get approx 50% instances with
class 'recurrence-events'and 50% with class 'no-recurrence-events'.
But in reality our data is not distribute to classes 50%, 50%.
in breast-cancer for example (201 of 286 instances belongs to the class 'no-recurrence-events'
and 85 to 'recurrence-events'), in addition also if we are correct about distribution of 
instances to classes, let's assume reality is also close to 50%, 50% so we have expectation
to be right in our guess on half of the instances (accuracy close to 50%)

if for example for class 'no-recurrence-events' we got more than 50% so the accuracy suppose to be higher 
because most of the instances actual class is 'no-recurrence-events' (201 in 286)
and if less than 50% so the accuracy suppose to be worse.
- I say suppose to be because, the number of 'no-recurrence-events' is increased, but maybe we have bad luck 
and we gave to more wrong instances the class 'no-recurrence-events' and they 
actually 'recurrence-events'.

now, probability of instance class to be like actual_class = 
the probability the actual and the instance class we choose randomly are 'no-recurrence-events'
+
the probability the actual and the instance class we choose randomly are 'recurrence-events'

(it's actually union between groups but because of disjoint set : P(A union B) = P(A)+P(B))

P(actual_class = 'no-recurrence-events', instance_class = 'no-recurrence-events') =

201/286 * 1/2 = 0.35

P(actual_class = 'recurrence-events', instance_class = 'recurrence-events') =
100/286 * 1/2 = 0.17
- I can multiply the probabilities because of independence, I choose instance class randomly

P(actual_class = 'no-recurrence-events', instance_class = 'no-recurrence-events')
+ P(actual_class = 'recurrence-events', instance_class = 'recurrence-events') = 0.35 + 0.17 = 0.52 = 
probability of instance class to be like actual_class

so, the distribution of classes is close to 50% instances class 'no-recurrence-events' and
50% 'recurrence-events' we will be close to 52% accuracy according to this data set.

So what I demonstrated is that we will get accuracy which is close to 50% (most of the time in range 45%-55%) 
and that's why we will prefer not to choose it deterministically.

"""
