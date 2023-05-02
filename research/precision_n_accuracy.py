import random
# set the probability of the event happening
p_h = 0.6
# probability in random guessing
p_p = 0.5

# set the total number of guesses
n = 1000000

# count the number of correct guesses
true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0

for i in range(n):
    guess = random.random()  # generate a random guess between 0 and 1
    actual = random.random() # generate a random number between 0 and 1
    if actual < p_h:
        happen = True
    else:
        happen = False

    if guess < p_p:
        pred = True
    else:
        pred = False
    
    if happen and pred:
        true_positive += 1
    elif happen and not pred:
        false_negative += 1
    elif not happen and pred:
        false_positive += 1
    elif not happen and not pred:
        true_negative += 1
'''
true_positive == total_positive * p_h
false_positive == total_positive * (1 - p_h)
'''
precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)
accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
print(f"precision: {precision*100:.2f}%")
print(f"recall: {recall*100:.2f}%")
print(f"accuracy: {accuracy*100:.2f}%")

''' 
    conclusion: 
    when it comes to random guessing, precision is always the same as the rate at which event happens.
    accuracy however, is affected by how prediction probability is set.
'''