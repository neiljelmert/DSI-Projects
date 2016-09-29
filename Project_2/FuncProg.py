######################################################################

# Project 2

import string

################################################################################################

# Lambda Functions

# Before compiling the following code snippets, write down what
# each individual lambda function will return in an inline comment.
# If you think it returns an error, why would it be the case.


(lambda val: val ** 2)(2)
# will return 4

(lambda x, y: x + y)(2, 3)
# will return 5

#(lambda x, y: x + y)((2, 3))
# will return error because (2,3) is one object and lambda is asking for two args

ga = (lambda s: s if 'General' in s else 'Specific ' + s)
ga('General Assembly') # Will return General Assembly
ga('Assembly') # Will return Specific Assembly

################################################################################################

# Comprehensions

'''
Exercise: Transform the following piece of code into a list comprehension.
letters = string.ascii_uppercase
letters_idx = []
for letter in letters:
    letters_idx.append(letters.index(letter))
print letters_idx
'''

letters = string.ascii_uppercase
letters_idx = [letters.index(letter) for letter in letters]
print letters_idx


'''
Exercise: Create a Cartesian product of t-shirt colors/sizes using a list comprehension.
Inputs:
colors = ['black', 'white']
sizes = ['S', 'M', 'L']

Output:
[('black', 'S'), ('black', 'M'), ('black', 'L'),
 ('white', 'S'), ('white', 'M'), ('white', 'L')]
'''

colors = ['black', 'white']
sizes = ['S', 'M', 'L']
cart = [(colors[i], sizes[j]) for i in range(0,2) for j in range(0,3)]
print cart

'''
Exercise: In the output above, change the 'S' size to 'Small' and 'M' and 'L' to be 'Large'.
'''

mydict = {"S": "Small", "M": "Medium", "L": "Large"}
new_cart = [(x[0], mydict[x[1]]) for x in cart]
print new_cart

'''
Exercise: What are the other types of comprehensions and how are they different from list comprehensions?
'''

# skip for now

################################################################################################

# Higher Order Functions

'''
Exercise: Read through the Blog Post on MapReduce (http://michaelnielsen.org/blog/write-your-first-mapreduce-program-in-20-minutes/) [no need to implement] and explain how it's related to the map and reduce functions we have covered.

'''

# skip for now

'''
Exercise: What do the following uses of sorted return?
# first example
fruits = ['strawberry','fig','apple','cherry','raspberry','banana']
sorted(fruits, key=len)

This will sort the fruits list in ascending order by length of element

# second example
sorted(fruits, key=lambda word: word[::-1])

This will sort the fruits list in alphabetical order by looking at each word's last letter
'''

#Exercise: Write a function to return the two words with the highest alphanumeric score of uppercase letters:
def alpha_score(upper_letters):

    return sum(map(lambda l: 1 + ord(l) - ord('A'), upper_letters))


alpha_score('ABC')  # => 6 = 1 ('A') + 2 ('B') + 3 ('C')


def two_best(words):
    words = [word for word in words]
    print sorted(words, key = alpha_score, reverse = True)[:2]

two_best(['hEllO', 'wOrLD', 'i', 'aM', 'PyThOn'])

################################################################################################

#Iterables and Generators

'''
[Expert] Exercise: For each of the following scenarios, discuss whether it would be more appropriate to use a generator expression or a list comprehension:
    Searching for a given entity in the entries of a 1TB database.
    Calculate cheap airfare using journey-to-destination flight information.
    Finding the first palindromic Fibonacci number greater than 1,000,000.
    Determine all multi-word anagrams of user-supplied 1000-character-or-more strings (very expensive to do).
    Return a list of all startups within 50 miles of San Francisco.

'''

'''
[Challenge] Exercise: In class, we dicussed how to generate primes using the following function.
'''

def make_divisibility_test(n):
    def divisible_by_n(m):
         return m % n == 0
    return divisible_by_n

make_divisibility_test(8)


def comps_under(n):
    tests = [make_divisibility_test(1), make_divisibility_test(2)]
    for i in range(3, n):
        #print i
        if map(lambda test: test(i), tests).count(True) > 1:
            print i #tests, map(lambda test: test(i), tests)
        tests.append(make_divisibility_test(i))
            #yield i

comps_under(18)

################################################################################################

# Decorators
