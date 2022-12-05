import random

# generate a list of 1s and 2s to determine which one will be the dual task visit for the subjects
dual_task_list = [1]*8 + [2]*8
random.seed(0)
random.shuffle(dual_task_list)
# print(dual_task_list)

i = 0
while i < 38:
    # print(random.randint(3000, 9000))
    i += 1


i = 0
num = 7599
while i < 40:
    num -= 7
    i +=1
    print(num)
