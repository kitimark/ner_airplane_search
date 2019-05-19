from anago import Sequence
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from util import load_data

# x_train, y_train = load_data_and_labels('train.txt')
# print(x_train)

# x_train = [['ไป','เชียงใหม่','จาก','กรุงเทพ'],
#            ['จาก','กรุงเทพ','ไป','เชียงใหม่'],
#            ['ไป','กรุงเทพ','จาก','เชียงใหม่'],
#            ['เชียงใหม่','-','กรุงเทพ'],
#            ['กรุงเทพ','-','เชียงใหม่']]
# y_train = [['O','B-DEST','O','B-START'],
#            ['O','B-START','O','B-DEST'],
#            ['O','B-DEST','O','B-START'],
#            ['B-START','O','B-DEST'],
#            ['B-START','O','B-DEST']]

x_train, y_train = [], []

for i in range(6):
    x, y = load_data('NLp_Project/laura'+str(i)+'.txt')
    x_train = x_train + x
    y_train = y_train + y
    if (i < 3) :
        for j in range(4):
            x, y = load_data('NLp_Project/laura'+str(i)+'w_p'+str(j)+'.txt')
            x_train = x_train + x
            y_train = y_train + y
        for j in range(6):
            x, y = load_data('NLp_Project/laura'+str(i)+'w_t'+str(j)+'.txt')
            x_train = x_train + x
            y_train = y_train + y

for i in range(2):
    x, y = load_data('NLp_Project/some_word'+str(i)+'.txt')
    x_train = x_train + x
    y_train = y_train + y

x_train, y_train = shuffle(x_train, y_train)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=2)

# with open('train.txt', 'w', encoding='utf-8-sig') as f:
#     for x_s, y_s in zip(x_train, y_train):
#         for x, y in zip(x_s, y_s):
#             f.writelines(x + ' ' + y + '\n')
        
#         f.writelines('\n')


# x_test = [['จาก','กรุงเทพ','ไป','เชียงใหม่']]
# y_test = [['O','B-START','O','B-DEST']]

model = Sequence(use_char=False, use_crf=False)
model.fit(x_train, y_train, x_test, y_test, epochs=10)
history = model.score(x_test, y_test)
print(history)
text = 'ไป เชียงใหม่ จาก กรุงเทพ'
result = model.analyze(text)
print(result)
result = model.analyze('หา ตั๋ว ไป ที่ เชียงใหม่')
print(result)
