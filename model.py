import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def load_message(txt_path):
    content = []
    label = []
    with open(txt_path, "r", encoding="utf-8") as f:
        con = f.readlines()
        num = len(con)
        for i in range(num):
            message = con[i].encode('utf-8').decode('utf-8-sig').strip().split('\t')
            label.append(message[0])
            content.append(message[1])
    return num, label, content

# Dataframe数据
def get_train_data(label, content):
    data = {"title": content, "class": label}
    train_data = pd.DataFrame(data)
    return train_data

# 结巴分词
def get_cut_train_data(train_data, process_feature):
    tmp_feature = train_data[process_feature]
    cut_list = []
    for i in tmp_feature:
        # print(i)
        i = list(jieba.cut(i))
        cut_list.append(i)
    train_data[process_feature] = cut_list
    print("结巴分词后的数据", train_data)
    return train_data


# 去除停用词
def remove_stop_words(train_data, process_feature, stopwords_list):
    tmp_feature = train_data[process_feature]
    remove_list = []
    for i in tmp_feature:   #i是一个jieba分词后的list
        index = len(i) - 1
        while index > 0:
            if i[index] in stopwords_list:
                del i[index]
                index -= 1
            else:
                index -= 1
        remove_list.append(i)
    train_data[process_feature] = remove_list
    print("移出停用词后的数据", train_data)
    return train_data


# 构造空格型字符串，将[觉得, 不是, 特别, 注重, 成绩, 那种]去掉中括号和逗号，逗号替换成空格
def change_list_to_str(train_data, process_feature):
    tmp_feature = train_data[process_feature]
    tmp_feature = tmp_feature.map(lambda x: " ".join(x))
    train_data[process_feature] = tmp_feature
    print("空格型数据", train_data)
    return train_data


def predict_class(train_data, feature, label, test_size, random_state, predict_method):
    try:
        x = train_data[feature]
        y = train_data[label]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

        # tfidf计算词向量和权重
        tfidf = TfidfVectorizer()
        #必须先用fit_transform(trainData)，之后再transform(testData)
        x_train = tfidf.fit_transform(x_train)
        x_test = tfidf.transform(x_test)
        print("训练样本集", x_train)
        print("测试集", x_test)
        if predict_method == 0:
            rfc = MultinomialNB()
        elif predict_method == 1:
            rfc = GridSearchCV(svm.SVC(kernel="linear"), param_grid={"C": [0.1, 1, 10], "gamma": [1, 0.1, 0.01]}, cv=4)
        elif predict_method == 2:
            rfc = DecisionTreeClassifier(criterion="gini", max_depth=5)
        elif predict_method == 3:
            rfc = RandomForestClassifier(n_estimators=10, max_depth=4, criterion="entropy")
        rfc.fit(x_train, y_train)
        print("模型", rfc)
        y_predict = rfc.predict(x_test).tolist()
        rescore = rfc.score(x_test, y_test)
        print(rescore)
        print("测试目标值", y_test.tolist())
        print("预测目标值", y_predict)
        report = classification_report(y_true=y_test, y_pred=y_predict)
        print(report)
        s = confusion_matrix(y_true=y_test, y_pred=y_predict)
        print(s)
    except Exception as e:
        print(e)


if __name__ == '__main__':
    # 垃圾短信数据集，用来验证模型效果
    txt_path = "message.txt"
    num, label, content = load_message(txt_path)

    # 转换成训练数据
    train_data = get_train_data(label, content)
    train_data = train_data
    print("DataFrame数据", train_data)

    # 结巴分词
    process_feature = "title"
    train_data = get_cut_train_data(train_data, process_feature)

    # 去除停用词（是，的，空格等等无实际意义的词，减少数据维度）
    stopwords_list = pd.read_table("stopwords.txt")["stop_words"].tolist()
    train_data = remove_stop_words(train_data, process_feature, stopwords_list)

    # 空格型字符数据
    train_data = change_list_to_str(train_data, process_feature)

    # 算法 = 数据切分成训练集和测试集 + TFIDF转换成向量 + 模型传入数据
    feature = "title"
    label = "class"
    test_size = 0.3
    random_state = 1
    predict_method = 1
    predict_class(train_data, feature, label, test_size, random_state, predict_method)
