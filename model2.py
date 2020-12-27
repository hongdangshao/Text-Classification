import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

def get_train_data(fu_txt_path,zheng_txt_path):
    fu_txt = pd.read_table(fu_txt_path, encoding="gbk")
    fu_txt.columns = ["data"]
    fu_txt["target"] = 0
    zheng_txt = pd.read_table(zheng_txt_path, encoding="gbk")
    zheng_txt.columns = ["data"]
    zheng_txt["target"] = 1
    train_data = fu_txt.append(zheng_txt).reset_index(drop=True)
    return train_data

# 结巴分词
def get_cut_train_data(train_data,process_feature):
    tmp_feature = train_data[process_feature]
    print(type(process_feature))
    cut_list = []
    for i in tmp_feature:
        # print(i)
        i  = list(jieba.cut(i))
        cut_list.append(i)
    train_data[process_feature] = cut_list
    print("结巴分词后的数据",train_data)
    return train_data

# 去除停用词
def remove_stop_words(train_data,process_feature,stopwords_list):
    tmp_feature = train_data[process_feature]
    remove_list = []
    for i in tmp_feature:
        index = len(i) - 1
        while index>0:
            if i[index] in stopwords_list:
                del i[index]
                index -=1
            else:
                index -=1
        remove_list.append(i)
    train_data[process_feature]=remove_list
    print("移出停用词后的数据",train_data)
    return train_data

# 构造空格型字符串，将[觉得, 不是, 特别, 注重, 成绩, 那种]去掉中括号和逗号，逗号替换成空格
def change_list_to_str(train_data,process_feature):
    tmp_feature = train_data[process_feature]
    tmp_feature = tmp_feature.map(lambda x: " ".join(x))
    train_data[process_feature] = tmp_feature
    print("空格型数据",train_data)
    return train_data

def predict_class(train_data,feature,label,test_size,random_state,predict_method):
    try:
        x = train_data[feature]
        y = train_data[label]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
        # print(x_train,x_test,y_train,y_test)
        # tfidf计算词向量和权重
        tfidf = TfidfVectorizer()
        x_train = tfidf.fit_transform(x_train)
        x_test = tfidf.transform(x_test)
        print("训练样本集",x_train.toarray())
        print("测试集",x_test.toarray())
        if predict_method == 0:
            rfc = MultinomialNB()
        elif predict_method == 1:
            rfc = GridSearchCV(svm.SVC(kernel="linear"), param_grid={"C": [0.1, 1, 10], "gamma": [1, 0.1, 0.01]}, cv=4)
        elif predict_method == 2:
            rfc = DecisionTreeClassifier(criterion="gini", max_depth=5)
        elif predict_method == 3:
            rfc = RandomForestClassifier(n_estimators=10, max_depth=4, criterion="entropy")
        rfc.fit(x_train, y_train)
        print("模型",rfc )
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
    # 转换成训练数据
    fu_txt_path = "2000_neg.txt"
    zheng_txt_path = "2000_pos.txt"
    train_data = get_train_data(fu_txt_path,zheng_txt_path)
    print(train_data)

    # 结巴分词
    process_feature = "data"
    train_data = get_cut_train_data(train_data,process_feature)

    # 去除停用词（是，的，空格等等无实际意义的词，减少数据维度）
    stopwords_list = pd.read_table("stopwords.txt")["stop_words"].tolist()
    train_data = remove_stop_words(train_data,process_feature,stopwords_list)

    # 空格型字符数据
    train_data = change_list_to_str(train_data,process_feature)

    # 算法 = 数据切分成训练集和测试集 + TFIDF转换成向量 + 模型传入数据
    feature="data"
    label = "target"
    test_size=0.3
    random_state=1
    predict_method=1
    predict_class(train_data,feature,label,test_size,random_state,predict_method)