import codecs
import os
import random
import linecache
import nltk
import math
import re
import numpy as np
import json
import time
#the_line = linecache.getline('d:/FreakOut.cpp', 222)
number_Re = re.compile(r'^[-+]?[0-9]+\.[0-9]+$')

def getline_from_score(data_path = None,begin = 0,end = 0):
    result = []
    if data_path is not None:
        path = os.path.join(os.getcwd(),data_path)
        for i in range(begin,end+1):
            line_con = linecache.getline(path, i)
            line_con = line_con.split('\t')
            result.append([
                int(line_con[0]),
                float(line_con[int(line_con[0])+1]),
                i
                ])
    return result

def getline_from_file(data_path = None,index = 0):
    
    if data_path is not None:
        path = os.path.join(os.getcwd(),data_path)
        result = linecache.getline(path, index)
    return result
def compute_bleu(reference,hypothesis,weights = [0.25,0.25,0.25,0.25]):
    if len(reference) == len(hypothesis):
        avg_bleu = 0
        for i in range(len(reference)):
            #print(reference[i])
            #print(hypothesis[i])
            #raise
            avg_bleu += nltk.translate.bleu_score.sentence_bleu([reference[i]], hypothesis[i], weights = weights) 
        return avg_bleu/len(reference)
    return None

def compute_bleu_5k(reference,hypothesis,reference_list,weights = [0.25,0.25,0.25,0.25]):
    avg_bleu = 0
    for i in range(len(reference)):
        avg_bleu += nltk.translate.bleu_score.sentence_bleu([reference[i]], hypothesis[reference_list[i]], weights = weights) 
    return avg_bleu/len(reference)
    #return None


def get_best_sentence_testA(weight_dict = {}, score_data_path=None,sentence_level_list=None):
    data_line = []
    for i_sengtence in range(8000):
        score_tmp = -10
        line_result = []
        _dict_key = 'level_%s'%sentence_level_list[i_sengtence]
        #print(weight_dict)
        for key,values in weight_dict[_dict_key].items():
            data_path_now = score_data_path + key
            #data_path_now_file = score_data_path+'test/'+key+'.outputs'
            #print(data_path_now)
            line_tmp  = getline_from_file(data_path_now,i_sengtence+1)
            #print(line_tmp)
            line_tmp = line_tmp.strip().split("\t")
            #print(line_tmp)
            #print("%s_bleu_score_is:%s"%(key,math.exp(float(line_tmp[1]))))
            #print(data_path_now)
            #print(i_sengtence)
            if number_Re.match(line_tmp[1]):
                if(math.exp(float(line_tmp[1]))*values > score_tmp):
                
                    score_tmp = math.exp(float(line_tmp[1]))*values
                    line_result = line_tmp[0]
            else:
                if number_Re.match(line_tmp[2]):
                    if(math.exp(float(line_tmp[2]))*values > score_tmp):
                
                        score_tmp = math.exp(float(line_tmp[2]))*values
                        line_result = line_tmp[1]
                #line_result = getline_from_file(data_path_now_file,i_sengtence+1).strip()
                #line_result = getline_from_file(data_path_now_file,i_sengtence+1).strip()
            #print(model_score)
        #raise
        data_line.append(line_result)
    return data_line

def get_best_sentence_valid(weight_dict = {}, score_data_path=None,sentence_level_list = None):
    data_line = []
    for i_sengtence in range(8000):
        score_tmp = -10000
        line_result = []
        _dict_key = 'level_%s'%sentence_level_list[i_sengtence]
        #print(weight_dict)
        for key,values in weight_dict[_dict_key].items():
            data_path_now = score_data_path + key
            data_path_now_file = score_data_path+'outputs/'+key[:-8]+'.outputs'
            #print(data_path_now)
            line_tmp  = getline_from_file(data_path_now,i_sengtence+1)
            #line_result = getline_from_file(data_path_now_file,i_sengtence+1).strip()
            #print(line_tmp)
            line_tmp = line_tmp.strip().split("\t")
            #print(line_tmp)
            #print("%s_bleu_score_is:%s"%(key,math.exp(float(line_tmp[1]))))
            #print(data_path_now)
            #print(i_sengtence)
            if number_Re.match(line_tmp[1]):
                if(math.exp(float(line_tmp[1]))*values > score_tmp):
                
                    score_tmp = math.exp(float(line_tmp[1]))*values
                    #line_result = line_tmp[0]
                    line_result = getline_from_file(data_path_now_file,i_sengtence+1).strip()
            else:
                #print('hahah')
                if number_Re.match(line_tmp[2]):
                    if(math.exp(float(line_tmp[2]))*values > score_tmp):
                
                        score_tmp = math.exp(float(line_tmp[2]))*values
                        #line_result = line_tmp[1]
                        line_result = getline_from_file(data_path_now_file,i_sengtence+1).strip()
                #line_result = getline_from_file(data_path_now_file,i_sengtence+1).strip()
                #line_result = getline_from_file(data_path_now_file,i_sengtence+1).strip()
            #print(model_score)
        if len(line_result) < 1:
            print(i_sengtence)
            print(score_tmp)
            raise
        data_line.append(line_result)
    return data_line

def get_level(data,level_th=None):
    for index,value in enumerate(level_th):
        if data < value:
            return index

def get_sentence_level(datalist = None):
    level_th = [20,30,40]
    result = [] 
    for key in datalist:
        if key < level_th[0]:
            result.append(0)
        elif key < level_th[1]:
            result.append(1) 
        elif key < level_th[2]:
            result.append(2) 
        else:
            result.append(3)
    if len(datalist) == len(result):
        return result
    else:
        raise


def get_sentence_level_v2(datalist = None,level_th = [16,1000]):
    
    #level_th = [15,1000] 38.73
    #level_th = [20,1000] 38.64
    #level_th = [12,1000] 38.38
    #level_th = [13,1000] 38.41
    #level_th = [17,1000] 38.53
    #level_th = [14,1000] 38.56
    #level_th = [10,1000]

    #level_th = [10,15,1000] 1:38.52
    #level_th = [10,16,1000] 2:38.47
    #level_th = [12,15,1000]  1:
    #level_th = [12,16,1000]  
    #level_th = [12,20,1000] 1:
    #level_th = [10,18,1000] 1:
    #level_th = [12,18,1000] 
    
    result = [] 
    for key in datalist:
        result.append(get_level(key,level_th))
    if len(datalist) == len(result):
        return result
    else:
        raise


def get_decode_file_name(dir_path = None):
    file_result = []
    #print(dir_path)
    file_list = os.listdir(dir_path)
    for file in file_list:
        file_path = os.path.join(dir_path, file)  
        if os.path.isfile(file_path) and file_path.endswith('.decodes'):  
            file_result.append(file)
    return file_result


if __name__ == '__main__':

    
    init_weights = False
    normal_sigma = 1.0

    testA_file_dir = 'data/testA/'
    valid_file_dir = 'data/valid/'
    
    testA_reference_file_path = 'data/reference/testA_5k.reference'
    valid_reference_file_path = 'data/reference/valid.reference'

    testA_reference_en_file_path = 'data/reference_en/testA.en'
    valid_reference_en_file_path = 'data/reference_en/valid.en'

    level_th = [15,1000] 


    '''reference load!!!'''
    testA_reference_data = []
    testA_reference_data_list = []
    with codecs.open(os.path.join(os.getcwd(),testA_reference_file_path),'r','utf-8') as reference_file:
        for line in reference_file.readlines():
            if line.strip() is not None:
                reference_tmp = line.strip().split('\t')
                testA_reference_data_list.append(int(reference_tmp[0]))
                testA_reference_data.append(reference_tmp[1].split(' '))

    valid_reference_data = []
    #valid_reference_data_list = []
    with codecs.open(os.path.join(os.getcwd(),valid_reference_file_path),'r','utf-8') as reference_file:
        for line in reference_file.readlines():
            if line.strip() is not None:
                reference_tmp = line.strip().split(' ')
                valid_reference_data.append(reference_tmp)


    '''reference_en load!!!'''
    testA_reference_en_data_length = []
    
    with codecs.open(os.path.join(os.getcwd(),testA_reference_en_file_path),'r','utf-8') as reference_file:
        for line in reference_file.readlines():
            if line.strip() is not None:
                reference_tmp = len(line.strip().split(' '))
                testA_reference_en_data_length.append(reference_tmp)
    
    testA_sentence_level = get_sentence_level_v2(testA_reference_en_data_length,level_th)

    valid_reference_en_data_length = []
    
    with codecs.open(os.path.join(os.getcwd(),valid_reference_en_file_path),'r','utf-8') as reference_file:
        for line in reference_file.readlines():
            if line.strip() is not None:
                reference_tmp = len(line.strip().split(' '))
                valid_reference_en_data_length.append(reference_tmp)
    
    valid_sentence_level = get_sentence_level_v2(valid_reference_en_data_length,level_th)


    #print(reference_data_list)
    '''weights file name list'''
    file_list = get_decode_file_name(testA_file_dir)


    data_model_score = {}
    sentence_level_class = len(set(valid_sentence_level))
    for key in range(sentence_level_class):
        dict_key = 'level_%s'%key
        data_model_score_tmp = {}
        for file_name in file_list:
            data_model_score_tmp[file_name] = 10.0
        data_model_score[dict_key] = data_model_score_tmp
    #print(data_model_score)
    
    
    '''超参数'''
    weights_search_count = 500
    
    weights_dict_best = data_model_score

    testA_data_line_best = []
    valid_data_line_best = []

    testA_best_bleu_score = 0
    valid_best_bleu_score = 0

    count = 0
    
    
    for weight_bias in range(weights_search_count):
        data_model_score_tmp = {}
        #sum_weights_avg = 0
        for key,value in data_model_score.items():
            weights_dict_inner = {}
            for key_inner,value_inner in value.items():
                weights_dict_inner[key_inner] = np.random.normal(loc=weights_dict_best[key][key_inner], scale=normal_sigma)
            data_model_score_tmp[key] = weights_dict_inner
            
        #print(data_model_score_tmp)
        valid_data_line = get_best_sentence_valid(data_model_score_tmp, valid_file_dir,valid_sentence_level)  
         
        valid_hypothesis = []
        for line_tmp in valid_data_line:
            valid_hypothesis.append([key for key in line_tmp])

        
        valid_bleu_tmp = compute_bleu(valid_reference_data,valid_hypothesis)
        #print(data_line)
        print("%s_valid_bleu_score_is:%s"%(count,valid_bleu_tmp))
        count += 1

        if valid_bleu_tmp > valid_best_bleu_score:


            testA_data_line = get_best_sentence_testA(data_model_score_tmp, testA_file_dir,testA_sentence_level)
            testA_hypothesis = []
            for line_tmp in testA_data_line:
                testA_hypothesis.append([key for key in line_tmp])

            testA_bleu_tmp = compute_bleu_5k(testA_reference_data,testA_hypothesis,testA_reference_data_list)

                
                
            valid_best_bleu_score = valid_bleu_tmp
            testA_best_bleu_score = testA_bleu_tmp

            testA_data_line_best = testA_data_line
            valid_data_line_best = valid_data_line

            weights_dict_best = data_model_score_tmp

            print("valid_best_bleu_score_is:%s,testA_best_bleu_score_is:%s"%(valid_best_bleu_score,testA_best_bleu_score))
                
    

    #data_line_best = []
    print("-------------------result--------------------")
    print("valid_best_bleu_score_is:%s,testA_best_bleu_score_is:%s"%(valid_best_bleu_score,testA_best_bleu_score))
    #print("update_best_bleu_score_is_dict:%s"%weights_dict_best)

    timestr = time.strftime('%m_%d_%H_%M_%S',time.localtime(time.time()))
    print(timestr)

    level_file_name = '_level'
    for key in level_th:
        level_file_name += '_%s'%key


    json_file_name = 'result/weights_json/weights_valid_%.4f_testA_%.4f_'%(valid_best_bleu_score,testA_best_bleu_score)
    json_file_name += timestr + level_file_name + '_epoch_%s'%weights_search_count + 'V0.1' +'.json'
    with open(json_file_name,"w") as json_file:
        json.dump(weights_dict_best,json_file)


    with codecs.open(os.path.join(os.getcwd(),'result/testA_result_normal_'+ timestr +'.decodes'),'w','utf-8') as output_data_file:
        for line in testA_data_line_best:
            #print(line)
            output_data_file.write(line+'\n')

    with codecs.open(os.path.join(os.getcwd(),'result/valid_result_normal_'+ timestr +'.decodes'),'w','utf-8') as output_data_file:
        for line in valid_data_line_best:
            #print(line)
            output_data_file.write(line+'\n')


    

