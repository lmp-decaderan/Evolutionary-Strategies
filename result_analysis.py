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
import csv
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
    result_from_file = []
    for i_sengtence in range(8000):
        score_tmp = -10
        line_result = []

        _dict_key = 'level_%s'%sentence_level_list[i_sengtence]

        sentence_list_tmp = []
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
                sentence_list_tmp.append(line_tmp[0])
                if(math.exp(float(line_tmp[1]))*values > score_tmp):
                
                    score_tmp = math.exp(float(line_tmp[1]))*values
                    line_result = line_tmp[0]


                    from_file_tmp = key.split('_')[0]
                    best_sentence_tmp = line_result
            else:
                sentence_list_tmp.append(line_tmp[1])
                if number_Re.match(line_tmp[2]):
                    if(math.exp(float(line_tmp[2]))*values > score_tmp):
                
                        score_tmp = math.exp(float(line_tmp[2]))*values
                        line_result = line_tmp[1]


                        from_file_tmp =  key.split('_')[0]
                        best_sentence_tmp = line_result
                #line_result = getline_from_file(data_path_now_file,i_sengtence+1).strip()
                #line_result = getline_from_file(data_path_now_file,i_sengtence+1).strip()
            #print(model_score)
        #raise
        data_line.append(line_result)
        result_from_file.append([best_sentence_tmp,from_file_tmp]+sentence_list_tmp)
    return data_line, result_from_file

def get_best_sentence_valid(weight_dict = {}, score_data_path=None,sentence_level_list = None):
    data_line = []
    result_from_file = []

    for i_sengtence in range(8000):
        score_tmp = -10000
        line_result = []
        _dict_key = 'level_%s'%sentence_level_list[i_sengtence]

        sentence_list_tmp = []
        #print(weight_dict)
        for key,values in weight_dict[_dict_key].items():
            data_path_now = score_data_path + key
            data_path_now_file = score_data_path+'outputs/'+key[:-8]+'.outputs'
            #print(data_path_now)
            line_tmp  = getline_from_file(data_path_now,i_sengtence+1)
            #line_result = getline_from_file(data_path_now_file,i_sengtence+1).strip()
            #print(line_tmp)
            line_tmp = line_tmp.strip().split("\t")
            #sentence_list_tmp.append(line_tmp)
            #print(line_tmp)
            #print("%s_bleu_score_is:%s"%(key,math.exp(float(line_tmp[1]))))
            #print(data_path_now)
            #print(i_sengtence)
            if number_Re.match(line_tmp[1]):
                sentence_list_tmp.append(line_tmp[0])
                if(math.exp(float(line_tmp[1]))*values > score_tmp):
                
                    score_tmp = math.exp(float(line_tmp[1]))*values
                    #line_result = line_tmp[0]
                    line_result = getline_from_file(data_path_now_file,i_sengtence+1).strip()

                    from_file_tmp = key.split('_')[0]
                    best_sentence_tmp = line_result
            else:
                sentence_list_tmp.append(line_tmp[1])
                #print('hahah')
                if number_Re.match(line_tmp[2]):
                    if(math.exp(float(line_tmp[2]))*values > score_tmp):
                
                        score_tmp = math.exp(float(line_tmp[2]))*values
                        #line_result = line_tmp[1]
                        line_result = getline_from_file(data_path_now_file,i_sengtence+1).strip()

                        from_file_tmp = key.split('_')[0]
                        best_sentence_tmp = line_result
                #line_result = getline_from_file(data_path_now_file,i_sengtence+1).strip()
                #line_result = getline_from_file(data_path_now_file,i_sengtence+1).strip()
            #print(model_score)
        if len(line_result) < 1:
            print(i_sengtence)
            print(score_tmp)
            raise
        data_line.append(line_result)
        result_from_file.append([best_sentence_tmp,from_file_tmp]+sentence_list_tmp)

    return data_line, result_from_file

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


def get_sentence_level_v2(datalist = None,level_th = [15,1000]):
  
    
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

def write_csv(save_path = None,context_list = None):
    with open(save_path, 'w', encoding = 'utf-8',newline='') as f:
        writer = csv.writer(f)
        for row in context_list:
            writer.writerow(row)


if __name__ == '__main__':

    
    init_weights = False
    normal_sigma = 1.0

    testA_file_dir = 'data/testA/'
    testB_file_dir = 'data/testB/'
    valid_file_dir = 'data/valid/'
    

    testA_reference_en_file_path = 'data/reference_en/testA.en'
    testB_reference_en_file_path = 'data/reference_en/testB.en'
    valid_reference_en_file_path = 'data/reference_en/valid.en'

    #weights_json_file_path = 'weigths_file/weights_valid_0.3354_testA_0.3221_11_11_23_12_01_level_16_1000.json'
    weights_json_file_path = 'weigths_file/weights__11_11_23_12_01_level_16_1000_8.json'
    

    '''sentence level split!!'''
    level_th = [16,1000]
    #level_th = [1000]


    '''reference_en load!!!'''

    testA_reference_en_data_length = []
    with codecs.open(os.path.join(os.getcwd(),testA_reference_en_file_path),'r','utf-8') as reference_file:
        for line in reference_file.readlines():
            if line.strip() is not None:
                reference_tmp = len(line.strip().split(' '))
                testA_reference_en_data_length.append(reference_tmp)
    testA_sentence_level = get_sentence_level_v2(testA_reference_en_data_length,level_th)


    testB_reference_en_data_length = []    
    with codecs.open(os.path.join(os.getcwd(),testB_reference_en_file_path),'r','utf-8') as reference_file:
        for line in reference_file.readlines():
            if line.strip() is not None:
                reference_tmp = len(line.strip().split(' '))
                testB_reference_en_data_length.append(reference_tmp)
    testB_sentence_level = get_sentence_level_v2(testB_reference_en_data_length,level_th)



    valid_reference_en_data_length = []    
    with codecs.open(os.path.join(os.getcwd(),valid_reference_en_file_path),'r','utf-8') as reference_file:
        for line in reference_file.readlines():
            if line.strip() is not None:
                reference_tmp = len(line.strip().split(' '))
                valid_reference_en_data_length.append(reference_tmp)
    valid_sentence_level = get_sentence_level_v2(valid_reference_en_data_length,level_th)


    with open(os.path.join(os.getcwd(),weights_json_file_path),"r") as json_file:
        data_model_score = json.load(json_file)

    #print(data_model_score)
    if(len(data_model_score) != len(level_th)):
        print('weight_level is not equal sentence_level!!')
        raise

    
    
    '''超参数'''
    
    weights_dict_best = data_model_score


    file_list_dict = []
    for key,values in weights_dict_best['level_0'].items():
        file_list_dict.append(key)
    
    file_list_dict = [['rerank_decode','ID']+file_list_dict]

    #print(file_list_dict)

    testA_data_line_best = []
    testB_data_line_best = []
    valid_data_line_best = []

    testA_data_from = []
    testB_data_from = []
    valid_data_from = []


    valid_data_line_best,valid_data_from = get_best_sentence_valid(data_model_score, valid_file_dir,valid_sentence_level) 
    testA_data_line_best,testA_data_from = get_best_sentence_testA(data_model_score, testA_file_dir,testA_sentence_level)
    testB_data_line_best,testB_data_from = get_best_sentence_testA(data_model_score, testB_file_dir,testB_sentence_level)
    #print(testB_data_from[0])
    

    
    print("-------------------result--------------------")

    timestr = time.strftime('%m_%d_%H_%M_%S',time.localtime(time.time()))
    print(timestr)


    with codecs.open(os.path.join(os.getcwd(),'result/testA_result_normal_'+ timestr + '_level_%s'%level_th[0] + '.decodes'),'w','utf-8') as output_data_file:
        for line in testA_data_line_best:
            #print(line)
            if line.strip() is not None:
               
                output_data_file.write( line+'\n')
    write_csv(os.path.join(os.getcwd(),'result/testA_result_normal_'+ timestr + '_level_%s'%level_th[0] + '.csv'),file_list_dict + testA_data_from)

    with codecs.open(os.path.join(os.getcwd(),'result/testB_result_normal_'+ timestr + '_level_%s'%level_th[0] + '.decodes'),'w','utf-8') as output_data_file:
        for line in testB_data_line_best:
            #print(line)
            if line.strip() is not None:
                output_data_file.write(line+'\n')
    write_csv(os.path.join(os.getcwd(),'result/testB_result_normal_'+ timestr + '_level_%s'%level_th[0] + '.csv'),file_list_dict+testB_data_from)

    with codecs.open(os.path.join(os.getcwd(),'result/valid_result_normal_'+ timestr + '_level_%s'%level_th[0] + '.decodes'),'w','utf-8') as output_data_file:
        for line in valid_data_line_best:
            if line.strip() is not None:
                output_data_file.write(line+'\n')
    write_csv(os.path.join(os.getcwd(),'result/valid_result_normal_'+ timestr + '_level_%s'%level_th[0] + '.csv'),file_list_dict+valid_data_from)





    

