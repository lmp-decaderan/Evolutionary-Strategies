# -*- coding: utf-8 -*-
# Time        : 18-11-09
# Author      : liaominpeng
# Version     : 
# Description : Merge the results and process the data
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import os
import linecache
import math
import re
import json
import time
import argparse

number_Re = re.compile(r'^[-+]?[0-9]+\.[0-9]+$')

def getline_from_file(data_path = None,index = 0):
    
    if data_path is not None:
        path = os.path.join(os.getcwd(),data_path)
        result = linecache.getline(path, index)
    return result

def get_best_sentence(weight_dict = {}, score_data_path=None,sentence_level_list = None,data_set_length = 8000):
    data_line = []
    for i_sentence in range(data_set_length):
        score_tmp = -10
        line_result = []
        _dict_key = 'level_%s'%sentence_level_list[i_sentence]
       
        for key,values in weight_dict[_dict_key].items():
            data_path_now = score_data_path + key
          
            line_tmp  = getline_from_file(data_path_now,i_sentence+1)
            
            line_tmp = line_tmp.strip().split("\t")
           
            if number_Re.match(line_tmp[1]):
                if(math.exp(float(line_tmp[1]))*values > score_tmp):
                
                    score_tmp = math.exp(float(line_tmp[1]))*values
                   
                    line_result = replace_punctuation(line_tmp[0])
            else:
               
                if number_Re.match(line_tmp[2]):
                    if(math.exp(float(line_tmp[2]))*values > score_tmp):
                
                        score_tmp = math.exp(float(line_tmp[2]))*values
                    
                        line_result = replace_punctuation(line_tmp[0])
               
        if len(line_result) < 1:
            print(i_sentence)
            print(score_tmp)
            raise
        data_line.append(line_result)
    return data_line

def get_level(data,level_th=None):
    for index,value in enumerate(level_th):
        if data < value:
            return index

def get_sentence_level(datalist = None,level_th = [16,1000]):
    result = [] 
    for key in datalist:
        result.append(get_level(key,level_th))
    if len(datalist) == len(result):
        return result
    else:
        raise

def get_sentence_level_of_dataset(reference_file,level_th = [16,1000]):
    reference_data_length = []
    with codecs.open(os.path.join(os.getcwd(),reference_file),'r','utf-8') as reference_file:
        for line in reference_file.readlines():
            if line.strip() is not None:
                reference_tmp = len(line.strip().split(' '))
                reference_data_length.append(reference_tmp)
    return get_sentence_level(reference_data_length,level_th)

'''换标点符号'''
def process_q(line,q_count):
    if (q_count == 1):
        line = line.replace('"', '”')
    elif (q_count > 1):
        q_count = 1
        for w in line:
            if (w == '"' and q_count % 2 != 0):
                line = line.replace('"', '“')
                q_count += 1
            elif (w == '"' and q_count % 2 == 0):
                line = line.replace('"', '”')
                q_count += 1
    return line

def process_m(line, m_count):
    if (m_count == 1):
        line = line
    elif (m_count == 2):
        line = line.replace('--','-')
    elif (m_count > 2):
        line = line.replace('-','')
    return line

def replace_punctuation(sentence_line):
    '''换中文的标点符号''' 
    line = sentence_line
    q_count = 0 # 引号检测
    m_count = 0 # -号检测
    if line:
        for w in line:
            if (w == '"'):
                q_count += 1
            if (w == '-'):
                m_count += 1

        line = process_q(line,q_count)
        line = process_m(line,m_count)

        res = line.replace(',','，')
        res = res.replace('!','！')
        res = res.replace('?','？')
        res = res.replace('...','…')
        res = res.replace(';','；')
        return res
    return None

def write_sgm(file_path,context_line):

    with codecs.open(file_path,'w','utf-8') as output_data_file:
        output_data_file.write("<?xml version='1.0' encoding='utf-8'?>\n")
        output_data_file.write("<mteval>\n")
        output_data_file.write('<tstset trglang="zh" setid="setid" srclang="en" trglang="zh">\n')
        output_data_file.write('<DOC sysid="DemoSystem" docid="docid" genre="talk" sysid="sysid">\n')
        

        for index,line in enumerate(context_line):
            if line is None:
                raise
            outline='<seg id="'+str(index+1)+'">'+line+'</seg>'+'\n'
            output_data_file.write(outline)
            
        output_data_file.write("</doc>\n")
        output_data_file.write("</tstset>\n")
        output_data_file.write("</mteval>\n")

def get_result(args):

    file_from_dir = args.file_from_dir
    result_to_file = args.result_to_file
    reference_file = args.reference_file
    weights_json_file = args.weights_json_file
    level_th = args.level_th
    dataset_length = args.dataset_length


    sentence_level = get_sentence_level_of_dataset(reference_file,level_th)

    
    with open(os.path.join(os.getcwd(),weights_json_file),"r") as json_file:
        data_model_score = json.load(json_file)


    if(len(data_model_score) != len(level_th)):
        print('weight_level is not equal sentence_level!!')
        raise
 

    data_line_best = get_best_sentence(data_model_score, file_from_dir,sentence_level,dataset_length) 
   
    print("---Merging result is completed!---")

    timestr = time.strftime('%m_%d_%H_%M_%S',time.localtime(time.time()))
    print('Time:' + timestr)

    level_name = '_level'
    for key in level_th:
        level_name += '_' + str(key)

    write_sgm(os.path.join(os.getcwd(), result_to_file +'_' + timestr + level_name + '.sgm'),data_line_best)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='get_result')
    parser.add_argument('-f','--file_from_dir', type=str, default='data/test/',help='The file path of Reranking')
    parser.add_argument('-r','--result_to_file', type=str, default='result/test_result',help='The file name of result')
    parser.add_argument('-rf','--reference_file', type=str, default='data/reference_en/test.en',help='The reference path')
    parser.add_argument('-wjson','--weights_json_file', type=str, default='weights_file/weights_11_11_23_12_01_level_16_1000.json',help='The weights file path,json')
    parser.add_argument('-l','--level_th', nargs='+', type=int, dest='level_th', default=[16,1000],help='The thresholds required for sentence classification')
    parser.add_argument('-dl','--dataset_length', type=int, dest='dataset_length', default=8000,help='The test data length')
    args = parser.parse_args()

    get_result(args)