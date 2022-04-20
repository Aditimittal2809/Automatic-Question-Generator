#!/usr/bin/env python
# coding: utf-8

# In[22]:


#Implementation for the publication https://ieeexplore.ieee.org/document/7732102/
# A few enhancements have been done to extract answers and generalize the rules based on syntax.
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk import word_tokenize,sent_tokenize

import streamlit as st

from stanfordcorenlp import StanfordCoreNLP
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk import Tree
from nltk import pos_tag
from nltk.chunk import RegexpParser
from nltk import ne_chunk
import itertools
import collections
import logging
#requires Java 1.8 or above
#Start  a stanforrrdd CoreNLP server - used stanford-corenlp-full-2018-02-27 for development
#java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer   -port 9000 -timeout 150000
#import logging

#nlp = StanfordCoreNLP('http://localhost', port=9000,logging_level=logging.DEBUG)


# In[2]:


#Starting New Paper Impl - A Rule based Question Generation Framework to deal with Simple and
#Complex Sentences
#sentence = "Barack Obama is the president of The United States of America."
#sentence = "IIIT Hyderabad is the venue of IASNLP-2018."
#sentence = "The boy went by bus."
#sentence =  sentence.rstrip().rstrip(".")
#sentence = "The contractor will build you a house for $100,000 dollars."
#sentence = "The book might cost me $10."
#sentence="The book might cost me $10 from the store."
#sentence = "$100,000 builds a house out of sticks."
#sentence = "The bill will cost them 500 million dollars in India."
#sentence = "His name is Robinson."
#sentence = "She will quickly pour the sticky liquid into the green flowery pot."
#sentence = "I am going quickly back on Saturday."
#sentence = "He wants to become a good doctor."
#sentence = "I want to work."
#sentence = "He hurriedly left the class in the morning."
#sentence = "He is addicted to smoking."
#sentence = "He will go by bus."
#sentence = "John gave Mary a book." #design more rules to catch the essence
#sentence = "He gave him a book."
#sentence = "He will buy a book."
#sentence = "He gave him a book."
#sentence = "John gave Mary a book."

#print(segments)
#dep = nlp.dependency_parse(sentence)
#print(dep)
#tree.draw()

#print(ner)
#pos = tree.treeposition_spanning_leaves(0,9)

def get_lca_length(location1, location2):
    i = 0
    while i < len(location1) and i < len(location2) and location1[i] == location2[i]:
        i+=1
    return i

def findLCA(ptree, text1, text2):
    leaf_values = ptree.leaves()
    leaf_index1 = leaf_values.index(text1)
    leaf_index2 = leaf_values.index(text2)

    location1 = ptree.leaf_treeposition(leaf_index1)
    location2 = ptree.leaf_treeposition(leaf_index2)

    #find length of least common ancestor (lca)
    lca_len = get_lca_length(location1, location2)
    return ptree[location1[:lca_len]]
#for ptree in parse_trees :
    #ptree.draw()


# In[3]:


def parse_chunks(tagged_segment , grammar):
    #print(tagged_segment)
    grammar = r"CHUNK: " + grammar
    #print(grammar)
    cp = RegexpParser(grammar)
    tree = cp.parse(tagged_segment)
    return tree
def find_chunk(chunks):
    if not isinstance(chunks, nltk.tree.Tree):
         return [ [subtree for subtree in chunk.subtrees(filter = lambda t: t.label() in ['CHUNK']) ] for chunk in chunks]   
    else :
        return [subtree for subtree in chunks.subtrees(filter = lambda t: t.label() in ['CHUNK'])]
def is_clause(segment_chunks_tree):
    if  not isinstance(chunks, nltk.Tree):
         return [ not not chunk for chunk in find_chunk(chunks) ]   
    else :
        return not not find_chunk(chunks)


# In[4]:


#handling Special case :If a segment contains only verb phrase,
#the previous segments are also checked for the existence
#of any subject phrase related to the verb phrase.

def is_only_VP(parse_trees):
    return [tree.label() == 'VP' for tree in parse_trees]


def find_the_closest_NP_for_VP(parse_trees):
    is_VP = is_only_VP(parse_trees)
    closest_NP = []
    for index,truth_val in enumerate(is_VP):
        if not truth_val:
            closest_NP.append(None)
        else:
            found_NP = False
            for index_tree in reversed(range(0,index)):
                for child in parse_trees[index_tree] :
                    if child.label() == 'NP':
                        closest_NP.append(child)
                        found_NP = True
                        break
                if found_NP:
                    break
            if not found_NP :
                closest_NP.append(None)
    return closest_NP
def enrich_VPs(parse_trees):
    enrichment_data = find_the_closest_NP_for_VP(parse_trees)
    enriched_parse_trees =  []
    enrichment_done =[]
    for ptree,enrich in zip(parse_trees,enrichment_data):
        if enrich :
            enriched_parse_trees.append(Tree('S', [enrich.copy(deep=True),ptree]))
        else:
            enriched_parse_trees.append(ptree)
    return enriched_parse_trees , [ not not data for data in enrichment_data ]



def find_VP_tree(parse_tree):
    for child  in parse_tree :
        #print (child.label())
        if child.label() == "VP":
            return child
def find_NP_tree(parse_tree):
    for child  in parse_tree :
        #print (child.label())
        if child.label() == "NP":
            return child

def chunk_VP_NP_parts(parse_tree,chunk_tree) :
    NP = find_NP_tree(parse_tree).leaves()
    VP = find_VP_tree(parse_tree).leaves()
    #print(NP,VP)
    NP_POS = []
    VP_POS = []
    #print(chunk_tree)
    chunk_pos = chunk_tree.pos()
    #print(chunk_pos)
    #print(chunk_pos)
    for pos in chunk_pos :
        if pos[0][0] in NP :
            NP.remove(pos[0][0])
            NP_POS.append(pos[0])
        elif pos[0][0] in VP:
            VP.remove(pos[0][0])
            VP_POS.append(pos[0])
    #print(NP_POS,VP_POS)
    return NP_POS,VP_POS

def verb_phrase_identification(parse_trees,is_clause,chunks):
    verb_phrase = [] 
    for tree,chunk,is_clause in zip(parse_trees,chunks,is_clause) :
        if is_clause :
            #print(tree)
            #print(type(chunk))
            chunk_tree = find_chunk(chunk)[0]
            #print(chunk_tree)
            #print(tree)
            NP_POS,VP_POS = chunk_VP_NP_parts(tree,chunk_tree)
            #print(VP_POS)       
            if len(VP_POS) > 1 :
                verb_phrase.append(VP_POS[0][0])
            else :
                vp_tag = VP_POS[0][1]
                if vp_tag == "VBD" :
                    verb_phrase.append("did")
                elif vp_tag == "VBP" or vp_tag == "VB" :
                    verb_phrase.append("do")
                elif vp_tag == "VBZ" :
                    verb_phrase.append("does")
                else :
                    verb_phrase.append(None)
        else :
            verb_phrase.append(None)
    return verb_phrase
            

        
        


# In[5]:


def find_subj(parse_tree):
    for child  in parse_tree :
        #print (child.label())
        if child.label() == "NP":
            return child.leaves()
    return []
def find_VP(parse_tree):
    for child  in parse_tree :
        #print (child.label())
        if child.label() == "VP":
            return child.leaves()
    return []
def QSG_Rule_6_1(chunks , parse_trees , is_clause_val, ner_split, tokens,verb_phrases):
    QA = []
    for chunk,parse_tree,is_cl,ner,tok,verb in zip(chunks , parse_trees , is_clause_val, ner_split,tokens,verb_phrases) :
    # Counter example for is_cl - "A house is for $100,00"
        rule_grammar = "{<IN>+<\$>*<CD>+}"
        #print(rule_grammar)
        seg_pos = parse_tree.pos()
        #print(seg_pos)
        rule6_1_chunks = parse_chunks(seg_pos,rule_grammar)
        #print(rule6_1_chunks)
        prep_chunk = find_chunk(rule6_1_chunks)
        #print(prep_chunk)
        if prep_chunk :
            prep_pos =  [ pos[0] for pos in prep_chunk[0].pos() ]
            prep_tokens = [ pos[0][0] for pos in prep_chunk[0].pos() ]
            prep_part_tokens = [ p[0] for p in find_chunk(parse_chunks(prep_pos , "{<IN>+}" ))[0].leaves()]
            #print(prep_part_tokens)
            answer_words = [x for x in prep_tokens if x not in prep_part_tokens ] 
            #print(answer)
            subject = find_subj(parse_tree)
            VP = find_VP(parse_tree)
            rem_verb_phrase =  verb if verb in VP else VP[0][0]
            VP = VP[1:]
            
            #Calc rest of VP for Question Generation
            [VP.remove(x) for x in prep_part_tokens + answer_words if x in VP]
            #calc rest of the sentence for question generation
            answer = " ".join(answer_words)
            tok = tok[:]
            [tok.remove(x) for x in subject + VP + prep_part_tokens + answer_words + [rem_verb_phrase] if x in tok ]
            quest_tok = prep_part_tokens + ["how","much"]+ [verb] + subject + VP + tok + ["?"]
            #print(quest_tok)
            question = " ".join(quest_tok)
            QA.append({"Q" : question , "A" : answer })
            
    return QA

def QSG_Rule_6_2(chunks , parse_trees , is_clause_val, ner_split, tokens,verb_phrases):
    QA = []
    for chunk,parse_tree,is_cl,ner,tok,verb in zip(chunks , parse_trees , is_clause_val, ner_split,tokens,verb_phrases) :
        rule_grammar = "{<\$>*<CD>+<MD>?<VB|VBD|VBG|VBP|VBN|VBZ|IN>+}"
        #print(rule_grammar)
        seg_pos = parse_tree.pos()
        #print(seg_pos)
        rule6_2_chunks = parse_chunks(seg_pos,rule_grammar)
        #print(rule6_2_chunks)
        rule_chunk = find_chunk(rule6_2_chunks)
        #print(rule_chunk)
        if rule_chunk :
            rule_pos =  [ pos[0] for pos in rule_chunk[0].pos() ]
            rule_tokens = [ pos[0][0] for pos in rule_chunk[0].pos() ]
            prep_part_tokens = [ p[0] for p in find_chunk(parse_chunks(rule_pos , "{<\$>*<CD>+}" ))[0].leaves()]
            #print(prep_part_tokens)
            answer_words = prep_part_tokens
            [rule_tokens.remove(x) for x in answer_words if x in rule_tokens ]
            #calc rest of the sentence for question generation
            answer = " ".join(answer_words)
            tok = tok[:]
            [tok.remove(x) for x in answer_words + rule_tokens ]
            #print(answer)
            quest_tok = ["how","much"]+ rule_tokens + tok + ["?"]
            #print(quest_tok)
            question = " ".join(quest_tok)
            QA.append({"Q" : question , "A" : answer })
    return QA

def QSG_Rule_6_3(chunks , parse_trees , is_clause_val, ner_split, tokens,verb_phrases):
    QA = []
    for chunk,parse_tree,is_cl,ner,tok,verb in zip(chunks , parse_trees , is_clause_val, ner_split,tokens,verb_phrases) :
    # Counter example for is_cl - "A house is for $100,00"
        rule_grammar = "{<MD>?<VB|VBD|VBG|VBP|VBN|VBZ>+<IN>?<NN|NNS|NNP|NNPS|PRP|PRP\$>?<\$>*<CD>+}"
        #print(rule_grammar)
        seg_pos = parse_tree.pos()
        #print(seg_pos)
        rule6_3_chunks = parse_chunks(seg_pos,rule_grammar)
        #print(rule6_1_chunks)
        prep_chunk = find_chunk(rule6_3_chunks)
        #print(prep_chunk)
        if prep_chunk :
            prep_pos =  [ pos[0] for pos in prep_chunk[0].pos() ]
            #print(prep_pos)
            prep_tokens = [ pos[0][0] for pos in prep_chunk[0].pos() ]
            ans_tokens = [p[0]for p in find_chunk(parse_chunks(prep_pos , "{<\$>*<CD>+}"))[0].leaves()]
            
            #print(prep_part_tokens)
            VP = find_VP(parse_tree)
            rem_verb_phrase =  verb if verb in VP else VP[0]
            [prep_tokens.remove(x) for x in ans_tokens + [rem_verb_phrase] if  x in prep_tokens ]
            #print(verb)
            #print(prep_tokens)
            subject = find_subj(parse_tree)
            #print(subject)
            
            #Calc rest of VP for Question Generation
            [VP.remove(x) for x in prep_tokens + ans_tokens if x in VP]
            #calc rest of the sentence for question generation
            answer = " ".join(ans_tokens)
            tok = tok[:]
            [tok.remove(x) for x in subject + VP + prep_tokens + ans_tokens + [rem_verb_phrase] if x in tok ]
            quest_tok =  ["how","much"]+ [verb] + subject + prep_tokens + VP + tok + ["?"]
            #print(quest_tok)
            question = " ".join(quest_tok)
            QA.append({"Q" : question , "A" : answer })
    return QA
                
                

#print(QSG_Rule_6_1(chunks , parse_trees , is_clause_val, ner_split, tokens,verb_phrases))
#print(QSG_Rule_6_2(chunks , parse_trees , is_clause_val, ner_split, tokens,verb_phrases))
#print(QSG_Rule_6_3(chunks , parse_trees , is_clause_val, ner_split, tokens,verb_phrases))


# In[6]:


def find_ner_tags_for_pos (ners,pos):
    return list(filter(lambda x : x[0] in [p[0] for p in pos] , ners))

def QSG_Rule_1(chunks , parse_trees , is_clause_val, ner_split, tokens,verb_phrases):
    QA = []
    for chunk,parse_tree,is_cl,ner,tok in zip(chunks , parse_trees , is_clause_val, ner_split,tokens) :
        if is_cl :
            chunk_pos = [ pos[0] for pos in find_chunk(chunk)[0].pos() ]
            grammar = "{<DT>?<JJ.?>*<NN.?|PRP|PRP$|POS|IN|DT|CC|VBG|VBN>+}"
            rule1_chunks = parse_chunks(chunk_pos,grammar)
            noun_chunk = find_chunk(rule1_chunks)
            if noun_chunk :
                noun_pos = noun_chunk[0].leaves()
                #print(ner)
                #print(noun_pos)
                tok = tok[:]
                ner_tags = find_ner_tags_for_pos(ner,noun_pos)
                qsd4,q_disambg = QSD_Rule_4(ner_tags,"QSG_RULE_1")
                answer_words = [ p[0] for p in noun_pos ]
                #print(tok)
                #print(answer_words)
                [ tok.remove(ans) for ans in answer_words if ans in tok]
                answer = " ".join(answer_words)
                #print(tok)
                quest_tok = [q_disambg] + tok + ["?"]
                #print(quest_tok)
                question = " ".join(quest_tok)
                QA.append({"Q" : question , "A" : answer })
    return QA
            
#QSG_Rule_1(chunks , parse_trees , is_clause_val, ner_split, tokens,verb_phrases)            
            


# In[7]:


def QSG_Rule_7(chunks , parse_trees , is_clause_val, ner_split, tokens,verb_phrases):
    QA = []
    for chunk,parse_tree,is_cl,ner,tok,verb in zip(chunks , parse_trees , is_clause_val, ner_split,tokens,verb_phrases) :
    # Counter example for is_cl - "A house is for $100,00"
        rule_grammar = "{<DT>?<CD>+<RB>?<JJ|JJR|JJS>?<NN|NNS|NNP|NNPS|VBG>+}"
        #print(rule_grammar)
        seg_pos = parse_tree.pos()
        #print(seg_pos)
        rule7_chunks = parse_chunks(seg_pos,rule_grammar)
        #print(rule7_chunks)
        prep_chunk = find_chunk(rule7_chunks)
        #print(prep_chunk)
        if prep_chunk :
            prep_pos =  [ pos[0] for pos in prep_chunk[0].pos() ]
            #print(prep_pos)
            prep_tokens = [ pos[0][0] for pos in prep_chunk[0].pos() ]
            ans_tokens = [p[0]for p in find_chunk(parse_chunks(prep_pos , "{<CD>+}"))[0].leaves()]
            
            #print(prep_part_tokens)
            VP = find_VP(parse_tree)
            if not VP :
                break
            rem_verb_phrase =  verb if verb in VP else VP[0]
            [prep_tokens.remove(x) for x in ans_tokens if  x in prep_tokens ]
            #print(verb)
            #print(prep_tokens)
            subject = find_subj(parse_tree)
            #print(subject)
            
            #Calc rest of VP for Question Generation
            [VP.remove(x) for x in prep_tokens + ans_tokens + [rem_verb_phrase] if x in VP]
            #calc rest of the sentence for question generation
            answer = " ".join(ans_tokens)
            tok = tok[:]
            [tok.remove(x) for x in subject + VP + prep_tokens + ans_tokens + [rem_verb_phrase] if x in tok ]
            quest_tok =  ["how","many"]+ prep_tokens + [verb] + subject + VP + tok + ["?"]
            #print(quest_tok)
            question = " ".join(quest_tok)
            QA.append({"Q" : question , "A" : answer })
    return QA


# In[8]:


def QSG_Rule_3(chunks , parse_trees , is_clause_val, ner_split, tokens,verb_phrases):
    QA = []
    for chunk,parse_tree,is_cl,ner,tok,verb in zip(chunks , parse_trees , is_clause_val, ner_split,tokens,verb_phrases) :
    # Counter example for is_cl - "A house is for $100,00"
        rule_grammar = "{<PRP\$|POS>+<RB.?>*<JJ.?>*<NN.?|VBG|VBN>+<VB.?|MD|RP>+}"
        #print(rule_grammar)
        seg_pos = parse_tree.pos()
        #print(seg_pos)
        rule3_chunks = parse_chunks(seg_pos,rule_grammar)
        #print(rule3_chunks)
        prep_chunk = find_chunk(rule3_chunks)
        #print(prep_chunk)
        if prep_chunk :
            prep_pos =  [ pos[0] for pos in prep_chunk[0].pos() ]
            #print(prep_pos)
            prep_tokens = [ pos[0][0] for pos in prep_chunk[0].pos() ]
            ans_tokens = [p[0]for p in find_chunk(parse_chunks(prep_pos , "{<PRP\$|POS>+}"))[0].leaves()]
            
            #print(prep_part_tokens)
            [prep_tokens.remove(x) for x in ans_tokens if  x in prep_tokens ]
            #print(verb)
            #print(prep_tokens)
            #subject = find_subj(parse_tree)
            #print(subject)
            
            #Calc rest of VP for Question Generation
            #[VP.remove(x) for x in prep_tokens + ans_tokens if x in tok]
            #calc rest of the sentence for question generation
            answer = " ".join(ans_tokens)
            tok = tok[:]
            [tok.remove(x) for x in prep_tokens + ans_tokens  if x in tok ]
            quest_tok =  ["Whose"]+ prep_tokens + tok + ["?"]
            #print(quest_tok)
            question = " ".join(quest_tok)
            QA.append({"Q" : question , "A" : answer })
    return QA


# In[9]:


def QSG_Rule_4(chunks , parse_trees , is_clause_val, ner_split, tokens,verb_phrases):
    QA = []
    for chunk,parse_tree,is_cl,ner,tok,verb in zip(chunks , parse_trees , is_clause_val, ner_split,tokens,verb_phrases) :
    # Counter example for is_cl - "A house is for $100,00"
        rule_grammar = "{<DT>?<JJ.?>?<RB>?<IN|TO|RP>+<DT>*<JJ.?>*<NN.?|PP|PRP|PRP\$ >+<VBG|POS|CD|RB|DT>*}"
        #print(rule_grammar)
        seg_pos = parse_tree.pos()
        #print(seg_pos)
        rule4_chunks = parse_chunks(seg_pos,rule_grammar)
        #print(rule4_chunks)
        prep_chunk = find_chunk(rule4_chunks)
        #print(prep_chunk)
        if prep_chunk :
            prep_pos =  [ pos[0] for pos in prep_chunk[0].pos() ]
            #print(prep_pos)
            ans_tokens = [ pos[0][0] for pos in prep_chunk[0].pos() ]
            prep_tokens = [p[0]for p in find_chunk(parse_chunks(prep_pos , "{<IN>+}"))[0].leaves()] if find_chunk(parse_chunks(prep_pos , "{<IN>+}")) else None
            #print(prep_tokens)
            if not prep_tokens  or  not any([x in ["under", "across", "around", "along", "through", "over", "into", "onto"] for x in prep_tokens]):
                break;
            
            VP = find_VP(parse_tree)
            if not VP :
                break
            rem_verb_phrase =  verb if verb in VP else VP[0]
            
            #print(prep_tokens)
            #print(VP)
            #print(verb)
            #print(prep_tokens)
            subject = find_subj(parse_tree)
            #print(subject)
            answer = " ".join(ans_tokens)
            #Calc rest of VP for Question Generation
            #[VP.remove(x) for x in ans_tokens + [rem_verb_phrase] if x in VP]
            #calc rest of the sentence for question generation
            VP = " ".join(VP).replace(answer , "").split(" ")
            VP.remove(rem_verb_phrase)
            tok = tok[:]
            [tok.remove(x) for x in ans_tokens + subject + VP + [rem_verb_phrase] if x in tok ]
            #print(tok)
            quest_tok =  ["Where"]+ [verb] + subject + VP + tok + ["?"]
            #print(quest_tok)
            question = " ".join(quest_tok)
            QA.append({"Q" : question , "A" : answer })
    return QA

#QSG_Rule_4(chunks , parse_trees , is_clause_val, ner_split, tokens,verb_phrases)


# In[10]:


def QSG_Rule_5(chunks , parse_trees , is_clause_val, ner_split, tokens,verb_phrases):
    QA = []
    for chunk,parse_tree,is_cl,ner,tok,verb in zip(chunks , parse_trees , is_clause_val, ner_split,tokens,verb_phrases) :
    # Counter example for is_cl - "A house is for $100,00"
        rule_grammar = "{<DT>?<JJ.?>?<RB>?<IN|TO|RP>+<DT>*<NN.?>+}"
        #print(rule_grammar)
        seg_pos = parse_tree.pos()
        #print(seg_pos)
        rule5_chunks = parse_chunks(seg_pos,rule_grammar)
        #print(rule5_chunks)
        prep_chunk = find_chunk(rule5_chunks)
        #print(prep_chunk)
        if prep_chunk :
            prep_pos =  [ pos[0] for pos in prep_chunk[0].pos() ]
            #print(prep_pos)
            prep_tokens = [ pos[0][0] for pos in prep_chunk[0].pos() ]
            ans_pos = [p for p in find_chunk(parse_chunks(prep_pos , "{<IN|TO|RP>+<DT>*<NN.?>+}"))[0].leaves()]
            ans_tokens = [ pos[0] for pos in ans_pos ]
            [prep_tokens.remove(x) for x in ans_tokens]
            #print(prep_tokens)
            ans_ner = [ p[1] for p in find_ner_tags_for_pos(ner,ans_pos) ]
            not_date_time_ner = not("DATE" in ans_ner or "TIME" in ans_ner )
            #print(ans_ner)
            if not_date_time_ner and (not any([x.lower() in ["tomorrow","yesterday", "today", "tonight", "am", "pm"] for x in ans_tokens])):
                break;
            
            VP = find_VP(parse_tree)
            if not VP :
                break
            rem_verb_phrase =  verb if verb in VP else VP[0]
            
            #print(prep_tokens)
            #print(VP)
            #print(verb)
            #print(prep_tokens)
            subject = find_subj(parse_tree)
            #print(subject)
            answer = " ".join(ans_tokens)
            #Calc rest of VP for Question Generation
            #[VP.remove(x) for x in ans_tokens + [rem_verb_phrase] if x in VP]
            #calc rest of the sentence for question generation
            VP = " ".join(VP).replace(answer , "").split(" ")
            VP.remove(rem_verb_phrase)
            [VP.remove(x) for x in prep_tokens ]
            tok = tok[:]
            [tok.remove(x) for x in ans_tokens + subject + VP + [rem_verb_phrase] + prep_tokens if x in tok ]
            #print(tok)
            quest_tok =  ["When"]+ [verb] + subject + VP + prep_tokens + tok + ["?"]
            #print(quest_tok)
            question = " ".join(quest_tok)
            QA.append({"Q" : question , "A" : answer })
    return QA

#QSG_Rule_5(chunks , parse_trees , is_clause_val, ner_split, tokens,verb_phrases)


# In[11]:


def QSG_Rule_2_4(chunks , parse_trees , is_clause_val, ner_split, tokens,verb_phrases):
    QA = []
    for chunk,parse_tree,is_cl,ner,tok,verb in zip(chunks , parse_trees , is_clause_val, ner_split,tokens,verb_phrases) :
    # Counter example for is_cl - "A house is for $100,00"
        rule_grammar = "{<TO>+<VB|VBP|RP>+<DT>?<RB.?>*<JJ.?>*<NN.?|PRP|PRP\$|POS|VBG|DT>*}"
        #print(rule_grammar)
        seg_pos = parse_tree.pos()
        #print(seg_pos)
        rule2_4_chunks = parse_chunks(seg_pos,rule_grammar)
        #print(rule2_4_chunks)
        prep_chunk = find_chunk(rule2_4_chunks)
        #print(prep_chunk)
        if prep_chunk :
            prep_pos =  [ pos[0] for pos in prep_chunk[0].pos() ]
            #print(prep_pos)
            prep_tokens = [ pos[0][0] for pos in prep_chunk[0].pos() ]
            ans_chunk = find_chunk(parse_chunks(prep_pos , "{<DT>?<RB.?>*<JJ.?>*<NN.?|PRP|PRP\$|POS|VBG|DT>*}"))
            ans_NP = None
            if ans_chunk :
                ans_NP = True
                rep_chunk = find_chunk(parse_chunks(prep_pos , "{<TO>+<VB|VBP|RP>+}"))
                
            else :
                ans_chunk = find_chunk(parse_chunks(prep_pos , "{<TO>+<VB|VBP|RP>+}"))
                ans_NP = False
            
            ans_pos = [p for p in ans_chunk[0].leaves()]
            ans_tokens = [ pos[0] for pos in ans_pos ]
            #print(ans_pos)
            #print(ans_tokens)
            if ans_NP :
                rep_pos = [p for p in rep_chunk[0].leaves()]
                rep_tokens = [ pos[0] for pos in rep_pos ]
            else:
                rep_tokens = ["to" , "do"]
            
            prep = " ".join(prep_tokens)
            rep = " ".join(rep_tokens)
            #print(prep)
            
            #print(rep)
            rem_VP = rep
            #print(rem_VP)
            #print(rep_tokens)
            #rem_VP = " ".join(prep_tokens).replace(" ".join(ans_tokens)," ".join(rep_tokens))
            #print(rem_VP)
            [prep_tokens.remove(x) for x in ans_tokens]
            #print(prep_tokens)
            #ans_ner = [ p[1] for p in find_ner_tags_for_pos(ner,ans_pos) ]
            #not_date_time_ner = not("DATE" in ans_ner or "TIME" in ans_ner )
            #print(ans_ner)
            #if not_date_time_ner and (not any([x.lower() in ["tomorrow","yesterday", "today", "tonight", "am", "pm"] for x in ans_tokens])):
                #break;
            
            VP = find_VP(parse_tree)
            if not VP :
                break
            
            #print(prep_tokens)
            #print(VP)
            #print(verb)
            #print(prep_tokens)
            subject = find_subj(parse_tree)
            #print(subject)
            answer = " ".join(ans_tokens)
            #Calc rest of VP for Question Generation
            [VP.remove(x) for x in ans_tokens + rem_VP.split(" ") if x in VP]
            #calc rest of the sentence for question generation
            #VP = " ".join(VP).replace(answer , "").split(" ")
            #VP.remove(rem_verb_phrase)
            #[VP.remove(x) for x in prep_tokens ]
            tok = tok[:]
            [tok.remove(x) for x in ans_tokens + subject + VP + rem_VP.split(" ") + ans_tokens if x in tok ]
            #print(tok)
            quest_tok =  ["What"]+ [verb] + subject + VP + rem_VP.split(" ") + tok + ["?"]
            #print(quest_tok)
            question = " ".join(quest_tok)
            QA.append({"Q" : question , "A" : answer })
    return QA

#QSG_Rule_2_4(chunks , parse_trees , is_clause_val, ner_split, tokens,verb_phrases)


# In[12]:


#Bringing here the implementation of QSD rule befor implementing QSG 2.1 - 2.4 
def find_ner_tag_for_token(ners,token):
    for tag in ner :
        if tag[0] == token :
            return tag[1]
    return None

def find_ner_tag_for_tokens(ners,tokens):
    return list(filter(lambda x : x[0] in tokens , ners))

def get_pos_tokens_from_chunk_tree(chunk_tree):
    rep_pos = [p for p in chunk_tree.leaves()] if chunk_tree else []
    rep_tokens = [ pos[0] for pos in rep_pos ] if chunk_tree else []
    return rep_pos,rep_tokens
def QSD_Rule_1(chunk_pos) :
    #print(ner_chunk_tags)
    disambg_value =  all( [x == "PRP" for x in [p[1] for p in chunk_pos]])
    if disambg_value :
        return disambg_value,"whom"
    else:
        return disambg_value,"what"
def QSD_Rule_2(parse_tree):
    VP = None
    #parse_tree.draw()
    for child  in parse_tree :
        #print (child.label())
        if child.label() == "VP":
            VP =  child
    if VP :
        NP_in_VP = []
        for child  in VP :
            if child.label() == "NP":
                NP_in_VP.append(child)
        #print(NP_in_VP)
        if len(NP_in_VP) > 1 :
            return True , NP_in_VP[0].leaves() , NP_in_VP[1].leaves()
        else :
            return False , [] , NP_in_VP[0].leaves() if NP_in_VP else []
    else :
        return False , [] , []

def QSD_Rule_3(chunk_pos,chunk_ners) :
    first_noun_chunk  = find_chunk(parse_chunks(chunk_pos , "{<NN.?>+}"))
    if first_noun_chunk :
        first_noun_pos, first_noun_tokens = get_pos_tokens_from_chunk_tree(first_noun_chunk[0])
        if find_ner_tag_for_token(chunk_ners,first_noun_tokens[0]) == "PERSON":
            return True,"Whom"
        else:
            return False,"What"
    else: 
        return False,"What"

    
        
def QSD_Rule_4(ner_chunk_tags,QSG_rule) :
    #print(ner_chunk_tags)
    disambg_value =  ner_chunk_tags[0][1] in ['LOCATION','ORGANIZATION', 'CITY','COUNTRY']
    if QSG_rule == "QSG_RULE_1" :
        if disambg_value :
            return disambg_value,"what"
        elif ner_chunk_tags[0][1] in ['PERSON']:
            return disambg_value,"who"
        else:
            return disambg_value,"who"
    if QSG_rule == "QSG_RULE_2_1":
        if disambg_value :
            return disambg_value,"where"
        else :
            return disambg_value,"To what"
    if QSG_rule == "QSG_RULE_2_2":
        if disambg_value :
            return disambg_value,"where"
        else :
            return disambg_value,"what"

def QSD_Rule_5(chunk_pos,chunk_ners) :
    noun_chunk  = find_chunk(parse_chunks(chunk_pos , "{<NN.?>+}"))
    if noun_chunk :
        noun_pos, noun_tokens = get_pos_tokens_from_chunk_tree(noun_chunk[0])
        noun_ners = find_ner_tag_for_tokens(chunk_ners,noun_tokens)
        ners = set([ x[1] for x in noun_ners])
        if "TIME" in ners or "DATE" in ners :
            return "when"
        else:
            return "what"
    else: 
        return "What"

    
        


# In[13]:


def QSG_Rule_2_1(chunks , parse_trees , is_clause_val, ner_split, tokens,verb_phrases):
    QA = []
    for chunk,parse_tree,is_cl,ner,tok,verb in zip(chunks , parse_trees , is_clause_val, ner_split,tokens,verb_phrases) :
    # Counter example for is_cl - "A house is for $100,00"
        rule_grammar = "{<TO>+<DT>?<RB.?>*<JJ.?>*<NN.?|PRP|PRP\$|VBG|DT|POS|CD|VBN>+}"
        seg_pos = parse_tree.pos()
        rule2_1_chunks = parse_chunks(seg_pos,rule_grammar)
        seg = " ".join([p[0] for p in seg_pos ])
        #print(seg)
        #print(rule2_1_chunks)
        prep_chunk = find_chunk(rule2_1_chunks)
        prep_pos,prep_tokens = get_pos_tokens_from_chunk_tree(prep_chunk[0]) if prep_chunk else (None,None)
        #print(prep_pos)
        #print(prep_tokens)
        prep = " ".join(prep_tokens) if prep_tokens else []
        clause_chunk  = find_chunk(chunk)
        #print(clause_chunk)
        if len(clause_chunk) > 1 :
            clause_strings = [" ".join(get_pos_tokens_from_chunk_tree(c)[1]) for c in clause_chunk ]
            prep_index = seg.index(seg)
            clause_index  = [abs(seg.index(cs)-prep_index)  for cs in clause_strings ]
            cl_chunk = clause_chunk[clause_index.index(min(clause_index))]
            cl_string = " ".split(clause_strings[clause_index.index(min(clause_index))])
            #print(cl_string)
            verb = verb_phrase_identification([findLCA(parse_tree,cl_string[0],cl_string[-1])],[True],[cl_chunk])
            #print(verb)
        else :
            cl_chunk = clause_chunk[0] if clause_chunk else None
        cl_pos,cl_tokens = get_pos_tokens_from_chunk_tree(cl_chunk)
        #print(seg_pos)
        #print(prep_chunk)
        #print(cl_chunk)
        if prep_chunk :
            
            ques = "To what"
            qsd1,ques = QSD_Rule_1(prep_pos)
            if qsd1 :
                ques = "To " + ques
            else :
                prep_ners = find_ner_tag_for_tokens(ner,prep_tokens)
                qsd3,ques = QSD_Rule_3(prep_pos,prep)
                ques = "To " + ques
                if not qsd3 :
                    qsd4,ques = QSD_Rule_4(prep_ners,"QSG_RULE_2_1")
                
            #print(ques)
            #print(verb)
                
            VP = find_VP(parse_tree)
            #print(parse_tree)
            if not VP :
                break
            rem_verb_phrase =  verb if verb in VP else VP[0]
            #print(VP)
            #print(verb)
            
            #print(prep_part_tokens)
            #[prep_tokens.remove(x) for x in ans_tokens if  x in prep_tokens ]
            #print(verb)
            #print(prep_tokens)
            subject = find_subj(parse_tree)
            #print(subject)
            
            #Calc rest of VP for Question Generation
            [VP.remove(x) for x in prep_tokens + [rem_verb_phrase]  if x in tok]
            #calc rest of the sentence for question generation
            #answer = " ".join(ans_tokens)
            tok = tok[:]
            [tok.remove(x) for x in prep_tokens + VP + [rem_verb_phrase] + subject   if x in tok ]
            quest_tok =  [ques]+ [verb] + subject + VP  + tok + ["?"]
            #print(quest_tok)
            question = " ".join(quest_tok)
            QA.append({"Q" : question , "A" : prep })
    return QA

#QSG_Rule_2_1(chunks , parse_trees , is_clause_val, ner_split, tokens,verb_phrases)


# In[14]:


def QSG_Rule_2_2(chunks , parse_trees , is_clause_val, ner_split, tokens,verb_phrases):
    QA = []
    for chunk,parse_tree,is_cl,ner,tok,verb in zip(chunks , parse_trees , is_clause_val, ner_split,tokens,verb_phrases) :
    # Counter example for is_cl - "A house is for $100,00"
        rule_grammar = "{<IN>+<DT>?<RB.?>*<JJ.?>*<NN.?|PRP|PRP\$|POS|VBG|DT|CD|VBN>+}"
        seg_pos = parse_tree.pos()
        rule2_2_chunks = parse_chunks(seg_pos,rule_grammar)
        seg = " ".join([p[0] for p in seg_pos ])
        #print(seg)
        #print(rule2_2_chunks)
        prep_chunk = find_chunk(rule2_2_chunks)
        prep_pos,prep_tokens = get_pos_tokens_from_chunk_tree(prep_chunk[0]) if prep_chunk else (None,None)
        #print(prep_pos)
        #print(prep_tokens)
        prep = " ".join(prep_tokens) if prep_tokens else []
        clause_chunk  = find_chunk(chunk)
        
        if len(clause_chunk) > 1 :
            clause_strings = [" ".join(get_pos_tokens_from_chunk_tree(c)[1]) for c in clause_chunk ]
            prep_index = seg.index(seg)
            clause_index  = [abs(seg.index(cs)-prep_index)  for cs in clause_strings ]
            cl_chunk = clause_chunk[clause_index.index(min(clause_index))]
            cl_string = clause_strings[clause_index.index(min(clause_index))].split(" ")
            verb = verb_phrase_identification([findLCA(parse_tree,cl_string[0],cl_string[-1])],[True],[cl_chunk])[0]
        else :
            cl_chunk = clause_chunk[0] if clause_chunk else None
        cl_pos,cl_tokens = get_pos_tokens_from_chunk_tree(cl_chunk)
        #print(cl_pos)
        #print(cl_tokens)
        if prep_chunk :
            q_prep = find_chunk(parse_chunks(seg_pos,"{<IN+>}"))[0]
            q_prep_pos,q_prep_tokens = get_pos_tokens_from_chunk_tree(q_prep)
            #print(q_prep_tokens)
            ques = "what"
            qsd1,ques = QSD_Rule_1(prep_pos)
            if not qsd1 :
                prep_ners = find_ner_tag_for_tokens(ner,prep_tokens)
                qsd3,ques = QSD_Rule_3(prep_pos,prep_ners)
                ques = ques
                if not qsd3 :
                    qsd4,ques = QSD_Rule_4(prep_ners,"QSG_RULE_2_2")
                if not qsd4 :
                    ques = QSD_Rule_5(prep_pos,prep_ners)
            #print(ques)
                
            VP = find_VP(parse_tree)
            if not VP :
                break
            rem_verb_phrase =  verb if verb in VP else VP[0]
            #print(VP)
            #print(verb)
            
            #print(prep_part_tokens)
            #[prep_tokens.remove(x) for x in ans_tokens if  x in prep_tokens ]
            #print(verb)
            #print(prep_tokens)
            subject = find_subj(parse_tree)
            #print(subject)
            
            #Calc rest of VP for Question Generation
            [VP.remove(x) for x in prep_tokens + [rem_verb_phrase]  if x in tok]
            #calc rest of the sentence for question generation
            #answer = " ".join(ans_tokens)
            tok = tok[:]
            [tok.remove(x) for x in prep_tokens + VP + [rem_verb_phrase] + subject   if x in tok ]
            quest_tok = q_prep_tokens +[ques]+ [verb] + subject + VP  + tok + ["?"]
            #print(quest_tok)
            question = " ".join(quest_tok)
            QA.append({"Q" : question , "A" : prep })
    return QA

#QSG_Rule_2_2(chunks , parse_trees , is_clause_val, ner_split, tokens,verb_phrases)


# In[15]:


def QSG_Rule_2_3(chunks , parse_trees , is_clause_val, ner_split, tokens,verb_phrases):
    QA = []
    for chunk,parse_tree,is_cl,ner,tok,verb in zip(chunks , parse_trees , is_clause_val, ner_split,tokens,verb_phrases) :
    # Counter example for is_cl - "A house is for $100,00"
        rule_grammar = "{<VB.?|MD|RP|RB.?>+<DT>?<RB.?>*<JJ.?>*<NN.?|PRP|PRP\$|POS|VBG|DT|CD|VBN>+}"
        seg_pos = parse_tree.pos()
        rule2_3_chunks = parse_chunks(seg_pos,rule_grammar)
        seg = " ".join([p[0] for p in seg_pos ])
        #print(seg)
        #print(rule2_3_chunks)
        prep_chunk = find_chunk(rule2_3_chunks)
        prep_pos,prep_tokens = get_pos_tokens_from_chunk_tree(prep_chunk[0]) if prep_chunk else (None,None)
        #print(prep_pos)
        #print(prep_tokens)
        prep = " ".join(prep_tokens) if prep_tokens else []
        
        #print(cl_pos)
        #print(cl_tokens)
        if prep_chunk :
            ques = "what"
            two_ques = False
            qsd1,ques = QSD_Rule_1(prep_pos)
            prep_ners = find_ner_tag_for_tokens(ner,prep_tokens)
            qsd2 , prp_tokens , noun_tokens = QSD_Rule_2(parse_tree)
            two_ques = qsd2
            if not qsd2 :
                qsd3,ques = QSD_Rule_3(prep_pos,prep_ners)
            #print(ques)
            VP = find_VP(parse_tree)
            if not VP :
                break
            #print(VP)
            rem_verb_phrase =  verb if verb in VP else VP[0]
            
            #print(verb)
            subject = find_subj(parse_tree)
            #print(subject)
            ans_tokens = noun_tokens
            if two_ques :
                ques1_ans = prp_tokens
                VP_q1 = VP[:]
                tok_q1 = tok[:]
                [VP_q1.remove(x) for x in ques1_ans  if x in VP_q1]
                [tok_q1.remove(x) for x in ques1_ans + VP_q1 + [rem_verb_phrase] + subject   if x in tok_q1 ]
                q1_token = ["Whom"]+ [verb] + subject + VP_q1  + tok_q1 + ["?"]
                ques1 = " ".join(q1_token)
                QA.append({"Q" : ques1 , "A" : " ".join(ques1_ans)  })



            #print(prep_part_tokens)
            #[prep_tokens.remove(x) for x in ans_tokens if  x in prep_tokens ]
            #print(verb)
            #print(prep_tokens)



            #Calc rest of VP for Question Generation
            [VP.remove(x) for x in ans_tokens  if x in VP]
            #calc rest of the sentence for question generation
            answer = " ".join(ans_tokens)
            tok = tok[:]
            [tok.remove(x) for x in ans_tokens + VP + [rem_verb_phrase] + subject   if x in tok ]
            quest_tok = [ques]+ [verb] + subject + VP  + tok + ["?"]
            #print(quest_tok)
            question = " ".join(quest_tok)
            QA.append({"Q" : question , "A" : answer  })
    return QA

#QSG_Rule_2_3(chunks , parse_trees , is_clause_val, ner_split, tokens,verb_phrases)


# In[16]:

if __name__ == '__main__':

    nlp = StanfordCoreNLP('http://localhost', port=9000, logging_level=logging.DEBUG)

    #sentence = "Barack Obama is the president of The United States of America."
    #sentence = "IIIT Hyderabad is the venue of IASNLP-2018."
    #sentence = "The boy went by bus."
    #sentence =  sentence.rstrip().rstrip(".")

    # sentence = """ Deputy Chief Minister and Services Department Minister-in-charge Manish Sisodia ordered the Department
    # to change the approving authority for transfers of officers from the Lieutenant-Governor and bureaucrats
    # to the Chief Minister and Ministers."""

    # sentence = """The contractor will build you a house for $100,000 dollars."""
    #
    # sentence_input = """The contractor will build you a house for $100,000 dollars."""



    #sentence = "The contractor will build you a house for $100,000 dollars."
    #sentence = "The book might cost me $10."
    #sentence="The book might cost me $10 from the store."
    #sentence = "$100,000 builds a house out of sticks."
    #sentence = "The bill will cost them 500 million dollars in India."
    #sentence = "His name is Robinson."
    #sentence = "She will quickly pour the sticky liquid into the green flowery pot."
    #sentence = "I am going quickly back on Saturday."
    #sentence = "He wants to become a good doctor."
    #sentence = "I want to work."
    #sentence = "He hurriedly left the class in the morning."
    #sentence = "He is addicted to smoking."
    #sentence = "He will go by bus."
    #sentence = "John gave Mary a book." #design more rules to catch the essence
    #sentence = "He gave him a book."
    #sentence = "He will buy a book."
    #sentence = "He gave him a book."
    #sentence = "John gave Mary a book."


    # In[23]:

    st.title('Automatic Question Generator')

    sentence = st.text_input('Enter the Paragragh here')
    sentence_input = sentence

    if st.button(label='Generate'):
        #All preprocessing
        segments = sentence.rstrip().rstrip(".").split(", ")
        tree = Tree.fromstring(nlp.parse(sentence))
        ner = nlp.ner(sentence)
        tokens = [word_tokenize(segment) for segment in segments]
        parse_trees = [findLCA(tree,seg[0],seg[-1]) for seg in tokens]
        ner_split = [list(g) for k,g in itertools.groupby(ner,lambda x: x[0] == ',') if not k]
        clause_identification_grammar = "{<DT>?<JJ.?>*<\$|CD|NN.?|PRP|PRP\$|POS|IN|DT|CC|VBG|VBN>+<RB.?|VB.?|MD|RP>+}"
        chunks = [ parse_chunks(pos_tag(word_tokenize(segment)) ,clause_identification_grammar) for segment in segments ]
        is_clause_val = is_clause(chunks)

        new_trees,enrichment_update =  enrich_VPs(parse_trees)
        parse_trees = new_trees
        update_indices = indices = [i for i, x in enumerate(enrichment_update) if x == True]

        for index in update_indices :
            tree = parse_trees[index]
            #Not changing original segments as they will be used later to form questions
            #segments[index] =  " ".join(tree.leaves())
            chunks[index] = parse_chunks(tree.pos() ,clause_identification_grammar)
            ner_split[index]  = nlp.ner(" ".join(tree.leaves()))
            tokens[index] = tree.leaves()
            #print(ner_split)
            #print(tree.pos())

        verb_phrases = verb_phrase_identification(parse_trees,is_clause_val,chunks)
            #all the pre processed data we have
        #print(segments)
        #print(chunks)
        #print(parse_trees)
        #for ptree in parse_trees :
            #ptree.draw()
        #print(verb_phrases)
        #print(is_clause_val)
        #print(tokens)
        #print(ner_split)



        # In[24]:


        # print(QSG_Rule_1(chunks , parse_trees , is_clause_val, ner_split, tokens,verb_phrases))
        # print(QSG_Rule_2_1(chunks , parse_trees , is_clause_val, ner_split, tokens,verb_phrases))
        # print(QSG_Rule_2_2(chunks , parse_trees , is_clause_val, ner_split, tokens,verb_phrases))
        # print(QSG_Rule_2_3(chunks , parse_trees , is_clause_val, ner_split, tokens,verb_phrases))
        # print(QSG_Rule_2_4(chunks , parse_trees , is_clause_val, ner_split, tokens,verb_phrases))
        # print(QSG_Rule_3(chunks , parse_trees , is_clause_val, ner_split, tokens,verb_phrases))
        # print(QSG_Rule_4(chunks , parse_trees , is_clause_val, ner_split, tokens,verb_phrases))
        # print(QSG_Rule_5(chunks , parse_trees , is_clause_val, ner_split, tokens,verb_phrases))
        # print(QSG_Rule_6_1(chunks , parse_trees , is_clause_val, ner_split, tokens,verb_phrases))
        # print(QSG_Rule_6_2(chunks , parse_trees , is_clause_val, ner_split, tokens,verb_phrases))
        # print(QSG_Rule_6_3(chunks , parse_trees , is_clause_val, ner_split, tokens,verb_phrases))
        # print(QSG_Rule_7(chunks , parse_trees , is_clause_val, ner_split, tokens,verb_phrases))

        lst = [QSG_Rule_1(chunks , parse_trees , is_clause_val, ner_split, tokens,verb_phrases),
               QSG_Rule_2_1(chunks, parse_trees, is_clause_val, ner_split, tokens, verb_phrases),
               QSG_Rule_2_2(chunks , parse_trees , is_clause_val, ner_split, tokens,verb_phrases),
               QSG_Rule_2_3(chunks , parse_trees , is_clause_val, ner_split, tokens,verb_phrases),
               QSG_Rule_2_4(chunks , parse_trees , is_clause_val, ner_split, tokens,verb_phrases),
               QSG_Rule_3(chunks , parse_trees , is_clause_val, ner_split, tokens,verb_phrases),
               QSG_Rule_4(chunks , parse_trees , is_clause_val, ner_split, tokens,verb_phrases),
               QSG_Rule_5(chunks , parse_trees , is_clause_val, ner_split, tokens,verb_phrases),
               QSG_Rule_6_1(chunks , parse_trees , is_clause_val, ner_split, tokens,verb_phrases),
               QSG_Rule_6_2(chunks , parse_trees , is_clause_val, ner_split, tokens,verb_phrases),
               QSG_Rule_6_3(chunks , parse_trees , is_clause_val, ner_split, tokens,verb_phrases),
               QSG_Rule_7(chunks , parse_trees , is_clause_val, ner_split, tokens,verb_phrases)]



        # In[ ]:

        # q1 = QSG_Rule_1(chunks , parse_trees , is_clause_val, ner_split, tokens,verb_phrases)
        # q2 = QSG_Rule_2_2(chunks , parse_trees , is_clause_val, ner_split, tokens,verb_phrases)
        # q3 = QSG_Rule_2_3(chunks , parse_trees , is_clause_val, ner_split, tokens,verb_phrases)
        # q4 = QSG_Rule_2_4(chunks , parse_trees , is_clause_val, ner_split, tokens,verb_phrases)

        st.subheader('Paragraph: ')

        print(sentence_input)
        st.success(sentence_input)


        for i in lst:
            if len(i) != 0:
                if (len(i[0]['Q']) > 0) and (len(i[0]['A']) > 0):
                    st.subheader('Question: ')
                    st.warning(i[0]['Q'])
                    st.subheader('Answer: ')
                    st.info(i[0]['A'])

        # if st.button(label = 'Generate'):
        #     st.subheader('Question 1: ')
        #     st.warning(q1[0]['Q'])
        #     st.subheader('Answer: ')
        #     st.info(q1[0]['A'])
        #
        #     st.subheader('Question 2: ')
        #     st.warning(q2[0]['Q'])
        #     st.subheader('Answer: ')
        #     st.info(q2[0]['A'])
        #
        #     st.subheader('Question 3: ')
        #     st.warning(q3[0]['Q'])
        #     st.subheader('Answer: ')
        #     st.info(q3[0]['A'])
        #
        #     st.subheader('Question 4: ')
        #     st.warning(q4[0]['Q'])
        #     st.subheader('Answer: ')
        #     st.info(q4[0]['A'])




