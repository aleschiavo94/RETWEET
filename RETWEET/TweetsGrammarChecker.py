import language_check
import csv
import pandas as pd
import time 

# Mention the language keyword 
tool = language_check.LanguageTool('en-US') 
to_be_auto_corrected_file =  open("./Utils/grammar_rules_autocorrect.txt", "r")
to_be_auto_corrected = to_be_auto_corrected_file.read()


def correct_grammar(sentence):
    matches = tool.check(sentence) 

    for mistake in matches: 
        rule_id = mistake.ruleId
        
        if("Samsung" in mistake.replacements):
            return sentence 

        #rules to be corrected automatically
        if(rule_id in to_be_auto_corrected):
            corrected = language_check.correct(sentence, matches) 
            return str(corrected)

    return sentence

